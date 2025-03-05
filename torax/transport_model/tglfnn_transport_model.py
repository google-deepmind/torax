import dataclasses
import json
from pathlib import Path
from typing import Callable, Final, Mapping

import immutabledict
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import linen as nn

from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import tglf_based_transport_model, transport_model
from torax.transport_model.tglf_based_transport_model import RuntimeParams, TGLFInputs

_ACTIVATION_FNS: Final[Mapping[str, Callable[[jax.Array], jax.Array]]] = (
    immutabledict.immutabledict({
        "relu": nn.relu,
        "tanh": nn.tanh,
        "sigmoid": nn.sigmoid,
    })
)


def normalize(
    data: jax.Array, *, mean: jax.Array, stddev: jax.Array
) -> jax.Array:
  """Normalizes data to have mean 0 and stddev 1."""
  return (data - mean) / jnp.where(stddev == 0, 1, stddev)


def unnormalize(
    data: jax.Array, *, mean: jax.Array, stddev: jax.Array
) -> jax.Array:
  """Unnormalizes data to the orginal distribution."""
  return data * jnp.where(stddev == 0, 1, stddev) + mean


INPUT_LABELS: Final[list[str]] = [
    "RLNS_1",
    "RLTS_1",
    "RLTS_2",
    "TAUS_2",
    "RMIN_LOC",
    "DRMAJDX_LOC",
    "Q_LOC",
    "SHAT",
    "XNUE",
    "KAPPA_LOC",
    "S_KAPPA_LOC",
    "DELTA_LOC",
    "S_DELTA_LOC",
    "BETAE",
    "ZEFF",
]
OUTPUT_LABELS: Final[list[str]] = ["efe_gb", "efi_gb", "pfi_gb"]


@dataclasses.dataclass
class TGLFNNModelConfig:
  n_ensemble: int
  hidden_size: int
  n_hidden_layers: int
  dropout: float
  scale: bool
  denormalise: bool

  @classmethod
  def load(cls, config_path: str) -> "TGLFNNModelConfig":
    with open(config_path, "r") as f:
      config = yaml.safe_load(f)

    return cls(
        n_ensemble=config["num_estimators"],
        hidden_size=512,
        n_hidden_layers=config["model_size"],
        dropout=config["dropout"],
        scale=config["scale"],
        denormalise=config["denormalise"],
    )


@dataclasses.dataclass
class TGLFNNModelStats:
  input_mean: jax.Array
  input_std: jax.Array
  output_mean: jax.Array
  output_std: jax.Array

  @classmethod
  def load(cls, stats_path: str) -> "TGLFNNModelStats":
    with open(stats_path, "r") as f:
      stats = json.load(f)

    return cls(
        input_mean=jnp.array([stats[label]["mean"] for label in INPUT_LABELS]),
        input_std=jnp.array([stats[label]["std"] for label in INPUT_LABELS]),
        output_mean=jnp.array(
            [stats[label]["mean"] for label in OUTPUT_LABELS]
        ),
        output_std=jnp.array([stats[label]["std"] for label in OUTPUT_LABELS]),
    )


class GaussianMLP(nn.Module):
  """An MLP with dropout, outputting a mean and variance."""

  num_hiddens: int
  hidden_size: int
  dropout: float
  activation: str

  @nn.compact
  def __call__(
      self,
      x,
      deterministic: bool = False,
  ):
    for _ in range(self.num_hiddens - 1):
      x = nn.Dense(self.hidden_size)(x)
      x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
      x = _ACTIVATION_FNS[self.activation](x)
    mean_and_var = nn.Dense(2)(x)
    mean = mean_and_var[..., 0]
    var = mean_and_var[..., 1]
    var = nn.softplus(var)

    return jnp.stack([mean, var], axis=-1)


class GaussianMLPEnsemble(nn.Module):
  """An ensemble of GaussianMLPs."""

  n_ensemble: int
  num_hiddens: int
  hidden_size: int
  dropout: float
  activation: str

  @nn.compact
  def __call__(
      self,
      x,
      deterministic: bool = False,
  ):
    ensemble_output = jnp.stack(
        [
            GaussianMLP(
                self.num_hiddens,
                self.hidden_size,
                self.dropout,
                self.activation,
            )(x, deterministic=deterministic)
            for _ in range(self.n_ensemble)
        ],
        axis=0,
    )
    mean = jnp.mean(ensemble_output[..., 0], axis=0)
    aleatoric = jnp.mean(ensemble_output[..., 1], axis=0)
    epistemic = jnp.var(ensemble_output[..., 0], axis=0)
    return jnp.stack([mean, aleatoric + epistemic], axis=-1)


class TGLFNNModel:

  def __init__(
      self,
      config: TGLFNNModelConfig,
      stats: TGLFNNModelStats,
      params: optax.Params | None,
  ):
    self.config = config
    self.stats = stats
    self.params = params
    self.network = GaussianMLPEnsemble(
        n_ensemble=config.n_ensemble,
        hidden_size=config.hidden_size,
        num_hiddens=config.n_hidden_layers,
        dropout=config.dropout,
        activation="relu",
    )

  @classmethod
  def load_from_pytorch(
      cls,
      config_path: str,
      stats_path: str,
      efe_gb_checkpoint_path: str,
      efi_gb_checkpoint_path: str,
      pfi_gb_checkpoint_path: str,
      *args,
      **kwargs,
  ) -> "TGLFNNModel":
    import torch

    def _convert_pytorch_state_dict(
        pytorch_state_dict: dict, config: TGLFNNModelConfig
    ) -> optax.Params:
      params = {}
      for i in range(config.n_ensemble):
        model_dict = {}
        for j in range(config.n_hidden_layers):
          layer_dict = {
              "kernel": jnp.array(
                  pytorch_state_dict[f"models.{i}.model.{j * 3}.weight"]
              ).T,
              "bias": jnp.array(
                  pytorch_state_dict[f"models.{i}.model.{j * 3}.bias"]
              ).T,
          }
          model_dict[f"Dense_{j}"] = layer_dict
        params[f"GaussianMLP_{i}"] = model_dict
      return {"params": params}

    config = TGLFNNModelConfig.load(config_path)
    stats = TGLFNNModelStats.load(stats_path)

    with open(efe_gb_checkpoint_path, "rb") as f:
      efe_gb_params = _convert_pytorch_state_dict(
          torch.load(f, *args, **kwargs), config
      )
    with open(efi_gb_checkpoint_path, "rb") as f:
      efi_gb_params = _convert_pytorch_state_dict(
          torch.load(f, *args, **kwargs), config
      )
    with open(pfi_gb_checkpoint_path, "rb") as f:
      pfi_gb_params = _convert_pytorch_state_dict(
          torch.load(f, *args, **kwargs), config
      )

    params = {
        "efe_gb": efe_gb_params,
        "efi_gb": efi_gb_params,
        "pfi_gb": pfi_gb_params,
    }

    return cls(config, stats, params)

  def predict(
      self,
      inputs: jax.Array,
  ) -> dict[str, jax.Array]:
    if self.config.scale:
      inputs = normalize(
          inputs, mean=self.stats.input_mean, stddev=self.stats.input_std
      )

    output = jnp.stack(
        [
            self.network.apply(self.params[label], inputs, deterministic=True)
            for label in OUTPUT_LABELS
        ],
        axis=-2,
    )

    if self.config.denormalise:
      output = unnormalize(
          output, mean=self.stats.output_mean, stddev=self.stats.output_std
      )

    return output


class TGLFNNTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculate turbulent transport coefficients using a TGLF surrogate model."""

  def __init__(
      self,
      config_path: str,
      stats_path: str,
      efe_gb_checkpoint_path: str,
      efi_gb_checkpoint_path: str,
      pfi_gb_checkpoint_path: str,
  ):
    super().__init__()

    self.model = TGLFNNModel.load_from_pytorch(
        config_path,
        stats_path,
        efe_gb_checkpoint_path,
        efi_gb_checkpoint_path,
        pfi_gb_checkpoint_path,
    )
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    tglf_inputs = self._prepare_tglf_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        geo=geo,
        core_profiles=core_profiles,
    )
    # Broadcast scalar inputs to the face grid
    tglf_inputs_array = jnp.stack(
        [
            jnp.broadcast_to(getattr(tglf_inputs, i), geo.rho_face.shape)
            for i in self.inputs
        ],
        axis=-1,
    )

    output = self.model.predict(tglf_inputs_array)

    # Curently, we just use the mean prediction and discard the variance
    return self._make_core_transport(
        qi=output[..., 1, 0],  # TODO: Get the ordering from OUTPUT_LABELS
        qe=output[..., 0, 0],
        pfe=output[..., 2, 0],
        quasilinear_inputs=tglf_inputs,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.Rmin,  # Device minor radius at LCFS
        gyrobohm_flux_reference_length=geo.Rmin,  # TODO: Check
    )


@dataclasses.dataclass(kw_only=True)
class TGLFNNTransportModelBuilder(transport_model.TransportModelBuilder):
  """When called, instantiates a TGLFSurrogateTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )
  model_dir: str = "~/tglfnn"
  model_version: str = "1.0.0"

  def __call__(
      self,
  ) -> TGLFNNTransportModel:
    model_dir_with_version = Path(self.model_dir) / self.model_version
    return TGLFNNTransportModel(
        config_path=model_dir_with_version / "config.yaml",
        stats_path=model_dir_with_version / "stats.json",
        efe_gb_checkpoint_path=model_dir_with_version / "regressor_efe_gb.pt",
        efi_gb_checkpoint_path=model_dir_with_version / "regressor_efi_gb.pt",
        pfi_gb_checkpoint_path=model_dir_with_version / "regressor_pfi_gb.pt",
    )
