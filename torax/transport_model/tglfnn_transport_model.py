import json
from copy import deepcopy
from pathlib import Path
import dataclasses
import chex
import jax.numpy as jnp
from flax import linen as nn
import jax
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import tglf_based_transport_model
from warnings import warn
from torax.transport_model.tglf_based_transport_model import TGLFInputs, RuntimeParams
from torax.transport_model import transport_model
from typing import Callable


class TGLFNNSurrogate(nn.Module):
  """A simple MLP with i/o scaling, dropout, ReLU activation, outputting mean and variance."""

  hidden_dimension: int
  n_hidden_layers: int
  dropout: float
  input_means: jax.Array
  input_stds: jax.Array
  output_mean: float
  output_std: float

  @nn.compact
  def __call__(
      self,
      x,
      deterministic: bool = False,
  ):
    # Rescale inputs
    x = (x - self.input_means) / self.input_stds

    # MLP
    x = nn.Dense(self.hidden_dimension)(x)
    x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
    x = nn.relu(x)
    for _ in range(self.n_hidden_layers):
      x = nn.Dense(self.hidden_dimension)(x)
      x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
      x = nn.relu(x)
    mean_and_var = nn.Dense(2)(x)
    mean = mean_and_var[..., 0]
    var = mean_and_var[..., 1]
    var = nn.softplus(var)

    # Rescale outputs
    rescaled_mean = mean * self.output_std + self.output_mean
    rescaled_var = var * self.output_std**2

    return jnp.stack([rescaled_mean, rescaled_var], axis=-1)


class EnsembleTGLFNNSurrogate(nn.Module):
  """An ensemble of TGLFNNSurrogate models."""

  input_means: jax.Array
  input_stds: jax.Array
  output_mean: float
  output_std: float
  n_models: int = 5
  hidden_dimension: int = 512
  n_hidden_layers: int = 4
  dropout: float = 0.05

  def setup(
      self,
  ):
    self.models = [
        TGLFNNSurrogate(
            hidden_dimension=self.hidden_dimension,
            n_hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            input_means=self.input_means,
            input_stds=self.input_stds,
            output_mean=self.output_mean,
            output_std=self.output_std,
        )
        for i in range(self.n_models)
    ]

  def __call__(self, x, *args, **kwargs):
    # Shape is batch size x 2 x n_models
    outputs = jnp.stack(
        [model(x, *args, **kwargs) for model in self.models], axis=-1
    )
    # Shape is batch_size
    mean = jnp.mean(outputs[:, 0, :], axis=-1)
    aleatoric_uncertainty = jnp.mean(outputs[:, 1, :], axis=-1)
    epistemic_uncertainty = jnp.var(outputs[:, 0, :], axis=-1)
    return jnp.stack(
        [mean, aleatoric_uncertainty + epistemic_uncertainty], axis=-1
    )

  def get_params_from_pytorch_state_dict(self, pytorch_state_dict: dict):
    params = {}
    for i in range(self.n_models):
      model_dict = {}
      for j in range(
          self.n_hidden_layers + 2
      ):  # +2 for input and output layers
        # j*3 to skip dropout and activation
        layer_dict = {
            "kernel": jnp.array(
                pytorch_state_dict[f"models.{i}.model.{j*3}.weight"]
            ).T,
            "bias": jnp.array(
                pytorch_state_dict[f"models.{i}.model.{j*3}.bias"]
            ).T,
        }
        model_dict[f"Dense_{j}"] = layer_dict
      params[f"models_{i}"] = model_dict

    return params


class TGLFNNTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculate turbulent transport coefficients using a TGLF surrogate model."""

  def __init__(
      self,
      path_to_model_weights_json: str | Path,
      path_to_model_scaling_json: str | Path,
  ):
    super().__init__()

    # NN surrogate expects inputs to be stacked in this order
    self.inputs = [
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

    with open(path_to_model_scaling_json, "r") as f:
      scaling_dict = json.load(f)
      input_means = jnp.stack(
          [scaling_dict[i]["mean"] for i in self.inputs],
          axis=-1,
      )
      input_stds = jnp.stack(
          [scaling_dict[i]["std"] for i in self.inputs],
          axis=-1,
      )

    # TODO: In future, TGLFNN models will be saved in distinct files, so we will
    # have to load separate state dicts for each model. For the time being, we
    # only load once, in order to to speed things up
    with open(path_to_model_weights_json, "r") as f:
      state_dict = json.load(f)

    self.models = {}
    self.params = {}
    for model in ["efi_gb", "efe_gb", "pfi_gb"]:
      self.models[model] = EnsembleTGLFNNSurrogate(
          input_means=input_means,
          input_stds=input_stds,
          # TODO: Load output scaling from scaling_dict, once it is saved there
          output_mean=0,
          output_std=1,
          n_models=state_dict["num_estimators"],
          # TODO: Load hidden_dimension from state_dict, once it is saved there
          hidden_dimension=512,
          # Subtract 2 from model size to account for input and output layers
          n_hidden_layers=state_dict["model_size"] - 2,
          dropout=state_dict["dropout"],
      )
      self.params[model] = self.models[
          model
      ].get_params_from_pytorch_state_dict(state_dict[f"regressor_{model}"])

    warn(
        "TGLFSurrogateTransportModel currently operates with the assumption"
        " that electron particle flux = ion particle flux. This might produce"
        " incorrect results for non-Deuterium plasmas."
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
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
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

    # Apply the model
    efi_gb_mean_and_var = self.models["efi_gb"].apply(
        {"params": self.params["efi_gb"]}, tglf_inputs_array, deterministic=True
    )
    efe_gb_mean_and_var = self.models["efe_gb"].apply(
        {"params": self.params["efe_gb"]}, tglf_inputs_array, deterministic=True
    )
    pfi_gb_mean_and_var = self.models["pfi_gb"].apply(
        {"params": self.params["pfi_gb"]}, tglf_inputs_array, deterministic=True
    )

    # Curently, we just use the mean prediction and discard the variance
    return self._make_core_transport(
        qi=efi_gb_mean_and_var[:, 0],
        qe=efe_gb_mean_and_var[:, 0],
        pfe=pfi_gb_mean_and_var[:, 0],
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
  weights_path: str = (
      "/home/theo/documents/ukaea/torax/tglfnn/1.0.0/tglfnn_checkpoint.json"
  )
  scaling_path: str = "/home/theo/documents/ukaea/torax/tglfnn/1.0.0/stats.json"

  def __call__(
      self,
  ) -> TGLFNNTransportModel:
    return TGLFNNTransportModel(
        path_to_model_weights_json=(self.weights_path),
        path_to_model_scaling_json=(self.scaling_path),
    )
