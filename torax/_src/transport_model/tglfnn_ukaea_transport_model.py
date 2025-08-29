import dataclasses
import os

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_config
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(tglf_based_transport_model.DynamicRuntimeParams):
  pass


class TGLFNNukaeaTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):

  def __init__(
      self,
      machine: str | tglfnn_ukaea_config.Machine,
      config_path: str | os.PathLike,
      stats_path: str | os.PathLike,
      efe_gb_pt: str | os.PathLike,
      efi_gb_pt: str | os.PathLike,
      pfi_gb_pt: str | os.PathLike,
  ):
    if isinstance(machine, str):
      self.machine = tglfnn_ukaea_config.Machine(machine)
    else:
      self.machine = machine
    self._config_path = config_path
    self._stats_path = stats_path
    self._efe_gb_pt = efe_gb_pt
    self._efi_gb_pt = efi_gb_pt
    self._pfi_gb_pt = pfi_gb_pt

    self.model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(
            machine=self.machine, config_path=config_path
        ),
        stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(
            machine=self.machine, stats_path=stats_path
        ),
    )
    # TODO: memoization?
    self.model.load_params(
        efe_gb_pt=efe_gb_pt, efi_gb_pt=efi_gb_pt, pfi_gb_pt=pfi_gb_pt
    )
    super().__init__()
    self._frozen = True

  def _prepare_tglfnn_inputs(
      self,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[tglf_based_transport_model.TGLFInputs, jax.Array]:
    tglf_inputs = tglf_based_transport_model._prepare_tglf_inputs(
        transport, geo, core_profiles
    )

    # TODO: double check jax compatability, as these input tensors are different
    # shapes
    match self.machine:
      case tglfnn_ukaea_config.Machine.STEP:
        tglfnn_inputs = self._make_input_tensor_step(tglf_inputs)
      case tglfnn_ukaea_config.Machine.MULTIMACHINE:
        tglfnn_inputs = self._make_input_tensor_multimachine(tglf_inputs)
      case _:
        raise ValueError(f"Unrecognised machine {self.machine}")

    return tglf_inputs, tglfnn_inputs

  def _make_input_tensor_step(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as s_hat in s-alpha geometry.
    s_hat = (tglf_inputs.r_minor / tglf_inputs.q) ** 2 * tglf_inputs.q_prime
    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.S_KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.S_DELTA_LOC,
            tglf_inputs.BETAE,
            tglf_inputs.ZEFF,
        ],
        axis=-1,
    )

  def _make_input_tensor_multimachine(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as s_hat in s-alpha geometry.
    s_hat = (tglf_inputs.r_minor / tglf_inputs.q) ** 2 * tglf_inputs.q_prime

    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.ZEFF,
            tglf_inputs.VEXB_SHEAR,
        ],
        axis=-1,
    )

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: tglf_based_transport_model.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    tglf_inputs, tglfnn_inputs = self._prepare_tglfnn_inputs(
        transport=transport_dynamic_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    predictions = self.model.predict(tglfnn_inputs)

    # TODO: expose variance output
    return self._make_core_transport(
        qi=predictions["efi_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT_IDX],
        qe=predictions["efe_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT_IDX],
        # TODO: TGLFNN outputs pfi, TORAX wants pfe
        pfe=predictions["pfi_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT_IDX],
        quasilinear_inputs=tglf_inputs,
        transport=transport_dynamic_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        # TODO: explain choices here
        gradient_reference_length=1,
        gyrobohm_flux_reference_length=tglf_inputs.r_minor,
    )

  def __hash__(self) -> int:
    return hash((
        self.machine,
        str(self._config_path),
        str(self._stats_path),
        str(self._efe_gb_pt),
        str(self._efi_gb_pt),
        str(self._pfi_gb_pt),
    ))

  def __eq__(self, other) -> bool:
    return (
        self.machine == other.machine
        and self._config_path == other._config_path
        and self._stats_path == other._stats_path
        and self._efe_gb_pt == other._efe_gb_pt
        and self._efi_gb_pt == other._efi_gb_pt
        and self._pfi_gb_pt == other._pfi_gb_pt
    )
