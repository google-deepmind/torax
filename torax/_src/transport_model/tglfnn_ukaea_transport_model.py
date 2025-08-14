from fusion_surrogates.tglfnn_ukaea import config as tglfnn_ukaea_config
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib


class TGLFNNukaeaTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):

  def __init__(
      self,
      config_path: str,
      stats_path: str,
      efe_gb_pt: str,
      efi_gb_pt: str,
      pfi_gb_pt: str,
  ):
    self._config_path = config_path
    self._stats_path = stats_path
    self._efe_gb_pt = efe_gb_pt
    self._efi_gb_pt = efi_gb_pt
    self._pfi_gb_pt = pfi_gb_pt

    self.model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(config_path),
        stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(stats_path),
    )
    self.model.load_params(
        efe_gb_pt=efe_gb_pt, efi_gb_pt=efi_gb_pt, pfi_gb_pt=pfi_gb_pt
    )
    super().__init__()
    self._frozen = True

  def _make_input_tensor(
      self,
      transport,
      geo,
      core_profiles,
  ) -> (tglf_based_transport_model.TGLFInputs, jax.Array):
    tglf_inputs = self._prepare_tglf_inputs(transport, geo, core_profiles)

    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as s_hat in s-alpha geometry.
    s_hat = (tglf_inputs.r_minor / tglf_inputs.q) ** 2 * tglf_inputs.q_prime
    tglfnn_inputs = jnp.stack(
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
    return tglf_inputs, tglfnn_inputs

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: tglf_based_transport_model.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    tglf_inputs, tglfnn_inputs = self._make_input_tensor(
        transport=transport_dynamic_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    predictions = self.model.predict(tglfnn_inputs)

    # TODO: expose variance output
    return self._make_core_transport(
        qi=predictions["efi_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT],
        qe=predictions["efe_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT],
        # TODO: TGLFNN outputs pfi, TORAX wants pfe
        pfe=predictions["pfi_gb"][..., tglfnn_ukaea_config.MEAN_OUTPUT],
        quasilinear_inputs=tglf_inputs,
        transport=transport_dynamic_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        # TODO: explain choices here
        gradient_reference_length=1,
        gyrobohm_flux_reference_length=1,
    )

  def __hash__(self) -> int:
    combined_path = (
        self._config_path
        + self._stats_path
        + self._efe_gb_pt
        + self._efi_gb_pt
        + self._pfi_gb_pt
    )
    return hash(combined_path)

  def __eq__(self, other) -> bool:
    return (
        self._config_path == other._config_path
        and self._stats_path == other._stats_path
        and self._efe_gb_pt == other._efe_gb_pt
        and self._efi_gb_pt == other._efi_gb_pt
        and self._pfi_gb_pt == other._pfi_gb_pt
    )
