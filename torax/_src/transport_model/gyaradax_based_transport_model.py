"""Base class and utils for gyaradax-based transport models.

Same pattern as `qualikiz_based_transport_model` and `tglf_based_transport_model`
in TORAX: a `RuntimeParams` dataclass, a shared `GyaradaxBasedTransportModel`
parent that subclasses `QuasilinearTransportModel`, and a `_prepare_gyaradax_inputs`
factory that builds the per-face physics inputs (drives + geometry) for
gyaradax's linear solver.

The subclass (`GyaradaxQLTransportModel`) implements
`_per_radius(params, geom) -> (qi, qe, pfe)` and inherits the vmap-over-rho_match
+ interp-onto-face-grid call_implementation.
"""

import abc
import dataclasses
import math
from functools import lru_cache
from typing import Any, Dict, Tuple

from gyaradax.geometry import build_topology
from gyaradax.geometry import compute_continuous_geometry
from gyaradax.params import GKParams
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_transport_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib
from torax._src.transport_model.quasilinear_transport_model import calculate_chiGB
from torax._src.transport_model.quasilinear_transport_model import NormalizedLogarithmicGradients
from torax._src.transport_model.quasilinear_transport_model import QuasilinearInputs
from torax._src.transport_model.quasilinear_transport_model import QuasilinearTransportModel

# safe-operating clip ranges (same philosophy as QLKNN clip_inputs)
_RLT_MIN, _RLT_MAX = 0.0, 30.0
_RLN_MIN, _RLN_MAX = -15.0, 15.0
_Q_MIN, _Q_MAX = 0.5, 10.0
_SHAT_MIN, _SHAT_MAX = -3.0, 6.0
_EPS_MIN, _EPS_MAX = 0.02, 0.5

# gyaradax geometry / linear-solve defaults
_GKPARAMS_DT = 0.005
_VPAR_MAX = 3.0
_KRHOMAX = 1.4
_RREF = 100.0
_SIGNB = 1.0
_NPERIOD = 1
_KXMAX = 0.0
_GEOM_TYPE = "circ"

# initial-df seed amplitude (cosine along s, every non-zonal ky)
_DF_SEED_AMPLITUDE = 1e-3

# gyaradax gyrobohm flux unit is 2*sqrt(2) larger than TORAX
_GYARADAX_GB_FLUX_FACTOR = 2.0 * math.sqrt(2.0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  """Runtime parameters shared by all gyaradax-based transport models."""


@lru_cache(maxsize=8)
def _get_topology_cached(nkx: int, nky: int, ikxspace: int, ns: int):
  """Topology dict keyed on static grid sizes (cached across `from_config`)."""
  return build_topology(nkx=nkx, nky=nky, ikxspace=ikxspace, ns=ns)


def build_quasilinear_inputs(core_profiles, geo) -> QuasilinearInputs:
  """Build TORAX's QuasilinearInputs on the full face grid."""
  # gyaradax normalizes the gyrobohm flux to the MAJOR radius
  chi_gb = calculate_chiGB(
      reference_temperature=core_profiles.T_i.face_value(),
      reference_magnetic_field=geo.B_0,
      reference_mass=core_profiles.A_i,
      reference_length=geo.R_major,
  )
  log_grads = NormalizedLogarithmicGradients.from_profiles(
      core_profiles=core_profiles,
      radial_coordinate=geo.r_mid,
      radial_face_coordinate=geo.r_mid_face,
      reference_length=geo.R_major,
  )
  return QuasilinearInputs(
      chiGB=chi_gb,
      Rmaj=geo.R_major,
      Rmin=geo.a_minor,
      lref_over_lti=log_grads.lref_over_lti,
      lref_over_lte=log_grads.lref_over_lte,
      lref_over_lne=log_grads.lref_over_lne,
      lref_over_lni0=log_grads.lref_over_lni0,
      lref_over_lni1=log_grads.lref_over_lni1,
  )


def face_indices_for_radii(geo, rho_match: Tuple[float, ...]) -> jnp.ndarray:
  """Pick the face index closest to each rho_match value. Shape (K,)."""
  rho_face = geo.rho_face_norm
  return jnp.argmin(
      jnp.abs(rho_face[:, None] - jnp.asarray(rho_match)[None, :]), axis=0
  )


def gkparams_for_radius(
    rho_idx,
    ql_inputs: QuasilinearInputs,
    core_profiles,
    geo,
    config,
) -> GKParams:
  """Build a GKParams instance for a single flux-tube radius."""
  rlt = jnp.clip(ql_inputs.lref_over_lti[rho_idx], _RLT_MIN, _RLT_MAX)
  rln = jnp.clip(ql_inputs.lref_over_lne[rho_idx], _RLN_MIN, _RLN_MAX)
  q = jnp.clip(core_profiles.q_face[rho_idx], _Q_MIN, _Q_MAX)
  smag_face = psi_calculations.calc_s_rmid(geo, core_profiles.psi)
  shat = jnp.clip(smag_face[rho_idx], _SHAT_MIN, _SHAT_MAX)
  eps = jnp.clip(geo.epsilon_face[rho_idx], _EPS_MIN, _EPS_MAX)
  beta = jnp.asarray(0.0)
  backend = getattr(config, "backend", "jax")
  return GKParams(
      rlt=rlt,
      rln=rln,
      q=q,
      shat=shat,
      eps=eps,
      beta=beta,
      adiabatic_electrons=True,
      non_linear=False,  # subclass may flip via dataclasses.replace
      disable_per_ky_norm=True,  # subclass may flip via dataclasses.replace
      dt=_GKPARAMS_DT,
      backend=backend,
  )


def gyaradax_geometry_at(
    q, shat, eps, config, topology: Dict[str, Any]
) -> Dict[str, Any]:
  """Build a gyaradax geometry dict at one radius (jit/AD safe over q,shat,eps)."""
  return compute_continuous_geometry(
      q=q,
      shat=shat,
      eps=eps,
      ns=config.ns,
      nkx=config.nkx,
      nky=config.nky,
      nvpar=config.nvpar,
      nmu=config.nmu,
      vpar_max=_VPAR_MAX,
      nperiod=_NPERIOD,
      kxmax=_KXMAX,
      krhomax=_KRHOMAX,
      ikxspace=config.ikxspace,
      signB=_SIGNB,
      Rref=_RREF,
      geom_type=_GEOM_TYPE,
      topology=topology,
  )


def precompute_topology(config) -> Dict[str, Any]:
  """Build the static topology dict from grid sizes."""
  return _get_topology_cached(
      config.nkx, config.nky, config.ikxspace, config.ns
  )


def initial_df(config) -> jnp.ndarray:
  """Initial df: cosine in s, every non-zonal ky seeded. Bypasses init_f's .item()."""
  nv, nmu, ns, nkx, nky = (
      config.nvpar,
      config.nmu,
      config.ns,
      config.nkx,
      config.nky,
  )
  s_grid = (jnp.arange(ns) + 0.5) / ns - 0.5
  seed_s = _DF_SEED_AMPLITUDE * (jnp.cos(2 * jnp.pi * s_grid) + 1.0)
  df = jnp.zeros((nv, nmu, ns, nkx, nky), dtype=jnp.complex128)
  seed = jnp.broadcast_to(
      seed_s[None, None, :, None, None] / (nkx * (nky - 1)),
      (nv, nmu, ns, nkx, nky),
  )
  ky_mask = jnp.arange(nky) > 0
  return df + seed * ky_mask[None, None, None, None, :]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxBasedTransportModel(QuasilinearTransportModel, abc.ABC):
  """Shared frozen dataclass + call_implementation for all gyaradax models."""

  rho_match: Tuple[float, ...] = (0.35, 0.55, 0.75, 0.875)
  backend: str = "jax"
  nvpar: int = 32
  nmu: int = 8
  ns: int = 16
  nkx: int = 43
  nky: int = 16
  ikxspace: int = 5

  @property
  def topology(self):
    return _get_topology_cached(self.nkx, self.nky, self.ikxspace, self.ns)

  def _initial_df(self) -> jnp.ndarray:
    return initial_df(self)

  @abc.abstractmethod
  def _per_radius(
      self, params: GKParams, geom: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (qi, qe, pfe) at one radius in gyroBohm units."""

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    del pedestal_model_output, runtime_params

    ql_inputs = build_quasilinear_inputs(core_profiles, geo)
    match_idx = face_indices_for_radii(geo, self.rho_match)

    def per_radius(idx):
      params = gkparams_for_radius(idx, ql_inputs, core_profiles, geo, self)
      geom = gyaradax_geometry_at(
          q=params.q,
          shat=params.shat,
          eps=params.eps,
          config=self,
          topology=self.topology,
      )
      return self._per_radius(params, geom)

    qi_m, qe_m, pfe_m = jax.vmap(per_radius)(match_idx)

    rho_face = geo.rho_face_norm
    rho_match_arr = jnp.asarray(self.rho_match)
    qi_face = jnp.interp(rho_face, rho_match_arr, qi_m)
    qe_face = jnp.interp(rho_face, rho_match_arr, qe_m)
    pfe_face = jnp.interp(rho_face, rho_match_arr, pfe_m)

    qi_face = qi_face * _GYARADAX_GB_FLUX_FACTOR
    qe_face = qe_face * _GYARADAX_GB_FLUX_FACTOR
    pfe_face = pfe_face * _GYARADAX_GB_FLUX_FACTOR

    return self._make_core_transport(
        qi=qi_face,
        qe=qe_face,
        pfe=pfe_face,
        quasilinear_inputs=ql_inputs,
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.R_major,
        gyrobohm_flux_reference_length=geo.R_major,
    )
