# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QL gyaradax as a TORAX transport model.

Per rho_match radius: map TORAX profiles/geometry to gyaradax inputs, run
`gyaradax.quasilinear.point_eval.ql_at_point` (linear solve + saturation rule
+ Cn head), interpolate onto the face grid, and convert GKW gyroBohm units to
TORAX's (see gyaradax_normalization). The velocity/wavenumber grid is adopted
from the Cn calibration head's metadata unless set explicitly in the config.
"""

import dataclasses
from functools import lru_cache
import warnings
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import chex
from gyaradax.geometry import build_topology
from gyaradax.geometry import compute_continuous_geometry
from gyaradax.params import GKParams
from gyaradax.quasilinear import point_eval
from gyaradax.quasilinear.models import load_cn_payload
from gyaradax.quasilinear.models import select_cn_head
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import gyaradax_diagnostics as diag_lib
from torax._src.transport_model import gyaradax_normalization as gb_norm
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import quasilinear_transport_model
from torax._src.transport_model import register_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model.quasilinear_transport_model import QuasilinearInputs

# safe-operating clip ranges (same philosophy as QLKNN clip_inputs)
_RLT_MIN, _RLT_MAX = 0.0, 30.0
_RLN_MIN, _RLN_MAX = -15.0, 15.0
_Q_MIN, _Q_MAX = 0.5, 10.0
_SHAT_MIN, _SHAT_MAX = -3.0, 6.0
_EPS_MIN, _EPS_MAX = 0.02, 0.5
_BETA_MAX = 0.05

# gyaradax geometry / linear-solve defaults
_GKPARAMS_DT = 0.005
_VPAR_MAX = 3.0
_RREF = 100.0
_NPERIOD = 1
_GEOM_TYPE = "circ"

_GRID_KEYS = ("nvpar", "nmu", "ns", "nkx", "nky", "ikxspace", "krhomax")
# perpendicular resolution floor: 43x16 under-resolves the spectrum (ion flux
# within ~15% but the electron channel off by ~55%), so never run below this
_GRID_FLOOR = {"nkx": 85, "nky": 32}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  """Runtime parameters for the gyaradax-QL transport model."""


@lru_cache(maxsize=8)
def _get_topology_cached(nkx: int, nky: int, ikxspace: int, ns: int):
  """Topology dict keyed on static grid sizes (cached across `from_config`)."""
  return build_topology(nkx=nkx, nky=nky, ikxspace=ikxspace, ns=ns)


def _resolve_grid(cfg) -> dict:
  """Grid resolution: explicit config wins, else the head's calibration grid."""
  payload = load_cn_payload(cfg.cn_calibration_path or "")
  head_grid = payload.get("grid", {}) if isinstance(payload, dict) else {}
  grid = {
      key: (
          getattr(cfg, key, None)
          if getattr(cfg, key, None) is not None
          else head_grid.get(key)
      )
      for key in _GRID_KEYS
  }
  missing = [key for key, value in grid.items() if value is None]
  if missing:
    raise ValueError(
        f"gyaradax-ql grid underspecified ({missing}): the selected Cn head "
        f"('{cfg.cn_calibration_path}') carries no grid metadata — set these "
        "config fields explicitly or use a head fit with grid metadata."
    )
  for key, floor in _GRID_FLOOR.items():
    if int(grid[key]) < floor:
      warnings.warn(
          f"gyaradax: {key}={grid[key]} is below the {floor} resolution floor"
          " (source:"
          f" {'config' if getattr(cfg, key, None) else 'calibration head'});"
          f" raising to {floor}. Heads calibrated at a coarser grid should be"
          " refit.",
          RuntimeWarning,
          stacklevel=2,
      )
      grid[key] = floor
  return grid


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
  # jnp.asarray: epsilon_face may be numpy; keeps vmap tracer indexing legal
  eps = jnp.clip(jnp.asarray(geo.epsilon_face)[rho_idx], _EPS_MIN, _EPS_MAX)
  em = bool(getattr(config, "em", False))
  if em:
    # EXPERIMENTAL: GKW beta_ref = 2 mu0 n_e T_i / B_0^2; Cn head is ES-only
    n_e = core_profiles.n_e.face_value()[rho_idx]
    t_i_j = (
        core_profiles.T_i.face_value()[rho_idx] * constants.CONSTANTS.keV_to_J
    )
    beta = jnp.clip(
        2.0 * constants.CONSTANTS.mu_0 * n_e * t_i_j / geo.B_0**2,
        0.0,
        _BETA_MAX,
    )
  else:
    beta = jnp.asarray(0.0)
  return GKParams(
      rlt=rlt,
      rln=rln,
      q=q,
      shat=shat,
      eps=eps,
      beta=beta,
      nlapar=em,
      adiabatic_electrons=True,
      non_linear=False,
      disable_per_ky_norm=True,
      dt=_GKPARAMS_DT,
      backend=getattr(config, "backend", "jax"),
      mixed_precision=getattr(config, "mixed_precision", True),
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
      kxmax=0.0,
      krhomax=config.krhomax,
      ikxspace=config.ikxspace,
      signB=1.0,
      Rref=_RREF,
      geom_type=_GEOM_TYPE,
      topology=topology,
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxQLTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """QL gyaradax transport model."""

  rho_match: Tuple[float, ...] = (0.35, 0.55, 0.75, 0.875)
  backend: str = "jax"
  mixed_precision: bool = True
  nvpar: int = 32
  nmu: int = 8
  ns: int = 16
  nkx: int = 85
  nky: int = 32
  ikxspace: int = 5
  krhomax: float = 1.4
  n_steps_linear: int = 2000
  rule: str = "canonical"
  cn_calibration_path: str = "auto"
  cn_scalar: float = 1.0
  cn_override: float = -1.0
  early_stop: bool = True
  early_stop_block: int = 100
  early_stop_atol: float = 1e-4
  early_stop_rtol: float = 1e-3
  early_stop_min_steps: int = 200
  early_stop_patience: int = 2
  # EXPERIMENTAL: beta + A_par flutter; needs a beta-aware Cn calibration
  em: bool = False
  diagnostics_path: str = ""

  @classmethod
  def from_config(cls, cfg) -> "GyaradaxQLTransportModel":
    fis = getattr(cfg, "fast_ion_stabilization", None)
    # raise on nonzero fast_ion_stabilization instead of silently ignoring it
    if fis is not None and float(fis.get_value(0.0)) != 0.0:
      raise NotImplementedError(
          "gyaradax-ql does not implement fast_ion_stabilization."
      )
    grid = _resolve_grid(cfg)
    # warm caches outside jit: build_topology arrays must not become tracers
    _get_topology_cached(
        int(grid["nkx"]),
        int(grid["nky"]),
        int(grid["ikxspace"]),
        int(grid["ns"]),
    )
    return cls(
        rho_match=tuple(cfg.rho_match),
        backend=cfg.backend,
        mixed_precision=cfg.mixed_precision,
        n_steps_linear=cfg.n_steps_linear,
        rule=cfg.rule,
        cn_override=(-1.0 if cfg.cn_override is None else float(cfg.cn_override)),
        nvpar=int(grid["nvpar"]),
        nmu=int(grid["nmu"]),
        ns=int(grid["ns"]),
        nkx=int(grid["nkx"]),
        nky=int(grid["nky"]),
        ikxspace=int(grid["ikxspace"]),
        krhomax=float(grid["krhomax"]),
        cn_calibration_path=cfg.cn_calibration_path or "",
        early_stop=cfg.early_stop,
        early_stop_block=cfg.early_stop_block,
        early_stop_atol=cfg.early_stop_atol,
        early_stop_rtol=cfg.early_stop_rtol,
        early_stop_min_steps=cfg.early_stop_min_steps,
        early_stop_patience=cfg.early_stop_patience,
        em=getattr(cfg, "em", False),
        diagnostics_path=getattr(cfg, "diagnostics_path", None) or "",
    )

  @property
  def topology(self):
    return _get_topology_cached(self.nkx, self.nky, self.ikxspace, self.ns)

  @property
  def sink(self) -> Any:
    """Host-side diagnostics sink, or None when nothing is being recorded."""
    return diag_lib.get_sink(
        self.diagnostics_path,
        getattr(self, "latent_dump_dir", ""),
        len(self.rho_match),
        getattr(self, "latent_dump_max_calls", 0),
        getattr(self, "latent_dump_decoded_max_calls", 0),
    )

  @property
  def cn_head(self):
    return select_cn_head(load_cn_payload(self.cn_calibration_path))

  def _initial_df(self) -> jnp.ndarray:
    return point_eval.initial_df(
        self.nvpar, self.nmu, self.ns, self.nkx, self.nky
    )

  def _cn(self, params: GKParams):
    """Cn from the head: scalar directly, parametric via the feature vector."""
    if self.cn_override >= 0.0:
      return jnp.asarray(self.cn_override)
    head = self.cn_head
    if head is None:
      return jnp.asarray(self.cn_scalar)
    if hasattr(head, "cn_jax"):
      features = jnp.array([[
          params.rlt,
          params.rln,
          params.rlt,
          params.rln,
          params.shat,
          params.q,
          params.eps,
          params.beta,
      ]])
      return head.cn_jax(features)[0]
    return jnp.asarray(head)

  def _per_radius_from_profiles(
      self, rho_idx, ql_inputs, core_profiles, geo
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Default path: local GKParams + geometry, then the solver-based _per_radius."""
    params = gkparams_for_radius(rho_idx, ql_inputs, core_profiles, geo, self)
    geom = gyaradax_geometry_at(
        q=params.q,
        shat=params.shat,
        eps=params.eps,
        config=self,
        topology=self.topology,
    )
    q_i, q_e, pfe, extra = self._per_radius(params, geom)
    diag_lib.record(
        self.sink,
        {"model": type(self).__name__},
        {
            "rho_idx": rho_idx,
            "rho_face": jnp.asarray(geo.rho_face_norm)[rho_idx],
            "rlt": params.rlt,
            "rln": params.rln,
            "q": params.q,
            "shat": params.shat,
            "eps": params.eps,
            "beta": params.beta,
            "qi_gb": q_i,
            "qe_gb": q_e,
            "pfe_gb": pfe,
            **extra,
        },
    )
    return q_i, q_e, pfe

  def _per_radius(
      self, params: GKParams, geom: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """(qi, qe, pfe) in GKW gyroBohm plus traced per-call solver diagnostics."""
    return point_eval.ql_at_point(
        params,
        geom,
        (self.nvpar, self.nmu, self.ns, self.nkx, self.nky),
        cn=self._cn(params),
        rule=self.rule,
        n_steps_linear=self.n_steps_linear,
        early_stop=self.early_stop,
        early_stop_opts=dict(
            block=self.early_stop_block,
            min_steps=self.early_stop_min_steps,
            atol=self.early_stop_atol,
            rtol=self.early_stop_rtol,
            patience=self.early_stop_patience,
        ),
        return_diagnostics=True,
    )

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    del runtime_params

    ql_inputs = gb_norm.build_quasilinear_inputs(core_profiles, geo)
    match_idx = face_indices_for_radii(geo, self.rho_match)
    sink = self.sink

    def per_radius(idx):
      idx = diag_lib.mark_begin(sink, idx)
      return self._per_radius_from_profiles(idx, ql_inputs, core_profiles, geo)

    qi_m, qe_m, pfe_m = jax.vmap(per_radius)(match_idx)

    rho_face = geo.rho_face_norm
    rho_match_arr = jnp.asarray(self.rho_match)
    # GKW gyroBohm fluxes -> TORAX QL convention (gyaradax_normalization)
    gb_factor = gb_norm.gb_flux_conversion_factor(
        gb_norm.GYARADAX, gb_norm.TORAX_QL, geo
    )
    qi_face = jnp.interp(rho_face, rho_match_arr, qi_m) * gb_factor
    qe_face = jnp.interp(rho_face, rho_match_arr, qe_m) * gb_factor
    pfe_face = jnp.interp(rho_face, rho_match_arr, pfe_m) * gb_factor

    return self._make_core_transport(
        qi=qi_face,
        qe=qe_face,
        pfe=pfe_face,
        quasilinear_inputs=ql_inputs,
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=gb_norm.gradient_reference_length(geo),
        gyrobohm_flux_reference_length=gb_norm.flux_reference_length(
            gb_norm.TORAX_QL, geo
        ),
        two_point_mask=two_point_mask,
    )


class GyaradaxQLConfig(pydantic_model_base.ComponentTransportBase):
  """Config for the gyaradax-QL transport model.

  Attributes:
    model_name: transport model selector. Hardcoded to 'gyaradax-ql'.
    rho_match: normalized-radius flux tubes where gyaradax is actually run;
      fluxes are interpolated from these onto the full face grid.
    backend: gyaradax compute backend, 'jax' (AD-clean) or 'cuda' (no AD).
      'cuda' with mixed_precision is ~4x faster per step than the fp64 jax path.
    mixed_precision: fp32 nonlinear FFTs, fp64 linear terms and field solve.
    nvpar: parallel-velocity grid points.
    nmu: magnetic-moment grid points.
    ns: parallel (field-line) grid points.
    nkx: radial wavenumber modes.
    nky: binormal wavenumber modes. All grid fields default to None,
      meaning: adopt the grid the selected Cn calibration head was fit at
      (its payload 'grid' metadata). Set explicitly to override.
    ikxspace: kx mode spacing (parallel boundary connection).
    krhomax: maximum binormal wavenumber k_theta*rho; same None semantics.
    n_steps_linear: hard cap on RK4 steps per linear gyaradax run.
    rule: saturation rule, a key of gyaradax.quasilinear.rules.RULES
      ('canonical', 'sat0_waltz', 'sat1_zonal', 'sat2_spectral',
      'sat3_regime', 'qualikiz'). Each rule has its own amplitude constant,
      so a non-default rule needs cn_override or a head fit for it.
    cn_override: amplitude constant Cn, bypassing the calibration head.
    cn_calibration_path: selects the Cn calibration head. 'auto' (default)
      uses the head bundled with gyaradax; a registry name
      (gyaradax.quasilinear.models.registry.MODELS) selects a named bundled
      head; any other value is a path to your own pickled head; None uses
      the uncalibrated cn_scalar = 1.
    early_stop: stop the linear solve once per-ky growth rates converge.
    early_stop_block: gksolve steps per convergence-check block.
    early_stop_atol: absolute tolerance on the growth-rate change.
    early_stop_rtol: relative tolerance on the growth-rate change.
    early_stop_min_steps: minimum steps before early-stop can trigger.
    early_stop_patience: consecutive converged checks required to stop.
    DV_effective: effective-D / effective-V particle-transport decomposition
      (inert while pfe is a zero placeholder).
    An_min: minimum |R/Ln| below which effective V is used instead of
      effective D.
    em: EXPERIMENTAL electromagnetic tier — finite local beta and A_par
      flutter in the linear solve; needs a beta-aware Cn calibration.
    diagnostics_path: when set, append one JSON row per (transport call,
      radius) to this `.jsonl`: local parameters, pre-unit-conversion fluxes,
      early-stop blocks/steps actually run, converged growth rate and
      wallclock. Written host-side through io_callback, so it is jit-safe;
      None (default) disables it at zero cost. See gyaradax_diagnostics.
  """

  model_name: Annotated[Literal["gyaradax-ql"], torax_pydantic.JAX_STATIC] = (
      "gyaradax-ql"
  )
  rho_match: Annotated[Tuple[float, ...], torax_pydantic.JAX_STATIC] = (
      0.35,
      0.55,
      0.75,
      0.875,
  )
  backend: Annotated[str, torax_pydantic.JAX_STATIC] = "jax"
  mixed_precision: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  nvpar: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nmu: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  ns: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nkx: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nky: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  ikxspace: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  krhomax: Annotated[Optional[float], torax_pydantic.JAX_STATIC] = None
  n_steps_linear: Annotated[int, torax_pydantic.JAX_STATIC] = 2000
  rule: Annotated[str, torax_pydantic.JAX_STATIC] = "canonical"
  cn_override: Annotated[Optional[float], torax_pydantic.JAX_STATIC] = None
  cn_calibration_path: Annotated[Optional[str], torax_pydantic.JAX_STATIC] = (
      "auto"
  )
  early_stop: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  early_stop_block: Annotated[int, torax_pydantic.JAX_STATIC] = 100
  early_stop_atol: Annotated[float, torax_pydantic.JAX_STATIC] = 1e-4
  early_stop_rtol: Annotated[float, torax_pydantic.JAX_STATIC] = 1e-3
  early_stop_min_steps: Annotated[int, torax_pydantic.JAX_STATIC] = 200
  early_stop_patience: Annotated[int, torax_pydantic.JAX_STATIC] = 2
  DV_effective: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  An_min: Annotated[float, torax_pydantic.JAX_STATIC] = 0.05
  em: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  diagnostics_path: Annotated[Optional[str], torax_pydantic.JAX_STATIC] = None

  def build_transport_model(self) -> "GyaradaxQLTransportModel":
    return GyaradaxQLTransportModel.from_config(self)

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        DV_effective=self.DV_effective, An_min=self.An_min, **base_kwargs
    )


register_model.register_transport_model(GyaradaxQLConfig)
