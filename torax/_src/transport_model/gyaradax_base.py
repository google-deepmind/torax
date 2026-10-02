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

"""Shared TORAX plumbing for the gyaradax/GKW transport models.

`BaseGKWGyaradaxPlugin` owns everything that is the same whichever way the
fluxes are produced: the flux-tube grid, the local GKParams/geometry built at
each `rho_match` radius, the vmap over radii, the GKW gyroBohm -> TORAX unit
conversion and the diagnostics sink. Subclasses supply only the flux model,
by overriding `_per_radius` (local params + geometry -> gyroBohm fluxes) or,
when they build their own geometry, `_per_radius_from_profiles`.
"""

import dataclasses
from functools import lru_cache
import warnings
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import chex
from gyaradax.geometry import build_topology
from gyaradax.geometry import compute_continuous_geometry
from gyaradax.params import GKParams
from gyaradax.quasilinear.models import load_cn_payload
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
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model.quasilinear_transport_model import QuasilinearInputs

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
  """Runtime parameters for the gyaradax transport models."""


@lru_cache(maxsize=8)
def _get_topology_cached(nkx: int, nky: int, ikxspace: int, ns: int):
  """Topology dict keyed on static grid sizes (cached across `from_config`)."""
  return build_topology(nkx=nkx, nky=nky, ikxspace=ikxspace, ns=ns)


def head_grid(path: str) -> dict:
  """Grid metadata carried by a Cn calibration head, for `_grid_fallback`."""
  payload = load_cn_payload(path or "")
  return payload.get("grid", {}) if isinstance(payload, dict) else {}


def _resolve_grid(cfg, fallback_grid=None) -> dict:
  """Grid resolution: explicit config wins, else `fallback_grid`."""
  head_grid = fallback_grid or {}
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
        f"gyaradax grid underspecified ({missing}): set these config fields "
        "explicitly, or select a source that carries grid metadata."
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
class BaseGKWGyaradaxPlugin(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Grid, geometry and TORAX wiring shared by the gyaradax transport models."""

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
  # EXPERIMENTAL: beta + A_par flutter
  em: bool = False
  diagnostics_path: str = ""

  @classmethod
  def _grid_fallback(cls, cfg) -> Optional[dict]:
    """Grid metadata to use for fields the config leaves unset."""
    del cfg
    return None

  @classmethod
  def _base_kwargs(cls, cfg) -> dict:
    """Constructor arguments for the fields defined on this base."""
    fis = getattr(cfg, "fast_ion_stabilization", None)
    # raise on nonzero fast_ion_stabilization instead of silently ignoring it
    if fis is not None and float(fis.get_value(0.0)) != 0.0:
      raise NotImplementedError(
          f"{cls.__name__} does not implement fast_ion_stabilization."
      )
    grid = _resolve_grid(cfg, cls._grid_fallback(cfg))
    # warm caches outside jit: build_topology arrays must not become tracers
    _get_topology_cached(
        int(grid["nkx"]),
        int(grid["nky"]),
        int(grid["ikxspace"]),
        int(grid["ns"]),
    )
    return dict(
        rho_match=tuple(cfg.rho_match),
        backend=cfg.backend,
        mixed_precision=cfg.mixed_precision,
        nvpar=int(grid["nvpar"]),
        nmu=int(grid["nmu"]),
        ns=int(grid["ns"]),
        nkx=int(grid["nkx"]),
        nky=int(grid["nky"]),
        ikxspace=int(grid["ikxspace"]),
        krhomax=float(grid["krhomax"]),
        em=getattr(cfg, "em", False),
        diagnostics_path=getattr(cfg, "diagnostics_path", None) or "",
    )

  @classmethod
  def from_config(cls, cfg) -> "BaseGKWGyaradaxPlugin":
    return cls(**cls._base_kwargs(cfg))

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
    """(qi, qe, pfe) in GKW gyroBohm plus per-call diagnostics."""
    raise NotImplementedError

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



class BaseGKWGyaradaxConfig(pydantic_model_base.ComponentTransportBase):
  """Config fields shared by every gyaradax transport model.

  Attributes:
    rho_match: normalized-radius flux tubes where the model is evaluated;
      fluxes are interpolated from these onto the full face grid.
    DV_effective: effective-D / effective-V particle-transport decomposition
      (inert while pfe is a zero placeholder).
    An_min: minimum |R/Ln| below which effective V is used instead of
      effective D.
    diagnostics_path: when set, append one JSON row per (transport call,
      radius) to this `.jsonl`. Written host-side through io_callback, so it
      is jit-safe; None (default) disables it at zero cost.
  """

  rho_match: Annotated[Tuple[float, ...], torax_pydantic.JAX_STATIC] = (
      0.35,
      0.55,
      0.75,
      0.875,
  )
  DV_effective: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  An_min: Annotated[float, torax_pydantic.JAX_STATIC] = 0.05
  diagnostics_path: Annotated[Optional[str], torax_pydantic.JAX_STATIC] = None

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        DV_effective=self.DV_effective, An_min=self.An_min, **base_kwargs
    )


class SolverGKWGyaradaxConfig(BaseGKWGyaradaxConfig):
  """Adds the flux-tube grid, for models that run the gyaradax solver.

  Attributes:
    backend: gyaradax compute backend, 'jax' (AD-clean) or 'cuda' (no AD).
      'cuda' with mixed_precision is ~4x faster per step than the fp64 jax path.
    mixed_precision: fp32 nonlinear FFTs, fp64 linear terms and field solve.
    nvpar: parallel-velocity grid points.
    nmu: magnetic-moment grid points.
    ns: parallel (field-line) grid points.
    nkx: radial wavenumber modes.
    nky: binormal wavenumber modes. Grid fields default to None, meaning the
      subclass's `_grid_fallback` supplies them; set explicitly to override.
    ikxspace: kx mode spacing (parallel boundary connection).
    krhomax: maximum binormal wavenumber k_theta*rho; same None semantics.
    em: EXPERIMENTAL electromagnetic tier — finite local beta and A_par
      flutter in the linear solve.
  """

  backend: Annotated[str, torax_pydantic.JAX_STATIC] = "jax"
  mixed_precision: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  nvpar: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nmu: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  ns: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nkx: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  nky: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  ikxspace: Annotated[Optional[int], torax_pydantic.JAX_STATIC] = None
  krhomax: Annotated[Optional[float], torax_pydantic.JAX_STATIC] = None
  em: Annotated[bool, torax_pydantic.JAX_STATIC] = False
