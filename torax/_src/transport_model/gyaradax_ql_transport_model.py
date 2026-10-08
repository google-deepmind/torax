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

A linear gyaradax run per `rho_match` radius, turned into fluxes by a
saturation rule with an amplitude constant Cn from a calibration head. The
flux-tube grid, the TORAX wiring and the unit conversion live in
`gyaradax_base`; only the quasilinear flux model is here. Grid fields fall
back to the Cn calibration head's metadata unless set explicitly.
"""

import dataclasses
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

from gyaradax.params import GKParams
from gyaradax.quasilinear import point_eval
from gyaradax.quasilinear.models import load_cn_payload
from gyaradax.quasilinear.models import select_cn_head
import jax.numpy as jnp
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import gyaradax_base as base_lib
from torax._src.transport_model import register_model

RuntimeParams = base_lib.RuntimeParams


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxQLTransportModel(base_lib.BaseGKWGyaradaxPlugin):
  """QL gyaradax transport model."""

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

  @classmethod
  def _grid_fallback(cls, cfg):
    payload = load_cn_payload(getattr(cfg, "cn_calibration_path", "") or "")
    return payload.get("grid", {}) if isinstance(payload, dict) else {}

  @classmethod
  def from_config(cls, cfg) -> "GyaradaxQLTransportModel":
    return cls(
        **cls._base_kwargs(cfg),
        n_steps_linear=cfg.n_steps_linear,
        rule=cfg.rule,
        cn_override=(-1.0 if cfg.cn_override is None else float(cfg.cn_override)),
        cn_calibration_path=cfg.cn_calibration_path or "",
        early_stop=cfg.early_stop,
        early_stop_block=cfg.early_stop_block,
        early_stop_atol=cfg.early_stop_atol,
        early_stop_rtol=cfg.early_stop_rtol,
        early_stop_min_steps=cfg.early_stop_min_steps,
        early_stop_patience=cfg.early_stop_patience,
    )

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


class GyaradaxQLConfig(base_lib.SolverGKWGyaradaxConfig):
  """Config for the gyaradax-QL transport model.

  Attributes:
    model_name: transport model selector. Hardcoded to 'gyaradax-ql'.
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
      the uncalibrated cn_scalar = 1. Grid fields left unset adopt this
      head's 'grid' metadata.
    early_stop: stop the linear solve once per-ky growth rates converge.
    early_stop_block: gksolve steps per convergence-check block.
    early_stop_atol: absolute tolerance on the growth-rate change.
    early_stop_rtol: relative tolerance on the growth-rate change.
    early_stop_min_steps: minimum steps before early-stop can trigger.
    early_stop_patience: consecutive converged checks required to stop.
    Remaining attributes are inherited from BaseGKWGyaradaxConfig.
  """

  model_name: Annotated[Literal["gyaradax-ql"], torax_pydantic.JAX_STATIC] = (
      "gyaradax-ql"
  )
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

  def build_transport_model(self) -> "GyaradaxQLTransportModel":
    return GyaradaxQLTransportModel.from_config(self)


register_model.register_transport_model(GyaradaxQLConfig)
