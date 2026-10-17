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

"""Nonlinear gyaradax as a TORAX transport model.

A nonlinear gyaradax run per `rho_match` radius, time-averaged over trailing
blocks. The grid, geometry and TORAX wiring come from `gyaradax_base`; only
the nonlinear flux model is here. There is no saturation rule and no Cn: the
calibration head is consulted solely for grid metadata.
"""

import dataclasses
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

from gyaradax.params import GKParams
from gyaradax.quasilinear import point_eval
import jax.numpy as jnp
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import gyaradax_base as base_lib
from torax._src.transport_model import register_model


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxNLTransportModel(base_lib.BaseGKWGyaradaxPlugin):
  """Nonlinear ground-truth gyaradax transport model."""

  n_steps_nl: int = 30000
  nl_tail_blocks: int = 12
  nl_block: int = 500
  cn_calibration_path: str = "auto"

  @classmethod
  def _grid_fallback(cls, cfg):
    return base_lib.head_grid(getattr(cfg, "cn_calibration_path", "") or "")

  @classmethod
  def from_config(cls, cfg) -> "GyaradaxNLTransportModel":
    return cls(
        **cls._base_kwargs(cfg),
        n_steps_nl=cfg.n_steps_nl,
        nl_tail_blocks=cfg.nl_tail_blocks,
        nl_block=cfg.nl_block,
        cn_calibration_path=cfg.cn_calibration_path or "",
    )

  def _per_radius(
      self, params: GKParams, geom: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    q_i, q_e, pfe = point_eval.nl_at_point(
        params,
        geom,
        (self.nvpar, self.nmu, self.ns, self.nkx, self.nky),
        n_steps=self.n_steps_nl,
        tail_blocks=self.nl_tail_blocks,
        block=self.nl_block,
    )
    # step counts are static here; the linear early-stop diagnostics don't apply
    return q_i, q_e, pfe, {"n_steps": jnp.asarray(self.n_steps_nl)}


class GyaradaxNLConfig(base_lib.SolverGKWGyaradaxConfig):
  """Config for the gyaradax-nl ground-truth transport model.

  Attributes:
    model_name: transport model selector. Hardcoded to 'gyaradax-nl'.
    n_steps_nl: total nonlinear steps per radius (burn-in + tail).
    nl_tail_blocks: number of trailing blocks averaged into the flux.
    nl_block: steps per tail block.
    cn_calibration_path: consulted only for the grid metadata that unset grid
      fields fall back to; no Cn is applied in a nonlinear run.
    Remaining attributes are inherited from BaseGKWGyaradaxConfig.
  """

  model_name: Annotated[Literal["gyaradax-nl"], torax_pydantic.JAX_STATIC] = (
      "gyaradax-nl"
  )
  n_steps_nl: Annotated[int, torax_pydantic.JAX_STATIC] = 30000
  nl_tail_blocks: Annotated[int, torax_pydantic.JAX_STATIC] = 12
  nl_block: Annotated[int, torax_pydantic.JAX_STATIC] = 500
  cn_calibration_path: Annotated[Optional[str], torax_pydantic.JAX_STATIC] = (
      "auto"
  )

  def build_transport_model(self) -> GyaradaxNLTransportModel:
    return GyaradaxNLTransportModel.from_config(self)


register_model.register_transport_model(GyaradaxNLConfig)
