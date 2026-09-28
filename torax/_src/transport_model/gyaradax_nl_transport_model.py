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

"""Full nonlinear gyaradax as a TORAX transport model (ground truth).

Identical profile/geometry mapping and unit conversion as `gyaradax-ql`, but
each rho_match radius runs a full nonlinear turbulence simulation to
saturation (`gyaradax.quasilinear.point_eval.nl_at_point`) instead of the
quasilinear estimate. Orders of magnitude slower per transport call — meant
for offline ground-truth validation and surrogate training-set generation,
not production transport loops.
"""

import dataclasses
from typing import Annotated, Any, Dict, Literal, Tuple

from gyaradax.params import GKParams
from gyaradax.quasilinear import point_eval
import jax.numpy as jnp
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import register_model
from torax._src.transport_model import gyaradax_ql_transport_model as ql_lib


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxNLTransportModel(ql_lib.GyaradaxQLTransportModel):
  """Nonlinear ground-truth gyaradax transport model."""

  n_steps_nl: int = 30000
  nl_tail_blocks: int = 12
  nl_block: int = 500

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


class GyaradaxNLConfig(ql_lib.GyaradaxQLConfig):
  """Config for the gyaradax-nl ground-truth transport model.

  Attributes:
    model_name: transport model selector. Hardcoded to 'gyaradax-nl'.
    n_steps_nl: total nonlinear steps per radius (burn-in + tail).
    nl_tail_blocks: number of trailing blocks averaged into the flux.
    nl_block: steps per tail block.
    Remaining attributes are inherited from GyaradaxQLConfig (grid fields
    resolve from the Cn head metadata; the head itself is unused here).
  """

  model_name: Annotated[Literal["gyaradax-nl"], torax_pydantic.JAX_STATIC] = (
      "gyaradax-nl"
  )
  n_steps_nl: Annotated[int, torax_pydantic.JAX_STATIC] = 30000
  nl_tail_blocks: Annotated[int, torax_pydantic.JAX_STATIC] = 12
  nl_block: Annotated[int, torax_pydantic.JAX_STATIC] = 500

  def build_transport_model(self) -> GyaradaxNLTransportModel:
    base = GyaradaxNLTransportModel.from_config(self)
    return dataclasses.replace(
        base,
        n_steps_nl=self.n_steps_nl,
        nl_tail_blocks=self.nl_tail_blocks,
        nl_block=self.nl_block,
    )


register_model.register_transport_model(GyaradaxNLConfig)
