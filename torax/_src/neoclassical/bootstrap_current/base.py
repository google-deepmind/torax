# Copyright 2024 DeepMind Technologies Limited
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

"""Base class for bootstrap current models."""
from __future__ import annotations

import abc
import dataclasses
import logging
import jax
import jax.numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
  """Values returned by a bootstrap current model."""

  j_parallel_bootstrap: jax.Array
  j_parallel_bootstrap_face: jax.Array

  # TODO(b/434175938). Remove these deprecated properties in V2.
  @property
  def j_bootstrap(self) -> jax.Array:
    """DEPRECATED: use j_parallel_bootstrap."""
    logging.warning(
        '`j_bootstrap` is deprecated, use `j_parallel_bootstrap` instead.'
        ' `j_bootstrap` will be removed in a future version.'
    )
    return self.j_parallel_bootstrap

  @property
  def j_bootstrap_face(self) -> jax.Array:
    """DEPRECATED: use `j_parallel_bootstrap_face`."""
    logging.warning(
        '`j_bootstrap_face` is deprecated, use `j_parallel_bootstrap_face`'
        ' instead. `j_bootstrap_face` will be removed in a future version.'
    )
    return self.j_parallel_bootstrap_face

  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> 'BootstrapCurrent':
    """Returns a BootstrapCurrent with all values set to zero."""
    return cls(
        j_parallel_bootstrap=jnp.zeros_like(geometry.rho_norm),
        j_parallel_bootstrap_face=jnp.zeros_like(geometry.rho_face_norm),
    )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Converts bootstrap current profiles into an OutputVar mapping."""
    j_bootstrap = output_grid_context.extend_cell_grid_to_boundaries(
        self.j_parallel_bootstrap,
        self.j_parallel_bootstrap_face,
    )
    return {
        output_keys.J_PARALLEL_BOOTSTRAP: context.pack(
            output_keys.J_PARALLEL_BOOTSTRAP, j_bootstrap
        ),
    }


def calculate_analytic_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    p_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    geo: geometry_lib.Geometry,
    L31: array_typing.FloatVectorFace,
    L32: array_typing.FloatVectorFace,
    L34: array_typing.FloatVectorFace,
    alpha: array_typing.FloatVectorFace,
) -> BootstrapCurrent:
  """Shared function for computing bootstrap current from analytic fits."""
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = p_e.face_value()
  pi = p_i.face_value()

  dpsi_drnorm = psi.face_grad()
  dlnne_drnorm = n_e.face_grad() / n_e.face_value()
  dlnni_drnorm = n_i.face_grad() / n_i.face_value()
  dlnte_drnorm = T_e.face_grad() / T_e.face_value()
  dlnti_drnorm = T_i.face_grad() / T_i.face_value()

  global_coeff = prefactor[1:] / dpsi_drnorm[1:]  # pyrefly: ignore[bad-index]
  global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])

  necoeff = L31 * pe
  nicoeff = L31 * pi
  tecoeff = (L31 + L32) * pe
  ticoeff = (L31 + alpha * L34) * pi

  j_parallel_bootstrap_face = global_coeff * (
      necoeff * dlnne_drnorm
      + nicoeff * dlnni_drnorm
      + tecoeff * dlnte_drnorm
      + ticoeff * dlnti_drnorm
  )
  j_parallel_bootstrap = geometry_lib.face_to_cell(j_parallel_bootstrap_face)

  return BootstrapCurrent(
      j_parallel_bootstrap=j_parallel_bootstrap,  # pyrefly: ignore[bad-argument-type]
      j_parallel_bootstrap_face=j_parallel_bootstrap_face,
  )


class BootstrapCurrentModel(abc.ABC):
  """Base class for bootstrap current models."""

  @abc.abstractmethod
  def calculate_bootstrap_current(
      self,
      runtime_params: bootstrap_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> BootstrapCurrent:
    """Calculates bootstrap current."""


class BootstrapCurrentModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for bootstrap current model configs."""
  bootstrap_multiplier: pydantic.NonNegativeFloat = 1.0

  @abc.abstractmethod
  def build_runtime_params(self) -> bootstrap_runtime_params.RuntimeParams:
    """Builds runtime params."""

  @abc.abstractmethod
  def build_model(self) -> BootstrapCurrentModel:
    """Builds bootstrap current model."""
