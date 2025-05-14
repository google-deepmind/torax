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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""
import abc
import chex
import jax
import jax.numpy as jnp
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry

# pylint: disable=invalid-name
# Using physics notation naming convention


@chex.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of the PedestalModel."""

  # The location of the pedestal.
  rho_norm_ped_top: array_typing.ScalarFloat
  # The index of the pedestal in rho_norm.
  rho_norm_ped_top_idx: array_typing.ScalarInt
  # The ion temperature at the pedestal.
  T_i_ped: array_typing.ScalarFloat
  # The electron temperature at the pedestal.
  T_e_ped: array_typing.ScalarFloat
  # The electron density at the pedestal in units 10^-3.
  n_e_ped: array_typing.ScalarFloat


class PedestalModel(abc.ABC):
  """Calculates temperature and density of the pedestal.

  Subclass responsbilities:
  - Must set _frozen = True at the end of the subclass __init__ method to
    activate immutability.
  """

  def __setattr__(self, attr, value):
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """Override __setattr__ to make the class (sort of) immutable.

    Note that you can still do obj.field.subfield = x, so it is not true
    immutability, but this to helps to avoid some careless errors.
    """
    if getattr(self, "_frozen", False):
      raise AttributeError("PedestalModels are immutable.")
    return super().__setattr__(attr, value)

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    if not getattr(self, "_frozen", False):
      raise RuntimeError(
          f"Subclass implementation {type(self)} forgot to "
          "freeze at the end of __init__."
      )

    return jax.lax.cond(
        dynamic_runtime_params_slice.pedestal.set_pedestal,
        lambda: self._call_implementation(
            dynamic_runtime_params_slice, geo, core_profiles
        ),
        # Set the pedestal location to infinite to indicate that the pedestal is
        # not present.
        # Set the index to outside of bounds of the mesh to indicate that the
        # pedestal is not present.
        lambda: PedestalModelOutput(
            rho_norm_ped_top=jnp.inf, T_i_ped=0.0, T_e_ped=0.0, n_e_ped=0.0,
            rho_norm_ped_top_idx=geo.torax_mesh.nx
        ),
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    """Calculate the pedestal values."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hash function for the pedestal model.

    Needed for jax.jit caching to work.
    """
    ...

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality function for the pedestal model."""
    ...
