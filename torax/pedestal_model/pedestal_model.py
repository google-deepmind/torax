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
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry


@chex.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of the PedestalModel."""

  # The location of the pedestal.
  rho_norm_ped_top: array_typing.ScalarFloat
  # pylint: disable=invalid-name
  # Using physics notation naming convention
  # The ion temperature at the pedestal.
  Tiped: array_typing.ScalarFloat
  # The electron temperature at the pedestal.
  Teped: array_typing.ScalarFloat
  # The electron density at the pedestal in units of nref.
  neped: array_typing.ScalarFloat
  # pylint: enable=invalid-name


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

    # Calculate the pedestal values.
    return self._call_implementation(
        dynamic_runtime_params_slice, geo, core_profiles
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    """Calculate the pedestal values."""
