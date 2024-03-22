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

"""Module for a single source/sink term.

This module contains all the base classes for defining source terms. Other files
in this folder use these classes to define specific types of sources/sinks.

See Source class docstring for more details on what a TORAX source is and how to
use it.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Callable

import chex
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import jax_utils
from torax import state as state_lib
from torax.sources import source_config


def get_cell_profile_shape(
    unused_config: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    unused_state: state_lib.ToraxSimState | None,
):
  """Returns the shape of a source profile on the cell grid."""
  return ProfileType.CELL.get_profile_shape(geo)


# Any callable which takes the dynamic config, geometry, and optional mesh
# state, and outputs a shape corresponding to the expected output of a source.
# See how these types of functions are used in the Source class below.
SourceOutputShapeFunction = Callable[
    [  # Arguments
        config_slice.DynamicConfigSlice,
        geometry.Geometry,
        state_lib.ToraxSimState | None,
    ],
    # Returns shape of the source's output.
    tuple[int, ...],
]


@enum.unique
class AffectedMeshStateAttribute(enum.Enum):
  """Defines which part of the state the source helps evolve.

  The profiles of each source/sink are terms included in equations evolving
  different parts of the mesh state. This enum maps a source to those equations.
  """

  # Source profile is not used for any mesh state equation
  NONE = 0
  # Current density equation.
  PSI = 1
  # Electron density equation.
  NE = 2
  # Ion temperature equation.
  TEMP_ION = 3
  # Electron temperature equation.
  TEMP_EL = 4


@dataclasses.dataclass(frozen=True, kw_only=True)
class Source:
  """Base class for a single source/sink term.

  Sources are used to compute source profiles (see source_profiles.py), which
  are in turn used to compute coeffs in sim.py.

  NOTE: For most use cases, you should extend or use SingleProfileSource defined
  below.

  Attributes:
    name: Name of this source. Used as a key to find this source's configuraiton
      in the DynamicConfigSlice. Also used as a key for the output in the
      SourceProfiles.
    affected_mesh_states: Mesh state attributes affected by this source's
      profile(s). This attribute defines which equations the source profiles are
      terms for. By default, the number of affected mesh states should equal the
      rank of the output shape returned by output_shape_getter. Subclasses may
      override this requirement.
    supported_types: Defines how the source computes its profile. Can be set to
      zero, model-based, etc. At runtime, the input runtime config (the Config
      or the DynamicConfigSlice) will specify which supported type the Source is
      running with. If the runtime config specifies an unsupported type, an
      error will raise.
    output_shape_getter: Callable which returns the shape of the profiles given
      by this source.
    model_func: The function used when the the runtime type is set to
      "MODEL_BASED". If not provided, then it defaults to returning zeros.
    formula: The prescribed formula used when the runtime type is set to
      "FORMULA_BASED". If not provided, then it defaults to returning zeros.
    affected_mesh_state_ints: Derived property from the affected_mesh_states.
      Integer values of those enums.
  """

  name: str

  # Defining a default here for the affected mesh states helps allow us to
  # freeze the default in subclasses of Source. Without adding a default here,
  # it isn't possible to add a default value in a child class AND hide it from
  # the arguments of the subclasses's __init__ function.
  # Similar logic holds for all the other attributes below.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      AffectedMeshStateAttribute.NONE,
  )

  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.ZERO,
      source_config.SourceType.FORMULA_BASED,
  )

  output_shape_getter: SourceOutputShapeFunction = get_cell_profile_shape

  model_func: source_config.SourceProfileFunction | None = None

  formula: source_config.SourceProfileFunction | None = None

  @property
  def affected_mesh_state_ints(self) -> tuple[int, ...]:
    return tuple([int(state.value) for state in self.affected_mesh_states])

  def check_source_type(
      self,
      source_type: int | jnp.ndarray,
  ) -> jnp.ndarray:
    """Raises an error if the source type is not supported."""
    # This function is really just a wrapper around jax_utils.error_if with the
    # custom error message coming from this class.
    source_type = jnp.array(source_type)
    source_type = jax_utils.error_if(
        source_type,
        jnp.logical_not(self._is_type_supported(source_type)),
        self._unsupported_type_error_msg(source_type),
    )
    return source_type  # pytype: disable=bad-return-type

  def _is_type_supported(
      self,
      source_type: int | jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns whether the source type is supported."""
    source_type = jnp.array(source_type)
    return jnp.any(
        jnp.bool_([
            supported_type.value == source_type
            for supported_type in self.supported_types
        ])
    )

  def _unsupported_type_error_msg(
      self,
      source_type: source_config.SourceType | int | jnp.ndarray,
  ) -> str:
    return (
        f'{self.name} supports the following types: {self.supported_types}.'
        f' Unsupported type provided: {source_type}.'
    )

  def get_value(
      self,
      source_type: int,  # value of the source_config.SourceType enum.
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_lib.ToraxSimState | None = None,
  ) -> chex.ArrayTree:
    """Returns the profile for this source during one time step.

    Args:
      source_type: Method to use calculate the source profile (formula, model,
        etc.). This integer should be the enum value of desired SourceType
        instead of the actual enum because enums are not JAX-friendly. If the
        input source type is not one of the object's supported types, this will
        raise an error.
      dynamic_config_slice: Slice of the general TORAX config that can be used
        as input for this time step.
      geo: Geometry of the torus.
      sim_state: Complete state of the TORAX simulator, including the mesh state
        of profiles being evolved by the PDE system. May be the state at the
        start of the time step or a live state being actively updated depending
        on whether this source is explicit or implicit. Explicit sources get the
        state at the start of the time step, implicit sources get the "live"
        state that is updated through the course of the time step as the solver
        converges.

    Returns:
      Array, arrays, or nested dataclass/dict of arrays for the source profile.
    """
    source_type = self.check_source_type(source_type)
    output_shape = self.output_shape_getter(
        dynamic_config_slice, geo, sim_state
    )
    model_func = (
        (lambda _0, _1, _2: jnp.zeros(output_shape))
        if self.model_func is None
        else self.model_func
    )
    formula = (
        (lambda _0, _1, _2: jnp.zeros(output_shape))
        if self.formula is None
        else self.formula
    )
    return get_source_profiles(
        source_type=source_type,
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        sim_state=sim_state,
        model_func=model_func,
        formula=formula,
        output_shape=output_shape,
    )

  def get_profile_for_affected_state(
      self,
      profile: chex.ArrayTree,
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    """Returns the part of the profile to use for the given state.

    A single source can output profiles used as terms in more than one equation
    while evolving the mesh state (for instance, it can output profiles for both
    the ion temperature and electron temperature equations).

    Users of this source, though, may need to grab the specific parts of the
    output (from get_value()) that relate to a specific part of the mesh state.

    This function helps do that. By default, it returns the input profile as is
    if the requested mesh-state attribute is valid, otherwise returns zeros.

    NOTE: This function assumes the ArrayTree returned by get_value() is a JAX
    array with shape (num affected mesh states, cell grid length) and that the
    order of the arrays in the output match the order of the
    affected_mesh_states attribute.

    Subclasses can override this behavior to fit the type of ArrayTree they
    output.

    Args:
      profile: The profile output from get_value().
      affected_mesh_state: The part of the mesh state we want to pull the
        profile for. This is the integer value of the enum
        AffectedMeshStateAttribute because enums are not JAX-friendly as
        function arguments. If it is not one of the mesh states this source
        actually affects, this will return zeros.
      geo: Geometry of the torus.

    Returns: The profile on the cell grid for the requested state.
    """
    # Get a valid index that defaults to 0 if not present.
    affected_mesh_state_ints = self.affected_mesh_state_ints
    idx = jnp.argmax(
        jnp.asarray(affected_mesh_state_ints) == affected_mesh_state
    )
    return jnp.where(
        affected_mesh_state in affected_mesh_state_ints,
        profile[idx, ...],
        jnp.zeros_like(geo.r),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleProfileSource(Source):
  """Source providing a single output profile on the cell grid.

  Most sources in TORAX are instances (or subclasses) of this class.

  You can define custom sources inline when constructing the full list of
  sources to use in TORAX.

  ```python
  # Define an electron-density source with a Gaussian profile.
  my_custom_source_name = 'custom_ne_source'
  my_custom_source = source.SingleProfileSource(
      name=my_custom_source_name,
      supported_types=(
          source_config.SourceType.ZERO,
          source_config.SourceType.FORMULA_BASED,
      ),
      affected_mesh_states=source.AffectedMeshStateAttribute.NE,
      formula=formulas.Gaussian(my_custom_source_name),
  )
  all_torax_sources = source_profiles.Sources(
      additional_sources=[
          my_custom_source,
      ]
  )
  ```

  You must also include a runtime config for the custom source:

  ```python
  my_torax_config = config.Config(
      sources=dict(
          ...  # Configs for other sources.
          # Set some params for the new source
          custom_ne_source=source_config.SourceConfig(
              source_type=source_config.SourceType.FORMULA_BASED,
              formula=formula_config.FormulaConfig(
                  gaussian=formula_config.Gaussian(
                      total=1.0,
                      c1=2.0,
                      c2=3.0,
                  ),
              ),
          ),
      ),
  )
  ```

  If you want to create a subclass of SingleProfileSource with frozen
  parameters, you can provide default implementations/attributes. This is an
  example of a model-based source with a frozen custom model that cannot be
  changed by a config:

  ```python

  def _my_foo_model(dynamic_config_slice, geo, state) -> jnp.ndarray:
    # implement your foo model.

  class FooSource(SingleProfileSource):

    name: str = 'foo_source'  # the default name for this source.

    # By default, FooSource's can be model-based or set to 0.
    supported_types: tuple[source_config.SourceType, ...] = (
        source_config.SourceType.ZERO,
        source_config.SourceType.MODEL_BASED,
    )

    # Don't include model_func in the __init__ arguments and freeze it.
    model_func: source_config.SourceProfileFunction = dataclasses.field(
        init=False,
        default_factory=lambda: _my_foo_model,
    )
  ```
  """

  # Don't include output_shape_getter in the __init__ arguments.
  # Freeze this parameter so that it always outputs a single cell profile.
  output_shape_getter: SourceOutputShapeFunction = dataclasses.field(
      init=False,
      default_factory=lambda: get_cell_profile_shape,
  )

  def get_value(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_lib.ToraxSimState | None = None,
  ) -> jnp.ndarray:
    """Returns the profile for this source during one time step."""
    output_shape = self.output_shape_getter(
        dynamic_config_slice, geo, sim_state
    )
    profile = super().get_value(
        source_type=source_type,
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        sim_state=sim_state,
    )
    assert isinstance(profile, jnp.ndarray)
    chex.assert_rank(profile, 1)
    chex.assert_shape(profile, output_shape)
    return profile

  def get_profile_for_affected_state(
      self,
      profile: chex.ArrayTree,
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    return jnp.where(
        affected_mesh_state in self.affected_mesh_state_ints,
        profile,
        jnp.zeros_like(geo.r),
    )


class ProfileType(enum.Enum):
  """Describes what kind of profile is expected from a source."""

  # Source should return a profile on the cell grid.
  CELL = enum.auto()

  # Source should return a profile on the face grid.
  FACE = enum.auto()

  def get_profile_shape(self, geo: geometry.Geometry) -> tuple[int, ...]:
    """Returns the expected length of the source profile."""
    profile_type_to_len = {
        ProfileType.CELL: geo.r.shape,
        ProfileType.FACE: geo.r_face.shape,
    }
    return profile_type_to_len[self]

  def get_zero_profile(self, geo: geometry.Geometry) -> jnp.ndarray:
    """Returns a source profile with all zeros."""
    return jnp.zeros(self.get_profile_shape(geo))


def get_source_profiles(
    source_type: int | jnp.ndarray,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_lib.ToraxSimState | None,
    model_func: source_config.SourceProfileFunction,
    formula: source_config.SourceProfileFunction,
    output_shape: tuple[int, ...],
) -> jnp.ndarray:
  """Returns source profiles requested by the source_config.

  This function handles MODEL_BASED, FORMULA_BASED, and ZERO sources. All other
  source types will be ignored.

  Args:
    source_type: Method to use to get the source profile.
    dynamic_config_slice: Slice of the general TORAX config that can be used as
      input for this time step.
    geo: Geometry information. Used as input to the source profile functions.
    sim_state: Full simulation state. Used as input to the source profile
      functions.
    model_func: Model function.
    formula: Formula implementation.
    output_shape: Expected shape of the outut array.

  Returns:
    Output array of a profile or concatenated/stacked profiles.
  """
  zeros = jnp.zeros(output_shape)
  output = jnp.zeros(output_shape)
  output += jnp.where(
      source_type == source_config.SourceType.MODEL_BASED.value,
      model_func(dynamic_config_slice, geo, sim_state),
      zeros,
  )
  output += jnp.where(
      source_type == source_config.SourceType.FORMULA_BASED.value,
      formula(dynamic_config_slice, geo, sim_state),
      zeros,
  )
  return output


# Convenience classes to reduce a little boilerplate for some of the common
# sources defined in the other files in this folder.


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleProfilePsiSource(SingleProfileSource):

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(AffectedMeshStateAttribute.PSI,),
      )
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleProfileNeSource(SingleProfileSource):

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(AffectedMeshStateAttribute.NE,),
      )
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleProfileTempIonSource(SingleProfileSource):

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(AffectedMeshStateAttribute.TEMP_ION,),
      )
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleProfileTempElSource(SingleProfileSource):

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(AffectedMeshStateAttribute.TEMP_EL,),
      )
  )


def _get_ion_el_output_shape(unused_config, geo, unused_state):
  return (2,) + ProfileType.CELL.get_profile_shape(geo)


@dataclasses.dataclass(frozen=True, kw_only=True)
class IonElectronSource(Source):
  """Base class for a source/sink that can be used for both ions / electrons.

  Some ion and electron heat sources share a lot of computation resulting in
  values that are often simply proportionally scaled versions of the other. To
  help with defining those sources where you'd like to (a) keep the values
  similar and (b) get some small efficiency gain by doing some computations
  once instead of twice (once for ions and again for electrons), this class
  gives a hook for doing that.

  This class is set to always return 2 source profiles on the cell grid, the
  first being ion profile and the second being the electron profile.
  """

  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.FORMULA_BASED,
      source_config.SourceType.ZERO,
  )

  # Don't include affected_mesh_states in the __init__ arguments.
  # Freeze this param.
  affected_mesh_states: tuple[AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(
              AffectedMeshStateAttribute.TEMP_ION,
              AffectedMeshStateAttribute.TEMP_EL,
          ),
      )
  )

  # Don't include output_shape_getter in the __init__ arguments.
  # Freeze this parameter so that it always outputs 2 cell profiles.
  output_shape_getter: SourceOutputShapeFunction = dataclasses.field(
      init=False,
      default_factory=lambda: _get_ion_el_output_shape,
  )

  def get_value(
      self,
      source_type: int,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      sim_state: state_lib.ToraxSimState | None = None,
  ) -> jnp.ndarray:
    """Computes the ion and electron values of the source.

    Args:
      source_type: Method to use calculate the source profile (formula, model,
        etc.). This is the enum value of SourceType instead of the actual enum
        instance because enums aren't JAX-friendly.
      dynamic_config_slice: Input config which can change from time step to time
        step.
      geo: Geometry of the torus.
      sim_state: Full simulation state including the mesh state to use while
        calculating this source profile.

    Returns:
      2 stacked arrays, the first for the ion profile and the second for the
      electron profile.
    """
    output_shape = self.output_shape_getter(
        dynamic_config_slice, geo, sim_state
    )
    profile = super().get_value(
        source_type=source_type,
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        sim_state=sim_state,
    )
    assert isinstance(profile, jnp.ndarray)
    chex.assert_rank(profile, 2)
    chex.assert_shape(profile, output_shape)
    return profile
