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

import abc
import dataclasses
import enum
import types
import typing
from typing import Any, ClassVar, Optional, Protocol

# We use Optional here because | doesn't work with string name types.
# We use string name 'source_models.SourceModels' in this file to avoid
# circular imports.

import chex
from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_profiles


# pytype bug: 'source_models.SourceModels' not treated as forward reference
# pytype: disable=name-error
@typing.runtime_checkable
class SourceProfileFunction(Protocol):
  """Sources implement these functions to be able to provide source profiles."""

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      source_name: str,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      source_models: Optional['source_models.SourceModels'],
  ) -> tuple[chex.Array, ...]:
    ...


# pytype: enable=name-error


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
  """Defines which part of the core profiles the source helps evolve.

  The profiles of each source/sink are terms included in equations evolving
  different core profiles. This enum maps a source to those equations.
  """

  # Current density equation.
  PSI = 1
  # Electron density equation.
  NE = 2
  # Ion temperature equation.
  TEMP_ION = 3
  # Electron temperature equation.
  TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source(abc.ABC):
  """Base class for a single source/sink term.

  Sources are used to compute source profiles (see source_profiles.py), which
  are in turn used to compute coeffs in sim.py.

  Attributes:
    SOURCE_NAME: The name of the source.
    DEFAULT_MODEL_FUNCTION_NAME: The name of the model function used with this
      source if another isn't specified.
    runtime_params: Input dataclass containing all the source-specific runtime
      parameters. At runtime, the parameters here are interpolated to a specific
      time t and then passed to the model_func, depending on the mode this
      source is running in.
    affected_core_profiles: Core profiles affected by this source's profile(s).
      This attribute defines which equations the source profiles are terms for.
      By default, the number of affected core profiles should equal the rank of
      the output shape returned by `output_shape`.
    model_func: The function used when the the runtime type is set to
      "MODEL_BASED". If not provided, then it defaults to returning zeros.
    affected_core_profiles_ints: Derived property from the
      affected_core_profiles. Integer values of those enums.
  """

  SOURCE_NAME: ClassVar[str] = 'source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'default'
  model_func: SourceProfileFunction | None = None

  @property
  @abc.abstractmethod
  def source_name(self) -> str:
    """Returns the name of the source."""

  @property
  @abc.abstractmethod
  def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
    """Returns the core profiles affected by this source."""

  @property
  def affected_core_profiles_ints(self) -> tuple[int, ...]:
    return tuple([int(cp) for cp in self.affected_core_profiles])

  def get_value(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
  ) -> tuple[chex.Array, ...]:
    """Returns the cell grid profile for this source during one time step.

    Args:
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice: Slice of the general TORAX config that can
        be used as input for this time step.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles. May be the profiles at the start of
        the time step or a "live" set of core profiles being actively updated
        depending on whether this source is explicit or implicit. Explicit
        sources get the core profiles at the start of the time step, implicit
        sources get the "live" profiles that is updated through the course of
        the time step as the solver converges.
      calculated_source_profiles: The source profiles which have already been
        calculated for this time step if they exist. This is used to avoid
        recalculating profiles that are used as inputs to other sources. These
        profiles will only exist for Source instances that are implicit. i.e.
        explicit sources cannot depend on other calculated source profiles. In
        addition, different source types will have different availability of
        specific calculated_source_profiles since the calculation order matters.
        See source_profile_builders.py for more details.

    Returns:
      A tuple of arrays of shape (cell grid length,) with one array per affected
      core profile.
    """
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        self.source_name
    ]

    mode = static_runtime_params_slice.sources[self.source_name].mode
    match mode:
      case runtime_params_lib.Mode.MODEL_BASED.value:
        if self.model_func is None:
          raise ValueError(
              'Source is in MODEL_BASED mode but has no model function.'
          )
        return self.model_func(
            static_runtime_params_slice,
            dynamic_runtime_params_slice,
            geo,
            self.source_name,
            core_profiles,
            calculated_source_profiles,
            getattr(self, 'source_models', None),
        )
      case runtime_params_lib.Mode.PRESCRIBED.value:
        # TODO(b/395854896) add support for sources that affect multiple core
        # profiles.
        return (dynamic_source_runtime_params.prescribed_values,)
      case runtime_params_lib.Mode.ZERO.value:
        zeros = jnp.zeros(geo.rho_norm.shape)
        return (zeros,) * len(self.affected_core_profiles)
      case _:
        raise ValueError(f'Unknown mode: {mode}')

  def get_source_profile_for_affected_core_profile(
      self,
      profile: tuple[chex.Array, ...],
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> chex.Array:
    """Returns the part of the profile to use for the given core profile.

    A single source can output profiles used as terms in more than one equation
    while evolving the core profiles (for instance, it can output profiles for
    both the ion temperature and electron temperature equations).

    Users of this source, though, may need to grab the specific parts of the
    output (from get_value()) that relate to a specific core profile.

    This function helps do that. By default, it returns the input profile as is
    if the requested core profile is valid, otherwise returns zeros.

    Args:
      profile: The profile output from get_value().
      affected_core_profile: The specific core profile we want to pull the
        profile for. This is the integer value of the enum AffectedCoreProfile
        because enums are not JAX-friendly as function arguments. If it is not
        one of the core profiles this source actually affects, this will return
        zeros.
      geo: Geometry of the torus.

    Returns: The source profile on the cell grid for the requested core profile.
    """
    # Get a valid index that defaults to 0 if not present.
    affected_core_profile_ints = self.affected_core_profiles_ints
    if affected_core_profile not in affected_core_profile_ints:
      return jnp.zeros_like(geo.rho)
    else:
      return profile[affected_core_profile_ints.index(affected_core_profile)]


@dataclasses.dataclass(frozen=False, kw_only=True)
class SourceBuilderProtocol(Protocol):
  """Make a best effort to define what SourceBuilders are with type hints.

  Note that these can't be used with `isinstance` or any other runtime
  evaluation, just static analysis.

  Attributes:
    runtime_params: Mutable runtime params that will continue to control the
      immutable Source after the Source has been built.
    links_back: If True, the Source will have a `source_models` field linking
      back to its SourceModels.
  """

  runtime_params: runtime_params_lib.RuntimeParams
  links_back: bool

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    # pylint: disable = g-doc-args
    """When called, the SourceBuilder builds a Source.

    This signature is used just to make pytype recognize SourceBuilders are
    callable. Actual SourceBuilders take either no args or if `links_back`
    they take a `source_models` argument.
    """
    ...


def is_source_builder(obj, raise_if_false: bool = False) -> bool:
  """Runtime type guard function for source builders.

  Args:
    obj: The object to type check.
    raise_if_false: If true, raises a TypeError explaining why the object is not
      a Source Builder.

  Returns:
    bool: True if `obj` is a valid source builder
  """
  if not dataclasses.is_dataclass(obj):
    if raise_if_false:
      raise TypeError('Not a dataclass')
    return False
  if not hasattr(obj, 'runtime_params'):
    if raise_if_false:
      raise TypeError('Has no runtime_params')
    return False
  if not callable(obj):
    if raise_if_false:
      raise TypeError('Not callable')
    return False
  return True


def _convert_source_builder_to_init_kwargs(
    source_builder: ...,
    model_func: SourceProfileFunction | None,
) -> dict[str, Any]:
  """Returns a dict of init kwargs for the source builder."""
  source_init_kwargs = {}
  for field in dataclasses.fields(source_builder):
    if field.name == 'runtime_params':
      continue
    # for loop with getattr copies each field exactly as it exists.
    # dataclasses.asdict will recursivesly convert fields to dicts,
    # including turning custom dataclasses with __call__ methods into
    # plain Python dictionaries.
    source_init_kwargs[field.name] = getattr(source_builder, field.name)
  source_init_kwargs['model_func'] = model_func
  return source_init_kwargs


def make_source_builder(
    source_type: ...,
    runtime_params_type: ... = runtime_params_lib.RuntimeParams,
    model_func: SourceProfileFunction | None = None,
    links_back=False,
) -> SourceBuilderProtocol:
  """Given a Source type, returns a Builder for that type.

  Builders are factories that also hold dynamic runtime parameters.

  Args:
    source_type: The Source class to make a builder for.
    runtime_params_type: The type of `runtime_params` field which will be added
      to the builder dataclass.
    model_func: The model function to pass to the source.
    links_back: If True, the Source class has a `source_models` field linking
      back to the SourceModels object. This must be passed to the builder's
      __call__ method.

  Returns:
    builder: a Builder dataclass for the given Source dataclass.
  """

  source_fields = dataclasses.fields(source_type)

  # Runtime params are mutable and must be in the builder only.
  # We have this check because earlier Sources held their runtime params so
  # a common problem is Sources that haven't removed theirs yet.
  for field in source_fields:
    if field.name == 'runtime_params':
      raise ValueError(
          'Source dataclasses must not have a `runtime_params` '
          f'field but {source_type} does.'
      )

  # Filter out fields that shouldn't be passed to constructor
  source_fields = [f for f in source_fields if f.init]

  if links_back:
    assert sum([f.name == 'source_models' for f in source_fields]) == 1
    source_fields = [f for f in source_fields if f.name != 'source_models']

  name_type_field_tuples = [
      (field.name, field.type, field) for field in source_fields
  ]

  runtime_params_ntf = (
      'runtime_params',
      runtime_params_type,
      dataclasses.field(default_factory=runtime_params_type),
  )

  new_field_ntfs = [runtime_params_ntf]
  builder_ntfs = name_type_field_tuples + new_field_ntfs
  builder_type_name = source_type.__name__ + 'Builder'  # pytype: disable=attribute-error

  def check_kwargs(source_init_kwargs, context_msg):
    for f in source_fields:
      v = source_init_kwargs[f.name]
      if isinstance(f.type, str):
        if f.type in [
            'tuple[AffectedCoreProfile, ...]',
            'tuple[source.AffectedCoreProfile, ...]',
        ]:
          assert isinstance(v, tuple)
          assert all([isinstance(var, AffectedCoreProfile) for var in v])
        elif f.type == 'tuple[runtime_params_lib.Mode, ...]':
          assert isinstance(v, tuple)
          assert all([isinstance(var, runtime_params_lib.Mode) for var in v])
        elif f.type == 'SourceProfileFunction | None':
          assert v is None or callable(v)
        elif f.type in [
            'source.SourceProfileFunction',
            'source_lib.SourceProfileFunction',
        ]:
          if not callable(v):
            raise TypeError(
                f'While {context_msg} {source_type} got field '
                f'{f.name} of type source.SoureProfileFunction '
                ' but was passed constructor argument with value '
                f'{v} of type {type(v)}. It is not callable, so '
                'it cannot be a SourceProfileFunction.'
            )
        elif f.type in [
            'source.SourceOutputShapeFunction',
            'SourceOutputShapeFunction',
        ]:
          if not callable(v):
            raise TypeError(
                f'While {context_msg} {source_type} got field '
                f'{f.name} of type source.SoureProfileFunction '
                ' but was passed constructor argument with value '
                f'{v} of type {type(v)}. It is not callable, so '
                'it cannot be a SourceProfileFunction.'
            )
        else:
          raise TypeError(f'Unrecognized type string: {f.type}')

      # Check if the field is a parameterized generic.
      # Python cannot check isinstance for parameterized generics, so we ignore
      # these cases for now.
      # For instance, if a field type is `tuple[float, ...]` and the value is
      # valid, like `(1, 2, 3)`, then `isinstance(v, f.type)` would raise a
      # TypeError.
      elif (
          type(f.type) == types.GenericAlias  # pylint: disable=unidiomatic-typecheck
          or typing.get_origin(f.type) is not None
      ):
        # For `Union`s check if the value is a member of the union.
        # `typing.Union` is for types defined with `Union[A, B, C]` syntax.
        # `types.UnionType` is for types defined with `A | B | C` syntax.
        if typing.get_origin(f.type) in [typing.Union, types.UnionType]:
          if not isinstance(v, typing.get_args(f.type)):
            raise TypeError(
                f'While {context_msg} {source_type} got argument '
                f'{f.name} of type {type(v)} but expected '
                f'{f.type}).'
            )
      else:
        try:
          type_works = isinstance(v, f.type)
        except TypeError as exc:
          raise TypeError(
              f'While {context_msg} {source_type} got field '
              f'{f.name} whose type is {f.type} of type'
              f'{type(f.type)}. This is not a valid type.'
          ) from exc
        if not type_works:
          raise TypeError(
              f'While {context_msg} {source_type} got argument '
              f'{f.name} of type {type(v)} but expected '
              f'{f.type}).'
          )

  # pylint doesn't like this function name because it doesn't realize
  # this function is to be installed in a class
  def __post_init__(self):  # pylint:disable=invalid-name
    source_init_kwargs = _convert_source_builder_to_init_kwargs(
        self, model_func
    )
    check_kwargs(source_init_kwargs, 'making builder')
    # check_kwargs checks only the kwargs to Source, not SourceBuilder,
    # so it doesn't check "runtime_params"
    runtime_params = self.runtime_params
    if not isinstance(runtime_params, runtime_params_type):
      raise TypeError(
          f'Expected {runtime_params_type}, got {type(runtime_params)}'
      )

  def check_source(source):
    """Check that the result is a valid Source."""
    # `dataclasses` module is designed to operate on instances, not
    # types, so we have to do all this type analysis in the post init
    if not dataclasses.is_dataclass(source):
      raise TypeError(f'{source_type} is not a dataclass type')

    if not getattr(source, '__dataclass_params__').eq:
      raise TypeError(f'{source_type} needs eq=True')

    if not getattr(source, '__dataclass_params__').frozen:
      raise TypeError(f'{source_type} needs frozen=True')

  if links_back:

    def build_source(self, source_models):
      source_init_kwargs = _convert_source_builder_to_init_kwargs(
          self,
          model_func,
      )
      source_init_kwargs['source_models'] = source_models
      check_kwargs(source_init_kwargs, 'building')
      source = source_type(**source_init_kwargs)
      check_source(source)
      return source

  else:

    def build_source(self):
      source_init_kwargs = _convert_source_builder_to_init_kwargs(
          self,
          model_func,
      )
      check_kwargs(source_init_kwargs, 'building')
      source = source_type(**source_init_kwargs)
      check_source(source)
      return source

  return dataclasses.make_dataclass(
      builder_type_name,
      builder_ntfs,
      namespace={
          '__call__': build_source,
          'links_back': links_back,
          '__post_init__': __post_init__,
      },
      frozen=False,  # One role of the Builder class is to hold
      # the mutable runtime params
      kw_only=True,
  )
