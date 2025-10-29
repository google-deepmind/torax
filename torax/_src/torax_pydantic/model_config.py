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

"""Pydantic config for Torax."""

import copy
import logging
from typing import Any, Mapping

import pydantic
from torax._src import physics_models
from torax._src import version
from torax._src.config import numerics as numerics_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.edge import pydantic_model as edge_pydantic_model
from torax._src.fvm import enums
from torax._src.geometry import geometry
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
import typing_extensions
from typing_extensions import Self


class ToraxConfig(torax_pydantic.BaseModelFrozen):
  """Base config class for Torax.

  Attributes:
    profile_conditions: Config for the profile conditions.
    numerics: Config for the numerics.
    plasma_composition: Config for the plasma composition.
    geometry: Config for the geometry.
    pedestal: Config for the pedestal model. If an empty dictionary is passed
      in, the pedestal model will be set to `no_pedestal`.
    sources: Config for the sources.
    neoclassical: Config for the neoclassical models.
    solver: Config for the solver. If an empty dictionary is passed in, the
      solver model will be set to `linear`.
    transport: Config for the transport model. If an empty dictionary is passed
      in, the transport model will be set to `constant`.
    mhd: Optional config for mhd models. If None, no MHD models are used.
    time_step_calculator: Optional config for the time step calculator. If not
      provided the default chi time step calculator is used.
    restart: Optional config for file restart. If None, no file restart is
      performed.
  """

  profile_conditions: profile_conditions_lib.ProfileConditions
  numerics: numerics_lib.Numerics
  plasma_composition: plasma_composition_lib.PlasmaComposition
  geometry: geometry_pydantic_model.Geometry
  sources: sources_pydantic_model.Sources
  neoclassical: neoclassical_pydantic_model.Neoclassical = (
      neoclassical_pydantic_model.Neoclassical()  # pylint: disable=missing-kwoa
  )
  solver: solver_pydantic_model.SolverConfig = pydantic.Field(
      discriminator='solver_type'
  )
  transport: transport_model_pydantic_model.TransportConfig = pydantic.Field(
      discriminator='model_name'
  )
  pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
      discriminator='model_name'
  )
  mhd: mhd_pydantic_model.MHD = mhd_pydantic_model.MHD()
  edge: edge_pydantic_model.EdgeConfig | None = None
  time_step_calculator: (
      time_step_calculator_pydantic_model.TimeStepCalculator
  ) = time_step_calculator_pydantic_model.TimeStepCalculator()
  restart: file_restart_pydantic_model.FileRestart | None = pydantic.Field(
      default=None
  )

  def build_physics_models(self):
    edge_model = self.edge.build_edge_model() if self.edge else None
    return physics_models.PhysicsModels(
        pedestal_model=self.pedestal.build_pedestal_model(),
        source_models=self.sources.build_models(),
        transport_model=self.transport.build_transport_model(),
        neoclassical_models=self.neoclassical.build_models(),
        mhd_models=self.mhd.build_mhd_models(),
        edge_model=edge_model,
    )

  # TODO(b/434175938): Remove this once V1 API is deprecated
  @pydantic.model_validator(mode='before')
  @classmethod
  def _v1_compatibility(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if 'calcphibdot' in configurable_data['numerics']:
      calcphibdot = configurable_data['numerics']['calcphibdot']
      configurable_data['geometry']['calcphibdot'] = calcphibdot
      del configurable_data['numerics']['calcphibdot']
    return configurable_data

  @pydantic.model_validator(mode='before')
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if (
        isinstance(configurable_data['pedestal'], dict)
        and 'model_name' not in configurable_data['pedestal']
    ):
      configurable_data['pedestal']['model_name'] = 'no_pedestal'
    if (
        isinstance(configurable_data['transport'], dict)
        and 'model_name' not in configurable_data['transport']
    ):
      configurable_data['transport']['model_name'] = 'constant'
    if (
        isinstance(configurable_data['solver'], dict)
        and 'solver_type' not in configurable_data['solver']
    ):
      configurable_data['solver']['solver_type'] = 'linear'
    return configurable_data

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    using_nonlinear_transport_model = self.transport.model_name in [
        'qualikiz',
        'qlknn',
        'CGM',
    ]
    using_linear_solver = isinstance(
        self.solver, solver_pydantic_model.LinearThetaMethod
    )

    # pylint: disable=g-long-ternary
    # pylint: disable=attribute-error
    initial_guess_mode_is_linear = (
        False
        if using_linear_solver
        else self.solver.initial_guess_mode == enums.InitialGuessMode.LINEAR
    )

    if (
        using_nonlinear_transport_model
        and (using_linear_solver or initial_guess_mode_is_linear)
        and not self.solver.use_pereverzev
    ):
      logging.warning("""
          use_pereverzev=False in a configuration where setting
          use_pereverzev=True is recommended.

          A nonlinear transport model is used. However, a linear solver is also
          being used, either directly, or to provide an initial guess for a
          nonlinear solver.

          With this configuration, it is strongly recommended to set
          use_pereverzev=True to avoid numerical instability in the solver.
          """)
    return self

  @pydantic.model_validator(mode='after')
  def _check_psidot_and_evolve_current(self) -> typing_extensions.Self:
    """Warns if psidot is provided but evolve_current is True."""
    if (
        self.profile_conditions.psidot is not None
        and self.numerics.evolve_current
    ):
      logging.warning("""
          profile_conditions.psidot input is ignored as numerics.evolve_current
          is True.

          Prescribed psidot is only applied when current diffusion is off.
          """)
    return self

  @pydantic.model_validator(mode='after')
  def _check_edge_with_circular_geometry(self) -> typing_extensions.Self:
    """Validates that edge models are not used with CircularGeometry."""
    if (
        self.edge is not None
        and self.geometry.geometry_type == geometry.GeometryType.CIRCULAR
    ):
      raise ValueError(
          'Edge models are not supported for use with CircularGeometry.'
      )
    return self

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these ancestral models will be
    re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """

    old_mesh = self.geometry.build_provider.torax_mesh
    self._update_fields(x)
    new_mesh = self.geometry.build_provider.torax_mesh

    if old_mesh != new_mesh:
      # The grid has changed, e.g. due to a new n_rho.
      # Clear the cached properties of all submodels and update the grid.
      for model in self.submodels:
        model.clear_cached_properties()
      torax_pydantic.set_grid(self, new_mesh, mode='force')
    else:
      # Update the grid on any new models which are added and have not had their
      # grid set yet.
      torax_pydantic.set_grid(self, new_mesh, mode='relaxed')

  @pydantic.model_validator(mode='after')
  def _set_grid(self) -> Self:
    # Interpolated `TimeVaryingArray` objects require a mesh, only available
    # once the geometry provider is built. This could be done in the before
    # validator, but is harder than setting it after construction.
    mesh = self.geometry.build_provider.torax_mesh
    # Note that the grid could already be set, eg. if the config is serialized
    # and deserialized. In this case, we do not want to overwrite it nor fail
    # when trying to set it, which is why mode='relaxed'.
    torax_pydantic.set_grid(self, mesh, mode='relaxed')
    return self

  # This is primarily used for serialization, so the importer can check which
  # version of Torax was used to generate the serialized config.
  @pydantic.computed_field
  @property
  def torax_version(self) -> str:
    return version.TORAX_VERSION

  @pydantic.model_validator(mode='before')
  @classmethod
  def _remove_version_field(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if 'torax_version' in data:
        data = {k: v for k, v in data.items() if k != 'torax_version'}
    return data
