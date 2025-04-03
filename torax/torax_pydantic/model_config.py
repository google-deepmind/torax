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

import logging
from typing import Any, Mapping
import pydantic
from torax import version
from torax.config import runtime_params as general_runtime_params
from torax.fvm import enums
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.mhd import pydantic_model as mhd_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax.torax_pydantic import file_restart as file_restart_pydantic_model
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import pydantic_model as transport_model_pydantic_model
import typing_extensions
from typing_extensions import Self


class ToraxConfig(torax_pydantic.BaseModelFrozen):
  """Base config class for Torax.

  Attributes:
    runtime_params: Config for the runtime parameters.
    geometry: Config for the geometry.
    pedestal: Config for the pedestal model.
    sources: Config for the sources.
    stepper: Config for the stepper.
    time_step_calculator: Config for the time step calculator.
    transport: Config for the transport model.
    mhd: Optional config for mhd models. If None, no MHD models are used.
    restart: Optional config for file restart. If None, no file restart is
      performed.
  """

  # TODO(b/401187494): Flatten the runtime_params config, is this nesting
  # doesn't add much value.
  runtime_params: general_runtime_params.GeneralRuntimeParams
  geometry: geometry_pydantic_model.Geometry
  pedestal: pedestal_pydantic_model.Pedestal
  sources: sources_pydantic_model.Sources
  stepper: stepper_pydantic_model.Stepper
  time_step_calculator: time_step_calculator_pydantic_model.TimeStepCalculator
  transport: transport_model_pydantic_model.Transport
  mhd: mhd_pydantic_model.MHD = mhd_pydantic_model.MHD()
  restart: file_restart_pydantic_model.FileRestart | None = pydantic.Field(
      default=None
  )

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    using_nonlinear_transport_model = (
        self.transport.transport_model_config.transport_model
        in ['qlknn', 'CGM']
    )
    using_linear_solver = isinstance(
        self.stepper.stepper_config, stepper_pydantic_model.LinearThetaMethod
    )
    initial_guess_mode_is_linear = (
        False  # pylint: disable=g-long-ternary
        if using_linear_solver
        else self.stepper.stepper_config.initial_guess_mode
        == enums.InitialGuessMode.LINEAR
    )

    if (
        using_nonlinear_transport_model
        and (using_linear_solver or initial_guess_mode_is_linear)
        and not self.stepper.stepper_config.use_pereverzev
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

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these nodes will be re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """

    self._update_fields(x)

    # If the nrho is updated, all model caches need to be invalidated and the
    # geometry mesh reset.
    if _is_nrho_updated(x):
      for model in self.submodels:
        model.clear_cached_properties()
      mesh = self.geometry.build_provider.torax_mesh
      torax_pydantic.set_grid(self, mesh, mode='force')

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


def _is_nrho_updated(x: Mapping[str, Any]) -> bool:
  for path in x.keys():
    chunks = path.split('.')
    if chunks[-1] == 'n_rho' and chunks[0] == 'geometry':
      return True
  return False
