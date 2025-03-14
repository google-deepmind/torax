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

"""Functions to build sim.Sim objects, which are used to run TORAX."""

from collections.abc import Mapping
from typing import Any
from torax import sim as sim_lib
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import pydantic_model as transport_pydantic_model


def build_sim_from_config(
    config: Mapping[str, Any],
) -> sim_lib.Sim:
  """Builds a sim.Sim object from the given TORAX config.

  A TORAX config is a nested Python dictionary mapping to arguments of the Sim
  object, which is an intermediate data structure used to execute a TORAX
  simulation run.

  The dictionary has an expected set of keys and structure to it. If the input
  configuration does not match the expected structure, then an error is raised.

  See https://torax.readthedocs.io/en/latest/configuration.html for
  documentation on all input variables.

  High-level, the input config has the following keys:

  .. code-block:: python

    {
        runtime_params: {...},
        geometry: {...},
        sources: {...},
        transport: {...},
        stepper: {...},
        time_step_calculator: {...},
    }

  To learn more about the Sim object and its components, see `sim.Sim`'s class
  docstring.

  Args:
    config: Input config dictionary outlining the necessary components of the
      `Sim` object. The input config requires these keys: `runtime_params`,
      `geometry`, `sources`, `transport`, `stepper`, and `time_step_calculator`.

  Returns:
    `Sim` object ready to be run.
  Raises:
    ValueError if any config parameters are missing or incorrect.
  """
  missing_keys = []
  required_keys = [
      'runtime_params',
      'geometry',
      'sources',
      'transport',
      'stepper',
      'time_step_calculator',
  ]
  for key in required_keys:
    if key not in config:
      missing_keys.append(key)
  if missing_keys:
    raise ValueError(
        f'The following required keys are not in the input dict: {missing_keys}'
    )
  if (
      'set_pedestal' in config['runtime_params']['profile_conditions']
      and config['runtime_params']['profile_conditions']['set_pedestal']
      and 'pedestal' not in config
  ):
    raise ValueError(
        'The pedestal config is required if set_pedestal is True in the runtime'
        ' params. See'
        ' https://torax.readthedocs.io/en/latest/configuration.html#detailed-configuration-structure'
        ' for more info.'
    )
  geo_provider = geometry_pydantic_model.Geometry.from_dict(
      config['geometry']
  ).build_provider

  runtime_params = runtime_params_lib.GeneralRuntimeParams.from_dict(
      config['runtime_params']
  )
  torax_pydantic.set_grid(runtime_params, geo_provider.torax_mesh)

  if 'restart' in config:
    file_restart = runtime_params_lib.FileRestart(**config['restart'])
  else:
    file_restart = None

  return sim_lib.Sim.create(
      runtime_params=runtime_params,
      geometry_provider=geo_provider,
      sources=source_pydantic_model.Sources.from_dict(config['sources']),
      transport_model=transport_pydantic_model.Transport.from_dict(
          config['transport']
      ),
      stepper=stepper_pydantic_model.Stepper.from_dict(config['stepper']),
      pedestal=pedestal_pydantic_model.Pedestal.from_dict(
          config['pedestal'] if 'pedestal' in config else {}
      ),
      time_step_calculator=time_step_calculator_pydantic_model.TimeStepCalculator.from_dict(
          config['time_step_calculator']
      ).time_step_calculator,
      file_restart=file_restart,
  )
