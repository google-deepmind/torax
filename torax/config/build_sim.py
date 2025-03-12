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
import copy
from typing import Any

from torax import sim as sim_lib
from torax.config import config_args
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.time_step_calculator import time_step_calculator as time_step_calculator_lib
from torax.transport_model import bohm_gyrobohm as bohm_gyrobohm_transport
from torax.transport_model import constant as constant_transport
from torax.transport_model import critical_gradient as critical_gradient_transport
from torax.transport_model import qlknn_transport_model
from torax.transport_model import transport_model as transport_model_lib


# pylint: disable=g-import-not-at-top
try:
  from torax.transport_model import qualikiz_transport_model

  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = True
except ImportError:
  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = False
# pylint: enable=g-import-not-at-top
# pylint: disable=invalid-name


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
        time_step_calculator: {...},  # Optional
    }

  See the following functions' docstrings to learn more about their expected
  input config structures and what parameters are available to use.

   -  `runtime_params`: geometry_pydantic_model.Geometry
   -  `geometry`: `build_geometry_from_config()`
   -  `sources`: `build_sources_from_config()`
   -  `transport`: `build_transport_model_builder_from_config()`
   -  `stepper`: stepper_pydantic_model.Stepper
   -  `time_step_calculator`: `build_time_step_calculator_from_config()`

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
  runtime_params = build_runtime_params_from_config(config['runtime_params'])
  geo_provider = geometry_pydantic_model.Geometry.from_dict(
      config['geometry']
  ).build_provider

  if 'restart' in config:
    file_restart = runtime_params_lib.FileRestart(**config['restart'])
  else:
    file_restart = None

  return sim_lib.Sim.create(
      runtime_params=runtime_params,
      geometry_provider=geo_provider,
      sources=source_pydantic_model.Sources.from_dict(
          config['sources']
      ),
      transport_model_builder=build_transport_model_builder_from_config(
          config['transport']
      ),
      stepper=stepper_pydantic_model.Stepper.from_dict(config['stepper']),
      pedestal=pedestal_pydantic_model.Pedestal.from_dict(
          config['pedestal'] if 'pedestal' in config else {}
      ),
      time_step_calculator=build_time_step_calculator_from_config(
          config['time_step_calculator']
      ),
      file_restart=file_restart,
  )


def build_runtime_params_from_config(
    general_runtime_params_config: Mapping[str, Any],
) -> runtime_params_lib.GeneralRuntimeParams:
  """Builds `GeneralRuntimeParams` from the input config.

  The input config has a required structure which maps directly to the
  parameters of `GeneralRuntimeParams`. More info about the allowed keys/values,
  as well as information about how the parameters are nested, can be found in
  the `GeneralRuntimeParams` docstring and class definition.

  Args:
    general_runtime_params_config: Python dictionary containing keys/values that
      map onto `GeneralRuntimeParams` and its attributes.

  Returns:
    A `GeneralRuntimeParams` based on the input config.
  """
  return config_args.recursive_replace(
      runtime_params_lib.GeneralRuntimeParams(),
      **general_runtime_params_config,
  )


def build_transport_model_builder_from_config(
    transport_config: dict[str, Any] | str,
) -> transport_model_lib.TransportModelBuilder:
  """Builds a `TransportModelBuilder` from the input config.

  The input config has one required key, `transport_model`, which can have the
  following values:

  -  `qualikiz`: QuaLiKiz transport.

  -  `qlknn`: QLKNN transport.

    -  See `transport_model.qlknn_transport_model.RuntimeParams` for
       model-specific params.

  -  `constant`: Constant transport

    -  See `transport_model.constant.RuntimeParams` for model-specific
       params.

  -  `CGM`: Critical gradient transport

    -  See `transport_model.critical_gradient.RuntimeParams` for model-specific
       params.

  -  `bohm-gyrobohm`: Bohm-GyroBohm transport

    -  See `transport_model.bohm_gyrobohm.RuntimeParams` for model-specific
        params.

  For all transport models, there are certain parameters which are shared
  amongst all models, as defined by
  `transport_model.runtime_params.RuntimeParams`. For each type of transport,
  there are also model-specific params which are used. These params are nested
  under an additional layer and only used when the `transport_model` key is set
  to that value.

  For example:

  .. code-block:: python

    {
        'transport_model': 'qlknn',  # The QLKNN model will be built.

        # Some shared params.
        chimin: 0.05,
        chimax: 100.0,

        # QLKNN-specific params.
        # These are used because transport_model='qlknn'.
        qlknn_params: {
            include_ITG: True,
        }

        # Constant-specific params.
        # Ignored because of the transport_model value.
        constant_params: {...},
        # CGM-specific params.
        # Ignored because of the transport_model value.
        cgm_params: {...},
    }

  Args:
    transport_config: Python dict describing how to build a `TransportModel`
      with the structure outlined above.

  Returns:
    A `TransportModelBuilder` object.
  """
  if isinstance(transport_config, str):
    transport_config = {'transport_model': transport_config}
  else:
    if 'transport_model' not in transport_config:
      raise ValueError('transport_model must be set in the input config.')
    transport_config = copy.deepcopy(transport_config)
  transport_model = transport_config.pop('transport_model')
  if transport_model == 'qlknn':
    qlknn_params = transport_config.pop('qlknn_params', {})
    if not isinstance(qlknn_params, dict):
      raise ValueError('qlknn_params must be a dict.')
    if 'model_path' in qlknn_params:
      model_path = qlknn_params.pop('model_path')
    else:
      model_path = qlknn_transport_model.get_default_model_path()
    qlknn_params.update(transport_config)
    # Remove params from the other models, if present.
    qlknn_params.pop('constant_params', None)
    qlknn_params.pop('cgm_params', None)
    qlknn_params.pop('bohm-gyrobohm_params', None)
    qlknn_params.pop('qualikiz_params', None)
    return qlknn_transport_model.QLKNNTransportModelBuilder(
        runtime_params=config_args.recursive_replace(
            qlknn_transport_model.get_default_runtime_params_from_model_path(
                model_path
            ),
            **qlknn_params,
        ),
        model_path=model_path,
    )
  elif transport_model == 'constant':
    constant_params = transport_config.pop('constant_params', {})
    if not isinstance(constant_params, dict):
      raise ValueError('constant_params must be a dict.')
    constant_params.update(transport_config)
    # Remove params from the other models, if present.
    constant_params.pop('qlknn_params', None)
    constant_params.pop('cgm_params', None)
    constant_params.pop('bohm-gyrobohm_params', None)
    constant_params.pop('qualikiz_params', None)
    return constant_transport.ConstantTransportModelBuilder(
        runtime_params=config_args.recursive_replace(
            constant_transport.RuntimeParams(),
            **constant_params,
        )
    )
  elif transport_model == 'CGM':
    cgm_params = transport_config.pop('cgm_params', {})
    if not isinstance(cgm_params, dict):
      raise ValueError('cgm_params must be a dict.')
    cgm_params.update(transport_config)
    # Remove params from the other models, if present.
    cgm_params.pop('qlknn_params', None)
    cgm_params.pop('constant_params', None)
    cgm_params.pop('bohm-gyrobohm_params', None)
    cgm_params.pop('qualikiz_params', None)

    return critical_gradient_transport.CriticalGradientModelBuilder(
        runtime_params=config_args.recursive_replace(
            critical_gradient_transport.RuntimeParams(),
            **cgm_params,
        )
    )
  elif transport_model == 'bohm-gyrobohm':
    bgb_params = transport_config.pop('bohm-gyrobohm_params', {})
    if not isinstance(bgb_params, dict):
      raise ValueError('bohm-gyrobohm_params must be a dict.')
    bgb_params.update(transport_config)
    # Remove params from the other models, if present.
    bgb_params.pop('qlknn_params', None)
    bgb_params.pop('constant_params', None)
    bgb_params.pop('cgm_params', None)
    return bohm_gyrobohm_transport.BohmGyroBohmModelBuilder(
        runtime_params=config_args.recursive_replace(
            bohm_gyrobohm_transport.RuntimeParams(),
            **bgb_params,
        )
    )
  elif transport_model == 'qualikiz':
    if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
      raise ValueError(
          'Qualikiz transport model is not available. Possible issue is that'
          ' the QuaLiKiz Pythontools are not installed.'
      )
    qualikiz_params = dict(transport_config.pop('qualikiz_params', {}))
    qualikiz_params.update(transport_config)
    # Remove params from the other models, if present.
    qualikiz_params.pop('qlknn_params', None)
    qualikiz_params.pop('cgm_params', None)
    qualikiz_params.pop('constant_params', None)
    qualikiz_params.pop('bohm-gyrobohm_params', None)
    # pylint: disable=undefined-variable
    return qualikiz_transport_model.QualikizTransportModelBuilder(
        runtime_params=config_args.recursive_replace(
            qualikiz_transport_model.RuntimeParams(),
            **qualikiz_params,
        )
    )
  # pylint: enable=undefined-variable
  raise ValueError(f'Unknown transport model: {transport_model}')


def build_time_step_calculator_from_config(
    time_step_calculator_config: dict[str, Any] | str,
) -> time_step_calculator_lib.TimeStepCalculator:
  """Builds a `TimeStepCalculator` from the input config.

  `TimeStepCalculator` calculates the dt for the time step and decides whether
  the sim is done. The input config has one required key, `calculator_type`,
  which must be one of the following values:

   -  `fixed`: Maps to `FixedTimeStepCalculator`.
   -  `chi`: Maps to `ChiTimeStepCalculator`.
   -  `array`: Maps to `ArrayTimeStepCalculator`.

  If the time-step calculator chosen has any constructor arguments, those can be
  passed to the `init_kwargs` key in the input config:

  .. code-block:: python

    {
        'calculator_type': 'array',
        'init_kwargs': {
            'array': [0, 0.1, 0.2, 0.4, 0.8],
        }
    }

  Args:
    time_step_calculator_config: Python dictionary configuring a
      `TimeStepCalculator` with the structure shown above.

  Returns:
    A `TimeStepCalculator`.

  Raises:
    ValueError if the `calculator_type` is not one of the ones listed above.
  """
  if isinstance(time_step_calculator_config, str):
    time_step_calculator_config = {
        'calculator_type': time_step_calculator_config
    }
  else:
    if 'calculator_type' not in time_step_calculator_config:
      raise ValueError('calculator_type must be set in the input config.')
    time_step_calculator_config = copy.deepcopy(time_step_calculator_config)
  calculator_type = time_step_calculator_config.pop('calculator_type')
  init_args = time_step_calculator_config.pop('init_kwargs', {})
  if calculator_type == 'fixed':
    return fixed_time_step_calculator.FixedTimeStepCalculator(**init_args)
  elif calculator_type == 'chi':
    return chi_time_step_calculator.ChiTimeStepCalculator(**init_args)
  raise ValueError(f'Unknown calculator type: {calculator_type}')
