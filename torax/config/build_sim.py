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

import copy
from typing import Any

from torax import geometry
from torax import sim as sim_lib
from torax.config import config_args
from torax.config import runtime_params as runtime_params_lib
from torax.sources import default_sources
from torax.sources import formula_config
from torax.sources import formulas
from torax.sources import runtime_params as source_runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import array_time_step_calculator
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.time_step_calculator import time_step_calculator as time_step_calculator_lib
from torax.transport_model import constant as constant_transport
from torax.transport_model import critical_gradient as critical_gradient_transport
from torax.transport_model import qlknn_wrapper
from torax.transport_model import transport_model as transport_model_lib


def build_sim_from_config(
    config: dict[str, Any],
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

   -  `runtime_params`: `build_runtime_params_from_config()`
   -  `geometry`: `build_geometry_from_config()`
   -  `sources`: `build_sources_from_config()`
   -  `transport`: `build_transport_model_from_config()`
   -  `stepper`: `build_stepper_builder_from_config()`
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
  runtime_params = build_runtime_params_from_config(config['runtime_params'])
  return sim_lib.build_sim_object(
      runtime_params=runtime_params,
      geo=build_geometry_from_config(config['geometry'], runtime_params),
      source_models=build_sources_from_config(config['sources']),
      transport_model=build_transport_model_from_config(config['transport']),
      stepper_builder=build_stepper_builder_from_config(config['stepper']),
      time_step_calculator=build_time_step_calculator_from_config(
          config['time_step_calculator']
      ),
  )


def build_runtime_params_from_config(
    general_runtime_params_config: dict[str, Any],
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


def build_geometry_from_config(
    geometry_config: dict[str, Any] | str,
    runtime_params: runtime_params_lib.GeneralRuntimeParams | None = None,
) -> geometry.Geometry:
  """Builds a `Geometry` from the input config.

  The input config has one required key: `geometry_type`. Its value must be one
  of:

   -  "circular"
   -  "chease"

  Depending on the `geometry_type` given, there are different keys/values
  expected in the rest of the config. See the following functions to get a full
  list of the arguments exposed:

   -  `geometry.build_circular_geometry()`
   -  `geometry.build_chease_geometry()`

  Args:
    geometry_config: Python dictionary containing keys/values that map onto a
      `geometry` module function that builds a `Geometry` object.
    runtime_params: `GeneralRuntimeParams` that will be set as a kwarg for
      `runtime_params` for chease geometry.

  Returns:
    A `Geometry` based on the input config.
  """
  if isinstance(geometry_config, str):
    kwargs = {'geometry_type': geometry_config}
  else:
    if 'geometry_type' not in geometry_config:
      raise ValueError('geometry_type must be set in the input config.')
    # Do a shallow copy to keep references to the original objects while not
    # modifying the original config dict with the pop-statement below.
    kwargs = copy.copy(geometry_config)
  geometry_type = kwargs.pop('geometry_type').lower()  # Remove from kwargs.
  if geometry_type == 'circular':
    return geometry.build_circular_geometry(**kwargs)
  elif geometry_type == 'chease':
    if runtime_params is not None:
      kwargs['runtime_params'] = runtime_params
    return geometry.build_chease_geometry(**kwargs)
  raise ValueError(f'Unknown geometry type: {geometry_type}')


def build_sources_from_config(
    source_configs: dict[str, Any],
) -> source_models_lib.SourceModels:
  """Builds a `SourceModels` from the input config.

  The input config has an expected structure which maps onto TORAX sources.
  Each key in the input config maps to a single source, and its value maps onto
  that source's input runtime parameters. Different sources have different input
  parameters, so to know which parameters to use, see the following dataclass
  definitions (source names to dataclass):

  -  `j_bootstrap`: `source.bootstrap_current_source.RuntimeParams`
  -  `jext`: `source.external_current_source.RuntimeParams`
  -  `nbi_particle_source`:
     `source.electron_density_sources.NBIParticleRuntimeParams`
  -  `gas_puff_source`: `source.electron_density_sources.GasPuffRuntimeParams`
  -  `pellet_source`: `source.electron_density_sources.PelletRuntimeParams`
  -  `generic_ion_el_heat_source`:
      `source.generic_ion_el_heat_source.RuntimeParams`
  -  `fusion_heat_source`: `source.runtime_params.RuntimeParams`
  -  `ohmic_heat_source`: `source.runtime_params.RuntimeParams`
  -  `qei_source`: `source.qei_source.RuntimeParams`

  If the input config includes a key that does not match one of the keys listed
  above, an error is raised. Sources are turned off unless included in the input
  config.

  For the source `Mode` enum, the string name can be provided as input:

  .. code-block:: python

    {
        'j_bootstrap': {
            'mode': 'zero',  # turns it off.
        },
    }

  If the `mode` is set to `formula_based`, then the you can provide a
  `formula_type` key which may have the following values:

  -  `default`: Uses the default impl (if the source has one) (default)

    -  The other config args are based on the source's RuntimeParams object
       outlined above.

  -  `exponential`: Exponential profile.

    - The other config args are from `sources.formula_config.Exponential`.

  -  `gaussian`: Gaussian profile.

    - The other config args are from `sources.formula_config.Gaussian`.

  E.g. for an example heat source:

  .. code-block:: python

    {
        mode: 'formula',
        formula_type: 'gaussian',
        total: 120e6,  # total heating
        c1: 0.0,  # Source Gaussian central location (in normalized r)
        c2: 0.25,  # Gaussian width in normalized radial coordinates
        use_normalized_r: True,
    }

  If you have custom source implementations, you may update this funtion to
  handle those new sources and keys, or you may use the "advanced" configuration
  method and build your `SourceModel` object directly.

  Args:
    source_configs: Input config dict defining all sources, with a structure as
      described above.

  Returns:
    A `SourceModels`.

  Raises:
    ValueError if an input key doesn't match one of the source names defined
      above.
  """
  sources = {}
  ohmic_name = 'ohmic_heat_source'
  for source_name, source_config in source_configs.items():
    if source_name == ohmic_name:
      # The ohmic heat source requires a pointer to the fully constructed
      # SourceModels object, so we add that source after the rest are built.
      continue
    sources[source_name] = _build_single_source_from_config(
        source_name, source_config
    )
  source_models = source_models_lib.SourceModels(sources=sources)
  # Add the OhmicHeatSource if requested.
  if ohmic_name in source_configs:
    ohmic = _build_single_source_from_config(
        source_name=ohmic_name,
        source_config=source_configs[ohmic_name],
        extra_init_kwargs={'source_models': source_models},
    )
    source_models.add_source(source_name=ohmic_name, source=ohmic)
  return source_models


def _build_single_source_from_config(
    source_name: str,
    source_config: dict[str, Any],
    extra_init_kwargs: dict[str, Any] | None = None,
) -> source_lib.Source:
  """Builds a `Source` from the input config."""
  runtime_params = default_sources.get_default_runtime_params(
      source_name,
  )
  # Update the defaults with the config provided.
  source_config = copy.copy(source_config)
  if 'mode' in source_config:
    mode = source_runtime_params_lib.Mode[source_config.pop('mode').upper()]
    runtime_params.mode = mode
  formula = None
  if 'formula_type' in source_config:
    func = source_config.pop('formula_type').lower()
    if func == 'default':
      pass  # Nothing to do here.
    elif func == 'exponential':
      runtime_params.formula = config_args.recursive_replace(
          formula_config.Exponential(),
          ignore_extra_kwargs=True,
          **source_config,
      )
      formula = formulas.Exponential()
    elif func == 'gaussian':
      runtime_params.formula = config_args.recursive_replace(
          formula_config.Gaussian(),
          ignore_extra_kwargs=True,
          **source_config,
      )
      formula = formulas.Gaussian()
    else:
      raise ValueError(f'Unknown formula_type for source {source_name}: {func}')
  runtime_params = config_args.recursive_replace(
      runtime_params, ignore_extra_kwargs=True, **source_config
  )
  kwargs = {'runtime_params': runtime_params}
  if formula is not None:
    kwargs['formula'] = formula
  if extra_init_kwargs is not None:
    kwargs.update(extra_init_kwargs)
  # pylint: disable=missing-kwoa
  # pytype: disable=missing-parameter
  return default_sources.get_source_type(source_name)(**kwargs)
  # pylint: enable=missing-kwoa
  # pytype: enable=missing-parameter


def build_transport_model_from_config(
    transport_config: dict[str, Any] | str,
) -> transport_model_lib.TransportModel:
  """Builds a `TransportModel` from the input config.

  The input config has one required key, `transport_model`, which can have the
  following values:

  -  `qlknn`: QLKNN transport.

    -  See `transport_model.qlknn_wrapper.RuntimeParams` for model-specific
       params.

  -  `constant`: Constant transport

    -  See `transport_model.constant.RuntimeParams` for model-specific
       params.

  -  `CGM`: Critical gradient transport
 
    -  See `transport_model.critical_gradient.RuntimeParams` for model-specific
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
    A `TransportModel` object.
  """
  if isinstance(transport_config, str):
    transport_config = {'transport_model': transport_config}
  else:
    if 'transport_model' not in transport_config:
      raise ValueError('transport_model must be set in the input config.')
    transport_config = copy.copy(transport_config)
  transport_model = transport_config.pop('transport_model')
  if transport_model == 'qlknn':
    qlknn_params = transport_config.pop('qlknn_params', {})
    qlknn_params.update(transport_config)
    # Remove params from the other models, if present.
    qlknn_params.pop('constant_params', None)
    qlknn_params.pop('cgm_params', None)
    return qlknn_wrapper.QLKNNTransportModel(
        runtime_params=config_args.recursive_replace(
            qlknn_wrapper.RuntimeParams(),
            **qlknn_params,
        )
    )
  elif transport_model == 'constant':
    constant_params = transport_config.pop('constant_params', {})
    constant_params.update(transport_config)
    # Remove params from the other models, if present.
    constant_params.pop('qlknn_params', None)
    constant_params.pop('cgm_params', None)
    return constant_transport.ConstantTransportModel(
        runtime_params=config_args.recursive_replace(
            constant_transport.RuntimeParams(),
            **constant_params,
        )
    )
  elif transport_model == 'CGM':
    cgm_params = transport_config.pop('cgm_params', {})
    cgm_params.update(transport_config)
    # Remove params from the other models, if present.
    cgm_params.pop('qlknn_params', None)
    cgm_params.pop('constant_params', None)
    return critical_gradient_transport.CriticalGradientModel(
        runtime_params=config_args.recursive_replace(
            critical_gradient_transport.RuntimeParams(),
            **cgm_params,
        )
    )
  raise ValueError(f'Unknown transport model: {transport_model}')


def build_stepper_builder_from_config(
    stepper_config: dict[str, Any],
) -> stepper_lib.StepperBuilder:
  """Builds a `StepperBuilder` from the input config.

  `Stepper` objects evolve the core profile state of the system. The input
  config has one required key, `stepper_type`, which must be one of the
  following values:

  -  `linear`: Linear theta method.

    - Additional config parameters are defined in `LinearRuntimeParams`.

  -  `newton_raphson`: Newton-Raphson nonlinear stepper.

    - Additional config parameters are defined in `NewtonRaphsonRuntimeParams`.

  - `optimizer`: jaxopt-based nonlinear stepper.

    - Additional config parameters are defined in `OptimizerRuntimeParams`.

  All steppers share some common config parameters defined in
  `stepper.runtime_params.RuntimeParams`. Stepper-specific params are defined in
  the dataclasses listed above.

  Args:
    stepper_config: Python dictionary containing arguments to build a stepper
      object and its runtime parameters, with the structure outlined above.

  Returns:
    A `StepperBuilder` object, configured with RuntimeParams defined based on
    the input config.

  Raises:
    ValueError if the `stepper_type` is unknown.
  """
  if isinstance(stepper_config, str):
    stepper_config = {'stepper_type': stepper_config}
  else:
    if 'stepper_type' not in stepper_config:
      raise ValueError('stepper_type must be set in the input config.')
    # Shallow copy so we don't modify the input config.
    stepper_config = copy.copy(stepper_config)
  stepper_type = stepper_config.pop('stepper_type')
  if stepper_type == 'linear':
    return linear_theta_method.LinearThetaMethodBuilder(
        runtime_params=config_args.recursive_replace(
            linear_theta_method.LinearRuntimeParams(),
            **stepper_config,
        )
    )
  elif stepper_type == 'newton_raphson':
    return nonlinear_theta_method.NewtonRaphsonThetaMethodBuilder(
        runtime_params=config_args.recursive_replace(
            nonlinear_theta_method.NewtonRaphsonRuntimeParams(),
            **stepper_config,
        )
    )
  elif stepper_type == 'optimizer':
    return nonlinear_theta_method.OptimizerThetaMethodBuilder(
        runtime_params=config_args.recursive_replace(
            nonlinear_theta_method.OptimizerRuntimeParams(),
            **stepper_config,
        )
    )
  raise ValueError(f'Unknown stepper type: {stepper_type}')


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
    time_step_calculator_config = copy.copy(time_step_calculator_config)
  calculator_type = time_step_calculator_config.pop('calculator_type')
  init_args = time_step_calculator_config.pop('init_kwargs', {})
  if calculator_type == 'fixed':
    return fixed_time_step_calculator.FixedTimeStepCalculator(**init_args)
  elif calculator_type == 'chi':
    return chi_time_step_calculator.ChiTimeStepCalculator(**init_args)
  elif calculator_type == 'array':
    return array_time_step_calculator.ArrayTimeStepCalculator(**init_args)
  raise ValueError(f'Unknown calculator type: {calculator_type}')
