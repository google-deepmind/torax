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
"""Surrogate model for ion-cyclotron resonance heating (ICRH) model."""
import dataclasses
import functools
import json
import logging
import os  # pylint: disable=unused-import
from typing import Any, ClassVar, Final, Literal, Sequence

import chex
import flax.linen as nn
import jax
from jax import numpy as jnp
import jaxtyping as jt
import pydantic
from torax import array_typing
from torax import jax_utils
from torax import math_utils
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# Internal import.


# Default value for the model function to be used for the ion cyclotron
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'toric_nn'

# If no path is set in either the config or the environment variable, use
# this path.
_DEFAULT_MODEL_PATH = '~/toric_surrogate/TORIC_MLP_v1/toricnn.json'
_TORIC_GRID_SIZE = 297
_HELIUM3_ID = 'He3'
_TRITIUM_SECOND_HARMONIC_ID = '2T'
_ELECTRON_ID = 'e'


def _from_json(json_file) -> dict[str, Any]:
  """Load the model config and weights from a JSON file."""
  if not os.path.exists(json_file):
    raise FileNotFoundError(f'Model file {json_file} does not exist.')
  with open(json_file) as file_:
    model_dict = json.load(file_)
  return model_dict


# pylint: disable=invalid-name
# Many of the variables below are named to match the physics quantities
# as defined by the TORIC ICRF solver, so we keep their naming for consistency.
@chex.dataclass(frozen=True)
class ToricNNInputs:
  """Inputs to the ToricNN model."""

  # ICRF wave frequency in MHz, training range = 119 to 121.
  frequency: array_typing.ScalarFloat
  # Volume average temperature in keV, training range = 1.5 to 8.5.
  volume_average_temperature: array_typing.ScalarFloat
  # Volume average density in 10^20 m^-3, training range = 1.1 to 5.1.
  volume_average_density: array_typing.ScalarFloat
  # He3 minority concentration relative to the electron density in %,
  # training range = 1 to 5.
  minority_concentration: array_typing.ScalarFloat
  # Distance from last closed flux surface (LCFS) to the inner wall in m,
  # training range = 0 to 0.03.
  gap_inner: array_typing.ScalarFloat
  # Distance from LCFS to the outer midplane limiter in m,
  # training range = 0 to 0.05.
  gap_outer: array_typing.ScalarFloat
  # Vertical position of magnetic axis in m, training range = -0.05 to 0.05.
  z0: array_typing.ScalarFloat
  # Temperature profile peaking factor, training range = 2 to 3.
  temperature_peaking_factor: array_typing.ScalarFloat
  # Density profile peaking factor, training range = 1.15 to 1.65.
  density_peaking_factor: array_typing.ScalarFloat
  # Toroidal magnetic field on axis in T, training range = 11.8 to 12.5.
  B_0: array_typing.ScalarFloat


@chex.dataclass(frozen=True)
class ToricNNOutputs:
  """Outputs from the ToricNN model."""

  # Power deposition on helium-3 in MW/m^3/MW_{abs}.
  power_deposition_He3: array_typing.ArrayFloat
  # Power deposition on tritium (second harmonic) in MW/m^3/MW_{abs}.
  power_deposition_2T: array_typing.ArrayFloat
  # Power deposition on electrons in MW/m^3/MW_{abs}.
  power_deposition_e: array_typing.ArrayFloat


class _ToricNN(nn.Module):
  """Surrogate heating model trained on TORIC ICRF solver simulation.

  This model takes input parameters from the `ToricNNInputs` class and outputs
  power deposition profiles for helium-3, tritium (second harmonic) and
  electrons on a radial grid.

  This Flax module is not intended to be used directly but rather through the
  `ToricNNWrapper` class.

  The modelling approach is described in:
  https://iopscience.iop.org/article/10.1088/1741-4326/ad645d/pdf. The model
  is trained on regression outputs from the TORIC ICRF solver. PCA is applied
  to the outputs of the solver to reduce the dimensionality of the model.

  The structure of the model consistents of:
  - Scaling and normalisation of the input parameters.
  - An MLP transforming the scaled inputs.
  - A projection back to true values using the PCA coefficients.
  """

  # Hidden layer sizes for the MLP.
  hidden_sizes: Sequence[int]
  # Number of PCA coefficients used by ToricNN.
  pca_coeffs: int
  # Input dimensionality of the ToricNN model.
  input_dim: int
  # Number of radial nodes in output of the ToricNN model.
  radial_nodes: int

  def setup(self):
    """Setup the parameters of the ToricNN model."""
    self.scaler_mean = self.param(
        'scaler_mean',
        jax.random.normal,
        (self.input_dim,),
    )
    self.scaler_scale = self.param(
        'scaler_scale',
        jax.random.normal,
        (self.input_dim,),
    )
    self.pca_components = self.param(
        'pca_components',
        jax.random.normal,
        (
            self.pca_coeffs,
            self.radial_nodes,
        ),
    )
    self.pca_mean = self.param(
        'pca_mean',
        jax.random.normal,
        (self.radial_nodes,),
    )

  @nn.compact
  def __call__(
      self,
      x: jt.Float32[jt.Array, 'B* {self.input_dim}'],
  ) -> jt.Float32[jt.Array, 'B* {self.radial_nodes}']:
    """Run a forward pass of the ToricNN model."""
    # Scale and normalise inputs.
    x = (x - self.scaler_mean) / self.scaler_scale

    # MLP.
    for hidden_size in self.hidden_sizes:
      x = nn.Dense(
          hidden_size,
      )(x)
      x = nn.relu(x)
    x = nn.Dense(
        self.pca_coeffs,
    )(x)

    x = x @ self.pca_components + self.pca_mean  # Project back to true values.
    x = x * (x > 0)  # Eliminate non-physical values for power deposition.
    return x


class ToricNNWrapper:
  """Wrapper for the Toric NN model.

  This wrapper is currently for a SPARC-specific ion cyclotron resosonanc
  heating scheme.

  TODO(b/378072116): Make the wrapper more general to work with other ICRH
  schemes and surrogate models.

  This wrapper is the main interface for interacting with the Toric NN model.
  for making predictions of heating power deposition profiles given
  `ToricNNInputs`.

  The wrapper constructs 3 separate instances of the `_ToricNN` class, one for
  each simulated output (Helium-3, 2nd-harmonic tritium and electrons).
  """

  def __init__(self, path: str | None = None):
    if path is None:
      path = _DEFAULT_MODEL_PATH
    self._path = path
    logging.info('Loading ToricNN model from %s', path)
    model_config = _from_json(path)
    self.model_config = model_config

    self._params = {}
    self.power_deposition_network = self._load_network()
    self.power_deposition_He3_params = self._load_params(_HELIUM3_ID)
    self.power_deposition_2T_params = self._load_params(
        _TRITIUM_SECOND_HARMONIC_ID
    )
    self.power_deposition_e_params = self._load_params(_ELECTRON_ID)
    logging.info('Loaded ToricNN model from %s', path)

  def _load_network(self) -> _ToricNN:
    return _ToricNN(
        hidden_sizes=self.model_config['hidden_sizes'],
        pca_coeffs=self.model_config['pca_coeffs'],
        input_dim=self.model_config['input_dim'],
        radial_nodes=self.model_config['radial_nodes'],
    )

  def _load_params(self, network_name: str) -> dict[str, Any]:
    """Load a ToricNN network and its parameters."""
    params = {}
    params['params'] = self.model_config[f'{network_name}']
    for i in range(len(self.model_config['hidden_sizes']) + 1):
      params['params'][f'Dense_{i}']['kernel'] = jnp.array(
          self.model_config[f'{network_name}'][f'Dense_{i}']['kernel']
      )
      params['params'][f'Dense_{i}']['bias'] = jnp.array(
          self.model_config[f'{network_name}'][f'Dense_{i}']['bias']
      )
    params['params']['pca_components'] = jnp.array(
        self.model_config[f'{network_name}']['pca_components']
    )
    params['params']['pca_mean'] = jnp.array(
        self.model_config[f'{network_name}']['pca_mean']
    )
    params['params']['scaler_mean'] = jnp.array(
        self.model_config[f'{network_name}']['scaler_mean']
    )
    params['params']['scaler_scale'] = jnp.array(
        self.model_config[f'{network_name}']['scaler_scale']
    )
    return params

  def __hash__(self) -> int:
    return hash(self._path)

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return isinstance(other, ToricNNWrapper)


@functools.partial(jax_utils.jit, static_argnames='toric_nn')
def _toric_nn_predict(
    toric_nn: ToricNNWrapper,
    inputs: ToricNNInputs,
) -> ToricNNOutputs:
  """Make a prediction given the inputs."""
  inputs = jnp.array(
      [
          inputs.frequency,
          inputs.volume_average_temperature,
          inputs.volume_average_density,
          inputs.minority_concentration,
          inputs.gap_inner,
          inputs.gap_outer,
          inputs.z0,
          inputs.temperature_peaking_factor,
          inputs.density_peaking_factor,
          inputs.B_0,
      ],
      dtype=jax_utils.get_dtype(),
  )
  outputs_He3 = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_He3_params, inputs
  )
  outputs_2T = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_2T_params, inputs
  )
  outputs_e = toric_nn.power_deposition_network.apply(
      toric_nn.power_deposition_e_params, inputs
  )
  return ToricNNOutputs(
      power_deposition_He3=outputs_He3,
      power_deposition_2T=outputs_2T,
      power_deposition_e=outputs_e,
  )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  frequency: array_typing.ScalarFloat
  minority_concentration: array_typing.ScalarFloat
  P_total: array_typing.ScalarFloat
  absorption_fraction: array_typing.ScalarFloat
  wall_inner: float
  wall_outer: float


def _helium3_tail_temperature(
    power_deposition_he3: jax.Array,
    core_profiles: state.CoreProfiles,
    minority_concentration: float,
    P_total: float,
) -> jax.Array:
  """Use a "Stix distribution" to estimate the tail temperature of He3."""
  helium3_mass = 3.016
  helium3_charge = 2
  helium3_fraction = minority_concentration / 100  # Min conc provided in %.
  absorbed_power_density = power_deposition_he3 * P_total
  n_e20 = core_profiles.n_e.value * core_profiles.density_reference / 1e20
  # Use a "Stix distribution" [Stix, Nuc. Fus. 1975] to model the non-thermal
  # He3 distribution based on an analytic solution to the FP equation.
  xi = (
      0.24
      * jnp.sqrt(core_profiles.T_e.value)
      * helium3_mass
      * absorbed_power_density
  ) / (n_e20**2 * helium3_charge**2 * helium3_fraction)
  return core_profiles.T_e.value * (1 + xi)


def icrh_model_func(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
    toric_nn: ToricNNWrapper,
) -> tuple[chex.Array, ...]:
  """Compute ion/electron heat source terms."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

  # Construct inputs for ToricNN.
  volume_average_temperature = math_utils.volume_average(
      core_profiles.T_e.value, geo
  )
  volume_average_density = math_utils.volume_average(
      core_profiles.n_e.value, geo
  )

  # Peaking factors are core w.r.t volume averages.
  temperature_peaking_factor = (
      core_profiles.T_e.value[0] / volume_average_temperature
  )
  density_peaking_factor = core_profiles.n_e.value[0] / volume_average_density
  Router = geo.R_major + geo.a_minor
  Rinner = geo.R_major - geo.a_minor
  # Assumption: inner and outer gaps are not functions of z0.
  # This is a good assumption for the inner gap but perhaps less good for the
  # outer gap where there is significant curvature to the outer limiter.
  gap_inner = Rinner - dynamic_source_runtime_params.wall_inner
  gap_outer = dynamic_source_runtime_params.wall_outer - Router
  toric_inputs = ToricNNInputs(
      frequency=dynamic_source_runtime_params.frequency,
      volume_average_temperature=volume_average_temperature,
      volume_average_density=volume_average_density,
      minority_concentration=dynamic_source_runtime_params.minority_concentration,
      gap_inner=gap_inner,
      gap_outer=gap_outer,
      z0=geo.z_magnetic_axis(),
      temperature_peaking_factor=temperature_peaking_factor,
      density_peaking_factor=density_peaking_factor,
      B_0=geo.B_0,
  )

  toric_nn_outputs = _toric_nn_predict(toric_nn, toric_inputs)
  toric_grid = jnp.linspace(0.0, 1.0, _TORIC_GRID_SIZE)

  # Ideally total ICRH power should equal one but normalise if not.
  power_deposition_he3 = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_He3,
  )
  power_deposition_e = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_e,
  )
  power_deposition_2T = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_2T,
  )
  power_deposition_all = (
      power_deposition_2T + power_deposition_e + power_deposition_he3
  )

  total_power_deposition = math_utils.volume_integration(
      power_deposition_all, geo
  )
  power_deposition_he3 /= total_power_deposition
  power_deposition_e /= total_power_deposition
  power_deposition_2T /= total_power_deposition

  # For helium-3 we use a "Stix distribution" to model the non-thermal He3 tail.
  helium3_birth_energy = _helium3_tail_temperature(
      power_deposition_he3,
      core_profiles,
      dynamic_source_runtime_params.minority_concentration,
      dynamic_source_runtime_params.P_total / 1e6,  # required in MW.
  )
  helium3_mass = 3.016
  frac_ion_heating = collisions.fast_ion_fractional_heating_formula(
      helium3_birth_energy,
      core_profiles.T_e.value,
      helium3_mass,
  )
  absorbed_power = (
      dynamic_source_runtime_params.P_total
      * dynamic_source_runtime_params.absorption_fraction
  )
  source_ion = power_deposition_he3 * frac_ion_heating * absorbed_power
  source_el = power_deposition_he3 * (1 - frac_ion_heating) * absorbed_power

  # Assume that all the power from the electron power profile goes to electrons.
  source_el += power_deposition_e * absorbed_power

  # Assume that all the power from the tritium power profile goes to ions.
  source_ion += power_deposition_2T * absorbed_power

  return (source_ion, source_el)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class IonCyclotronSource(source.Source):
  """Ion cyclotron source with surrogate model."""

  SOURCE_NAME: ClassVar[str] = 'icrh'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )


# Cache the result of this function to avoid re-creating the partial function
# every time it is called and ensure we hit the same JAX compile cache (as
# model_func) is part of the key.
# maxsize=1 is sufficient as the ToricNNWrapper only changes if a new path
# is provided. This is not expected to happen very often.
@functools.lru_cache(maxsize=1)
def _icrh_model_func_with_toric_nn(
    toric_nn: ToricNNWrapper,
) -> source.SourceProfileFunction:
  """Returns a function that computes the ICRH source terms given a ToricNN."""
  return functools.partial(
      icrh_model_func,
      toric_nn=toric_nn,
  )


class IonCyclotronSourceConfig(base.SourceModelBase):
  """Configuration for the IonCyclotronSource.

  Attributes:
    model_path: Path to JSON weights and model config of ToricNN model.
    wall_inner: Inner radial location of first wall at plasma midplane level
      [m].
    wall_outer: Outer radial location of first wall at plasma midplane level
      [m].
    frequency: ICRF wave frequency [Hz].
    minority_concentration: He3 minority concentration relative to the electron
      density in %.
    P_total: Total heating power [W].
    absorption_fraction: Fraction of absorbed power.
  """

  model_name: Literal['toric_nn'] = 'toric_nn'
  model_path: str | None = None
  wall_inner: torax_pydantic.Meter = 1.24
  wall_outer: torax_pydantic.Meter = 2.43
  frequency: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  minority_concentration: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(3.0)
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      10e6
  )
  absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @pydantic.model_validator(mode='after')
  def _load_toric_nn(self) -> typing_extensions.Self:
    self._toric_nn = ToricNNWrapper(self.model_path)
    return self

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return _icrh_model_func_with_toric_nn(self._toric_nn)

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        wall_inner=self.wall_inner,
        wall_outer=self.wall_outer,
        frequency=self.frequency.get_value(t),
        minority_concentration=self.minority_concentration.get_value(t),
        P_total=self.P_total.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
    )

  def build_source(self) -> IonCyclotronSource:
    return IonCyclotronSource(model_func=self.model_func)
