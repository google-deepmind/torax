# Copyright 2026 DeepMind Technologies Limited
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
"""Tests for neoclassical model and sub-model registration."""

import dataclasses
from typing import Annotated, Literal

from absl.testing import absltest
import jax
import jax.numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import solver_profiles
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import config as neoclassical_config
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical import neoclassical_model
from torax._src.neoclassical import register_model
from torax._src.neoclassical import runtime_params as neoclassical_runtime_params_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base as poloidal_velocity_base
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.neoclassical.poloidal_velocity import zeros as poloidal_velocity_zeros
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib


class CustomConductivityModel(conductivity_base.ConductivityModel):

  def calculate_conductivity(
      self,
      runtime_params: conductivity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> state.Conductivity:
    del runtime_params, core_profiles, neoclassical_intermediates
    return state.Conductivity(
        sigma=jnp.ones_like(geometry.rho_norm),
        sigma_face=jnp.ones_like(geometry.rho_face_norm),
    )

  def __hash__(self) -> int:
    return hash(self.__class__)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, CustomConductivityModel)


class CustomConductivityConfig(conductivity_base.ConductivityModelConfig):
  model_name: Annotated[
      Literal['custom_conductivity'], torax_pydantic.JAX_STATIC
  ] = 'custom_conductivity'

  def build_model(self) -> CustomConductivityModel:
    return CustomConductivityModel()


class CustomPoloidalVelocityModel(poloidal_velocity_base.PoloidalVelocityModel):

  def calculate_poloidal_velocity(
      self,
      runtime_params: poloidal_velocity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> poloidal_velocity_base.PoloidalVelocity:
    return poloidal_velocity_zeros.ZerosModel().calculate_poloidal_velocity(
        runtime_params, geometry, core_profiles, neoclassical_intermediates
    )

  def __hash__(self) -> int:
    return hash(self.__class__)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, CustomPoloidalVelocityModel)


class CustomPoloidalVelocityConfig(
    poloidal_velocity_base.PoloidalVelocityModelConfig
):
  model_name: Annotated[
      Literal['custom_poloidal_velocity'], torax_pydantic.JAX_STATIC
  ] = 'custom_poloidal_velocity'

  def build_model(self) -> CustomPoloidalVelocityModel:
    return CustomPoloidalVelocityModel()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CustomNeoclassicalRuntimeParams(
    neoclassical_runtime_params_lib.RuntimeParams
):
  """Custom runtime params for an integrated neoclassical model."""

  custom_sigma: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class CustomNeoclassicalModel(neoclassical_model.NeoclassicalModel):
  """Custom integrated neoclassical model that does not use sub-models."""

  def calculate_conductivity_current_and_rotation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      fast_ions: tuple[fast_ion_lib.FastIon, ...] = (),
  ) -> tuple[
      state.Conductivity,
      bootstrap_current_base.BootstrapCurrent,
      poloidal_velocity_base.PoloidalVelocity,
      neoclassical_intermediates_lib.NeoclassicalIntermediates,
  ]:
    del core_profiles, fast_ions
    assert isinstance(
        runtime_params.neoclassical, CustomNeoclassicalRuntimeParams
    )
    sigma_val = runtime_params.neoclassical.custom_sigma
    return (
        state.Conductivity(
            sigma=jnp.full_like(geo.rho_norm, sigma_val),
            sigma_face=jnp.full_like(geo.rho_face_norm, sigma_val),
        ),
        bootstrap_current_base.BootstrapCurrent.zeros(geo),
        poloidal_velocity_base.PoloidalVelocity.zeros(geo),
        neoclassical_intermediates_lib.NeoclassicalIntermediates(),
    )

  def calculate_transport(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      intermediates: neoclassical_intermediates_lib.NeoclassicalIntermediates,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    del runtime_params, core_profiles
    assert isinstance(
        intermediates,
        neoclassical_intermediates_lib.NeoclassicalIntermediates,
    )
    return transport_coeffs_lib.NeoclassicalTransport.zeros(geo)


class CustomTopLevelNeoclassicalConfig(
    neoclassical_config.NeoclassicalConfigBase
):
  """Custom top-level neoclassical model config for testing."""

  model_name: Annotated[
      Literal['custom_neoclassical'], torax_pydantic.JAX_STATIC
  ] = 'custom_neoclassical'
  custom_sigma: float = 42.0

  def build_runtime_params(
      self,
  ) -> CustomNeoclassicalRuntimeParams:
    return CustomNeoclassicalRuntimeParams(custom_sigma=self.custom_sigma)

  def build_model(self) -> CustomNeoclassicalModel:
    return CustomNeoclassicalModel()


class RegisterNeoclassicalModelTest(absltest.TestCase):

  def test_register_conductivity_submodel(self):
    register_model.register_conductivity_model(CustomConductivityConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict['neoclassical'] = {
        'conductivity': {'model_name': 'custom_conductivity'},
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(model, neoclassical_model.AnalyticalNeoclassicalModel)
    self.assertIsInstance(model.conductivity, CustomConductivityModel)

  def test_register_poloidal_velocity_submodel(self):
    register_model.register_poloidal_velocity_model(
        CustomPoloidalVelocityConfig
    )

    config_dict = default_configs.get_default_config_dict()
    config_dict['neoclassical'] = {
        'poloidal_velocity': {'model_name': 'custom_poloidal_velocity'},
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(model, neoclassical_model.AnalyticalNeoclassicalModel)
    self.assertIsInstance(model.poloidal_velocity, CustomPoloidalVelocityModel)

  def test_register_top_level_neoclassical_model(self):
    register_model.register_neoclassical_model(CustomTopLevelNeoclassicalConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict['neoclassical'] = {
        'model_name': 'custom_neoclassical',
        'custom_sigma': 123.0,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    self.assertIsInstance(
        torax_config.neoclassical, CustomTopLevelNeoclassicalConfig
    )
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(model, CustomNeoclassicalModel)

  def test_dynamic_registration_updates_discriminator(self):
    register_model.register_neoclassical_model(CustomTopLevelNeoclassicalConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict['neoclassical'] = {'model_name': 'invalid_name'}
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_name' found using 'model_name' does not match any"
        r".*'custom_neoclassical'",
    ):
      config.ToraxConfig.from_dict(config_dict)


if __name__ == '__main__':
  absltest.main()
