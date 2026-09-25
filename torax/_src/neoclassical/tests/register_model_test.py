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
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.neoclassical import register_model
from torax._src.neoclassical import runtime_params as neoclassical_runtime_params_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base as poloidal_velocity_base
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.neoclassical.poloidal_velocity import zeros as poloidal_velocity_zeros
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib


class CustomConductivityModel(conductivity_base.ConductivityModel):

  def calculate_conductivity(
      self,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> conductivity_base.Conductivity:
    del core_profiles, analytical_cache
    return conductivity_base.Conductivity(
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

  def build_runtime_params(self) -> conductivity_runtime_params.RuntimeParams:
    return conductivity_runtime_params.RuntimeParams()


class CustomPoloidalVelocityModel(poloidal_velocity_base.PoloidalVelocityModel):

  def calculate_poloidal_velocity(
      self,
      runtime_params: poloidal_velocity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> poloidal_velocity_base.PoloidalVelocity:
    return poloidal_velocity_zeros.ZerosModel().calculate_poloidal_velocity(
        runtime_params, geometry, core_profiles, analytical_cache
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

  def build_runtime_params(
      self,
  ) -> poloidal_velocity_runtime_params.RuntimeParams:
    return poloidal_velocity_runtime_params.RuntimeParams(
        poloidal_velocity_multiplier=self.poloidal_velocity_multiplier
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CustomNeoclassicalRuntimeParams(
    neoclassical_runtime_params_lib.RuntimeParams
):
  """Custom runtime params for an integrated neoclassical model."""

  custom_sigma: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class CustomNeoclassicalModel(neoclassical_models.NeoclassicalModel):
  """Custom integrated neoclassical model that does not use sub-models."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> neoclassical_models.NeoclassicalOutputs:
    del core_profiles
    assert isinstance(
        runtime_params.neoclassical, CustomNeoclassicalRuntimeParams
    )
    sigma_val = runtime_params.neoclassical.custom_sigma
    return neoclassical_models.NeoclassicalOutputs(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            geometry
        ),
        conductivity=conductivity_base.Conductivity(
            sigma=jnp.full_like(geometry.rho_norm, sigma_val),
            sigma_face=jnp.full_like(geometry.rho_face_norm, sigma_val),
        ),
        transport=transport_coeffs_lib.NeoclassicalTransport.zeros(geometry),
        poloidal_velocity=poloidal_velocity_base.PoloidalVelocity.zeros(
            geometry
        ),
    )


class CustomTopLevelNeoclassicalConfig(
    neoclassical_pydantic_model.BaseNeoclassical
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

    config = default_configs.get_default_config_dict()
    config['neoclassical'] = {
        'conductivity': {'model_name': 'custom_conductivity'},
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(
        model, neoclassical_models.AnalyticalNeoclassicalModel
    )
    self.assertIsInstance(model.conductivity, CustomConductivityModel)

  def test_register_poloidal_velocity_submodel(self):
    register_model.register_poloidal_velocity_model(
        CustomPoloidalVelocityConfig
    )

    config = default_configs.get_default_config_dict()
    config['neoclassical'] = {
        'poloidal_velocity': {'model_name': 'custom_poloidal_velocity'},
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(
        model, neoclassical_models.AnalyticalNeoclassicalModel
    )
    self.assertIsInstance(
        model.poloidal_velocity, CustomPoloidalVelocityModel
    )

  def test_register_top_level_neoclassical_model(self):
    register_model.register_neoclassical_model(CustomTopLevelNeoclassicalConfig)

    config = default_configs.get_default_config_dict()
    config['neoclassical'] = {
        'model_name': 'custom_neoclassical',
        'custom_sigma': 123.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.neoclassical, CustomTopLevelNeoclassicalConfig
    )
    model = torax_config.neoclassical.build_model()
    self.assertIsInstance(model, CustomNeoclassicalModel)

    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=torax_config.sources.build_models(),
        neoclassical_model=model,
    )
    np.testing.assert_allclose(core_profiles.sigma, 123.0)

  def test_dynamic_registration_updates_discriminator(self):
    register_model.register_neoclassical_model(CustomTopLevelNeoclassicalConfig)

    config = default_configs.get_default_config_dict()
    config['neoclassical'] = {'model_name': 'invalid_name'}
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_name' found using 'model_name' does not match any"
        r".*'custom_neoclassical'",
    ):
      model_config.ToraxConfig.from_dict(config)


if __name__ == '__main__':
  absltest.main()
