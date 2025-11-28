# Copyright 2025 DeepMind Technologies Limited
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

"""Tests for pedestal model registration."""

from typing import Annotated, Literal
from absl.testing import absltest
import chex
from torax._src import geometry
from torax._src import state
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import pydantic_model
from torax._src.pedestal_model import register_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
import jax.numpy as jnp


class RegisterModelTest(absltest.TestCase):
  """Tests for pedestal model registration functionality."""

  def test_register_custom_pedestal_model(self):
    """Test that a custom pedestal model can be registered and used."""

    # Define a simple custom pedestal model
    @chex.dataclass(frozen=True)
    class TestPedestalModel(pedestal_model.PedestalModel):
      """Test pedestal model."""

      def _call_implementation(
          self,
          runtime_params: 'TestRuntimeParams',
          geo: geometry.Geometry,
          core_profiles: state.CoreProfiles,
      ) -> pedestal_model.PedestalModelOutput:
        """Return fixed test values."""
        rho_norm_ped_top = 0.95
        rho_norm_ped_top_idx = jnp.argmin(
            jnp.abs(geo.rho_norm - rho_norm_ped_top)
        )

        return pedestal_model.PedestalModelOutput(
            rho_norm_ped_top=rho_norm_ped_top,
            rho_norm_ped_top_idx=rho_norm_ped_top_idx,
            T_i_ped=runtime_params.test_value,
            T_e_ped=runtime_params.test_value,
            n_e_ped=runtime_params.test_value * 1e20,
        )

    @chex.dataclass(frozen=True)
    class TestRuntimeParams(pedestal_runtime_params.RuntimeParams):
      """Test runtime parameters."""
      test_value: float = 42.0

    class TestPedestal(pydantic_model.BasePedestal):
      """Test Pydantic config."""

      model_name: Annotated[Literal['test_pedestal'], torax_pydantic.JAX_STATIC] = (
          'test_pedestal'
      )
      test_value: float = 42.0

      def build_pedestal_model(self) -> TestPedestalModel:
        return TestPedestalModel()

      def build_runtime_params(
          self, t: chex.Numeric
      ) -> TestRuntimeParams:
        return TestRuntimeParams(
            set_pedestal=self.set_pedestal.get_value(t),
            test_value=self.test_value,
        )

    # Register the model
    register_model.register_pedestal_model(TestPedestal)

    # Verify it can be instantiated through the ModelConfig API
    # Create a minimal ToraxConfig using from_dict to test the registration
    minimal_config_dict = {
        'profile_conditions': {},
        'numerics': {},
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular'},
        'sources': {},
        'pedestal': {
            'model_name': 'test_pedestal',
            'test_value': 99.0,
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(minimal_config_dict)

    # Verify the pedestal config is the correct type
    self.assertIsInstance(torax_config.pedestal, TestPedestal)
    self.assertEqual(torax_config.pedestal.test_value, 99.0)
    self.assertEqual(torax_config.pedestal.model_name, 'test_pedestal')

    # Verify it can build the model and runtime params
    model = torax_config.pedestal.build_pedestal_model()
    self.assertIsInstance(model, TestPedestalModel)

    runtime_params = torax_config.pedestal.build_runtime_params(t=0.0)
    self.assertEqual(runtime_params.test_value, 99.0)


  def test_multiple_registrations(self):
    """Test that multiple custom models can be registered."""

    @chex.dataclass(frozen=True)
    class Model1(pedestal_model.PedestalModel):
      def _call_implementation(
          self,
          runtime_params,
          geo,
          core_profiles,
      ) -> pedestal_model.PedestalModelOutput:
        return pedestal_model.PedestalModelOutput(
            rho_norm_ped_top=0.9,
            rho_norm_ped_top_idx=0,
            T_i_ped=1.0,
            T_e_ped=1.0,
            n_e_ped=1.0e20,
        )

    class Config1(pydantic_model.BasePedestal):
      model_name: Annotated[Literal['model1'], torax_pydantic.JAX_STATIC] = (
          'model1'
      )

      def build_pedestal_model(self):
        return Model1()

      def build_runtime_params(self, t):
        return pedestal_runtime_params.RuntimeParams(
            set_pedestal=self.set_pedestal.get_value(t),
        )

    @chex.dataclass(frozen=True)
    class Model2(pedestal_model.PedestalModel):
      def _call_implementation(
          self,
          runtime_params,
          geo,
          core_profiles,
      ) -> pedestal_model.PedestalModelOutput:
        return pedestal_model.PedestalModelOutput(
            rho_norm_ped_top=0.9,
            rho_norm_ped_top_idx=0,
            T_i_ped=2.0,
            T_e_ped=2.0,
            n_e_ped=2.0e20,
        )

    class Config2(pydantic_model.BasePedestal):
      model_name: Annotated[Literal['model2'], torax_pydantic.JAX_STATIC] = (
          'model2'
      )

      def build_pedestal_model(self):
        return Model2()

      def build_runtime_params(self, t):
        return pedestal_runtime_params.RuntimeParams(
            set_pedestal=self.set_pedestal.get_value(t),
        )

    # Register both models
    register_model.register_pedestal_model(Config1)
    register_model.register_pedestal_model(Config2)

    # Verify both can be instantiated through the ModelConfig API
    minimal_config_dict_1 = {
        'profile_conditions': {},
        'numerics': {},
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular'},
        'sources': {},
        'pedestal': {
            'model_name': 'model1',
        },
    }
    torax_config_1 = model_config.ToraxConfig.from_dict(minimal_config_dict_1)
    self.assertIsInstance(torax_config_1.pedestal, Config1)
    self.assertEqual(torax_config_1.pedestal.model_name, 'model1')

    minimal_config_dict_2 = {
        'profile_conditions': {},
        'numerics': {},
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular'},
        'sources': {},
        'pedestal': {
            'model_name': 'model2',
        },
    }
    torax_config_2 = model_config.ToraxConfig.from_dict(minimal_config_dict_2)
    self.assertIsInstance(torax_config_2.pedestal, Config2)
    self.assertEqual(torax_config_2.pedestal.model_name, 'model2')


if __name__ == '__main__':
  absltest.main()
