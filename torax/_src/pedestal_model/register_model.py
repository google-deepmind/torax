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
"""Register a pedestal model with TORAX."""

from typing import Union, get_args

from torax._src.pedestal_model import pydantic_model
from torax._src.torax_pydantic import model_config


def register_pedestal_model(
    pydantic_model_class: type[pydantic_model.BasePedestal],
):
    """Registers a pedestal model with TORAX.

    This function adds the pedestal model to the config model such that it can
    be configured via pydantic. The pydantic model class should inherit from
    BasePedestal and should have a distinct model_name. It should also define a
    build_pedestal_model method which returns a PedestalModel.

    Example:
      ```python
      from torax._src.pedestal_model import pydantic_model
      from torax._src.pedestal_model import register_model
      from torax._src.pedestal_model import pedestal_model as pm
      from torax._src.pedestal_model import runtime_params
      from typing import Annotated, Literal
      from torax._src.torax_pydantic import torax_pydantic
      import chex

      # Define your custom JAX pedestal model
      @chex.dataclass(frozen=True)
      class MyPedestalModel(pm.PedestalModel):
        def _call_implementation(
            self,
            runtime_params: runtime_params.RuntimeParams,
            geo: geometry.Geometry,
            core_profiles: state.CoreProfiles,
        ) -> pm.PedestalModelOutput:
          # Your custom pedestal calculation logic here
          T_e_ped = 5.0  # keV
          T_i_ped = 6.0  # keV
          n_e_ped = 0.7e20  # m^-3
          rho_norm_ped_top = 0.91

          return pm.PedestalModelOutput(
              rho_norm_ped_top=rho_norm_ped_top,
              rho_norm_ped_top_idx=...,  # compute from geo
              T_i_ped=T_i_ped,
              T_e_ped=T_e_ped,
              n_e_ped=n_e_ped,
          )

      # Define your Pydantic config class
      class MyPedestalConfig(pydantic_model.BasePedestal):
        model_name: Annotated[Literal['my_pedestal'], torax_pydantic.JAX_STATIC] = 'my_pedestal'

        # Add any configuration parameters you need
        scaling_factor: float = 1.0

        def build_pedestal_model(self) -> MyPedestalModel:
          return MyPedestalModel()

        def build_runtime_params(self, t: chex.Numeric) -> runtime_params.RuntimeParams:
          return runtime_params.RuntimeParams(
              set_pedestal=self.set_pedestal.get_value(t),
          )

      # Register your model
      register_model.register_pedestal_model(MyPedestalConfig)

      # Now you can use it in your config
      CONFIG = {
          'pedestal': {
              'model_name': 'my_pedestal',
              'set_pedestal': True,
              'scaling_factor': 1.5,
          },
      }
      ```

    Args:
      pydantic_model_class: The pydantic model class to register.
    """
    # Get the current PedestalConfig union types
    current_types = get_args(
        model_config.ToraxConfig.model_fields["pedestal"].annotation
    )

    # Check if already registered to avoid duplicate registration.
    # We check by model_name since the same class may be re-imported with a
    # different identity when loaded via importlib.
    new_model_name = pydantic_model_class.model_fields["model_name"].default
    for existing_type in current_types:
        if hasattr(existing_type, "model_fields"):
            existing_model_name = existing_type.model_fields.get("model_name", {})
            if hasattr(existing_model_name, "default"):
                if existing_model_name.default == new_model_name:
                    return

    # Create a new union with the additional pedestal model
    type_tuple = (*current_types, pydantic_model_class)
    model_config.ToraxConfig.model_fields["pedestal"].annotation = Union[*type_tuple]

    # Rebuild the model to incorporate the new type
    model_config.ToraxConfig.model_rebuild(force=True)
