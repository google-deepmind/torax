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
"""Tests for calc_coeffs."""

from absl.testing import absltest
from absl.testing import parameterized
from torax import core_profile_setters
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.fvm import calc_coeffs
from torax.geometry import circular_geometry
from torax.pedestal_model import set_tped_nped
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_profile_builders
from torax.stepper import runtime_params as stepper_params_lib
from torax.tests.test_lib import default_sources
from torax.transport_model import constant as constant_transport_model


class CoreProfileSettersTest(parameterized.TestCase):
  """Unit tests for calc_coeffs."""

  @parameterized.parameters([
      dict(num_cells=2, theta_imp=0, set_pedestal=False),
      dict(num_cells=2, theta_imp=0, set_pedestal=True),
      dict(num_cells=3, theta_imp=0.5, set_pedestal=False),
      dict(num_cells=3, theta_imp=0.5, set_pedestal=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_imp, set_pedestal
  ):
    """Smoke test for calc_coeffs both with and without pedestal."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=set_pedestal,
        ),
        numerics=numerics_lib.Numerics(
            el_heat_eq=False,
        ),
    )
    stepper_params = stepper_params_lib.RuntimeParams(
        predictor_corrector=False,
        theta_imp=theta_imp,
    )
    geo = circular_geometry.build_circular_geometry(n_rho=num_cells)
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                chimin=0,
                chii_const=1,
            ),
        )
    )
    pedestal_model_builder = (
        set_tped_nped.SetTemperatureDensityPedestalModelBuilder()
    )
    pedestal_model = pedestal_model_builder()
    transport_model = transport_model_builder()
    source_models_builder = default_sources.get_default_sources_builder()
    source_models_builder.runtime_params['qei_source'].Qei_mult = 0.0
    source_models_builder.runtime_params['generic_ion_el_heat_source'].Ptot = (
        0.0
    )
    source_models_builder.runtime_params['fusion_heat_source'].mode = (
        source_runtime_params.Mode.ZERO
    )
    source_models_builder.runtime_params['ohmic_heat_source'].mode = (
        source_runtime_params.Mode.ZERO
    )
    source_models = source_models_builder()
    dynamic_runtime_params_slice = (
        runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=source_models_builder.runtime_params,
            stepper=stepper_params,
            pedestal=pedestal_model_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        runtime_params_slice_lib.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    evolving_names = tuple(['temp_ion'])
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        static_runtime_params_slice=static_runtime_params_slice,
        source_models=source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        explicit=True,
    )
    calc_coeffs.calc_coeffs(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        transport_model=transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=source_models,
        pedestal_model=pedestal_model,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )


if __name__ == '__main__':
  absltest.main()
