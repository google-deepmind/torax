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

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import build_runtime_params
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.fvm import calc_coeffs
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.tests.test_lib import default_sources
from torax.transport_model import constant as constant_transport_model


class CoreProfileSettersTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=3, theta_imp=0, set_pedestal=False),
      dict(num_cells=3, theta_imp=0, set_pedestal=True),
      dict(num_cells=4, theta_imp=0.5, set_pedestal=False),
      dict(num_cells=4, theta_imp=0.5, set_pedestal=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_imp, set_pedestal
  ):
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=set_pedestal,
        ),
        numerics=numerics_lib.Numerics(
            el_heat_eq=False,
        ),
    )
    stepper_params = stepper_pydantic_model.Stepper.from_dict(
        dict(
            predictor_corrector=False,
            theta_imp=theta_imp,
        )
    )
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cells
    ).build_geometry()
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                chimin=0,
                chii_const=1,
            ),
        )
    )
    pedestal = pedestal_pydantic_model.Pedestal()
    pedestal_model = pedestal.build_pedestal_model()
    transport_model = transport_model_builder()
    sources = default_sources.get_default_sources()
    sources_dict = sources.to_dict()
    sources_dict = sources_dict['source_model_config']
    sources_dict['qei_source']['Qei_mult'] = 0.0
    sources_dict['generic_ion_el_heat_source']['Ptot'] = (
        0.0
    )
    sources_dict['fusion_heat_source']['mode'] = (
        source_runtime_params.Mode.ZERO
    )
    sources_dict['ohmic_heat_source']['mode'] = (
        source_runtime_params.Mode.ZERO
    )
    sources = sources_pydantic_model.Sources.from_dict(sources_dict)
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            stepper=stepper_params,
            pedestal=pedestal,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_params,
        )
    )
    core_profiles = initialization.initial_core_profiles(
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
