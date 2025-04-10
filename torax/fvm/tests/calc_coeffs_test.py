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
from torax.core_profiles import initialization
from torax.fvm import calc_coeffs
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.tests.test_lib import default_sources
from torax.torax_pydantic import model_config


class CoreProfileSettersTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=4, theta_imp=0, set_pedestal=False),
      dict(num_cells=4, theta_imp=0, set_pedestal=True),
      dict(num_cells=4, theta_imp=0.5, set_pedestal=False),
      dict(num_cells=4, theta_imp=0.5, set_pedestal=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_imp, set_pedestal
  ):
    sources_config = default_sources.get_default_source_config()
    sources_config['qei_source']['Qei_mult'] = 0.0
    sources_config['generic_ion_el_heat_source']['Ptot'] = 0.0
    sources_config['fusion_heat_source']['mode'] = (
        source_runtime_params.Mode.ZERO
    )
    sources_config['ohmic_heat_source']['mode'] = (
        source_runtime_params.Mode.ZERO
    )
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            runtime_params=dict(
                numerics=dict(el_heat_eq=False),
            ),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(
                set_pedestal=set_pedestal,
                pedestal_model='set_tped_nped'),
            sources=sources_config,
            stepper=dict(predictor_corrector=False, theta_imp=theta_imp),
            transport=dict(transport_model='constant', chimin=0, chii_const=1),
            time_step_calculator=dict(),
        )
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            torax_config.runtime_params,
            transport=torax_config.transport,
            sources=torax_config.sources,
            stepper=torax_config.stepper,
            pedestal=torax_config.pedestal,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            profile_conditions=torax_config.profile_conditions,
            numerics=torax_config.numerics,
            plasma_composition=torax_config.plasma_composition,
            sources=torax_config.sources,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
            stepper=torax_config.stepper,
        )
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
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
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    transport_model = torax_config.transport.build_transport_model()
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
