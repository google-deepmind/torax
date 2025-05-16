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
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.torax_pydantic import model_config
from torax.tests.test_lib import default_sources


class CoreProfileSettersTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=4, theta_implicit=0, set_pedestal=False),
      dict(num_cells=4, theta_implicit=0, set_pedestal=True),
      dict(num_cells=4, theta_implicit=0.5, set_pedestal=False),
      dict(num_cells=4, theta_implicit=0.5, set_pedestal=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_implicit, set_pedestal
  ):
    sources_config = default_sources.get_default_source_config()
    sources_config['ei_exchange']['Qei_multiplier'] = 0.0
    sources_config['generic_heat']['P_total'] = 0.0
    sources_config['fusion']['mode'] = 'ZERO'
    sources_config['ohmic']['mode'] = 'ZERO'
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(
                set_pedestal=set_pedestal, model_name='set_T_ped_n_ped'
            ),
            sources=sources_config,
            solver=dict(
                use_predictor_corrector=False, theta_implicit=theta_implicit
            ),
            transport=dict(transport_model='constant', chi_min=0, chi_i=1),
            time_step_calculator=dict(),
        )
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    evolving_names = tuple(['T_i'])
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
