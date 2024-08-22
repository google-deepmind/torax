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

"""Tests for plotting.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torax  # We want this import to make sure jax gets set to float64
from torax import geometry
from torax import geometry_provider as geometry_provider_lib
from torax.config import numerics as numerics_lib
from torax.config import runtime_params as general_runtime_params
from torax.sources import default_sources
from torax.sources import source_models as source_models_lib
from torax.spectators import plotting
from torax.spectators import spectator
from torax.stepper import linear_theta_method
from torax.time_step_calculator import chi_time_step_calculator
from torax.transport_model import constant as constant_transport_model


class PlottingTest(parameterized.TestCase):
  """Tests the plotting library."""

  def test_default_plot_config_has_valid_keys(self):
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    geo_provider = geometry_provider_lib.ConstantGeometryProvider(geo)
    plot_config = plotting.get_default_plot_config(geo)

    observer = spectator.InMemoryJaxArraySpectator()
    _run_sim_with_sources(runtime_params, geo_provider, observer)
    # Make sure all the keys in plot_config are collected by the observer.
    for plot in plot_config:
      for key in plot.keys:
        self.assertIn(key.key, observer.arrays)

  def test_plot_observer_runs_with_sim_with_sources(self):
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        numerics=numerics_lib.Numerics(t_final=0.2),
    )
    geo = geometry.build_circular_geometry()
    geo_provider = geometry_provider_lib.ConstantGeometryProvider(geo)
    observer = plotting.PlotSpectator(
        plots=plotting.get_default_plot_config(geo),
    )
    _run_sim_with_sources(runtime_params, geo_provider, observer)

  def test_plot_observer_runs_with_sim_without_sources(self):
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        numerics=numerics_lib.Numerics(t_final=0.2),
    )
    geo = geometry.build_circular_geometry()
    observer = plotting.PlotSpectator(
        plots=plotting.get_default_plot_config(geo),
    )
    geometry_provider = geometry_provider_lib.ConstantGeometryProvider(geo)
    _run_sim_without_sources(runtime_params, geometry_provider, observer)


def _run_sim_with_sources(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    observer: spectator.Spectator,
):
  torax.build_sim_object(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider,
      stepper_builder=linear_theta_method.LinearThetaMethodBuilder(),
      transport_model_builder=constant_transport_model.ConstantTransportModelBuilder(),
      source_models_builder=default_sources.get_default_sources_builder(),
      time_step_calculator=chi_time_step_calculator.ChiTimeStepCalculator(),
  ).run(
      spectator=observer,
  )


def _run_sim_without_sources(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    observer: spectator.Spectator,
):
  torax.build_sim_object(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider,
      stepper_builder=linear_theta_method.LinearThetaMethodBuilder(),
      transport_model_builder=constant_transport_model.ConstantTransportModelBuilder(),
      source_models_builder=source_models_lib.SourceModelsBuilder(),
      time_step_calculator=chi_time_step_calculator.ChiTimeStepCalculator(),
  ).run(
      spectator=observer,
  )


if __name__ == '__main__':
  absltest.main()
