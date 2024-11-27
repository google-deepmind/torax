"""Tests for radiation_heat_sink."""

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from torax import core_profile_setters
from torax import geometry
from torax import math_utils
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import generic_ion_el_heat_source
from torax.sources import radiation_heat_sink
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


class RadiationHeatSinkTest(test_lib.SourceTestCase):
    """Tests for RadiationHeatSink."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            source_class=radiation_heat_sink.RadiationHeatSink,
            runtime_params_class=radiation_heat_sink.RuntimeParams,
            unsupported_modes=[
                runtime_params_lib.Mode.MODEL_BASED,
                runtime_params_lib.Mode.PRESCRIBED,
            ],
            expected_affected_core_profiles=(source_lib.AffectedCoreProfile.TEMP_EL,),
            links_back=True,
        )

    def test_source_value(self):
        """Tests that the source value is correct."""
        # Source builder for this class
        radiation_sink_builder = self._source_class_builder()
        radiation_sink_builder.runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
        if not source_lib.is_source_builder(radiation_sink_builder):
            raise TypeError(f"{type(self)} has a bad _source_class_builder")

        # Source builder for generic_ion_el_heat_source
        # We don't test this class, as that should be done in its own test
        heat_source_builder_builder = source_lib.make_source_builder(
                    source_type=generic_ion_el_heat_source.GenericIonElectronHeatSource,
                    runtime_params_type=generic_ion_el_heat_source.RuntimeParams,
        )
        heat_source_builder = heat_source_builder_builder()

        # Runtime params
        runtime_params = general_runtime_params.GeneralRuntimeParams()

        # Source models
        source_models_builder = source_models_lib.SourceModelsBuilder(
            {
                radiation_heat_sink.SOURCE_NAME: radiation_sink_builder,
                generic_ion_el_heat_source.SOURCE_NAME: heat_source_builder,
            },
        )
        source_models = source_models_builder()

        # Extract the source we're testing and check that it's been built correctly
        radiation_sink = source_models.sources[radiation_heat_sink.SOURCE_NAME]
        self.assertIsInstance(radiation_sink, source_lib.Source)

        # Geometry, profiles, and dynamic runtime params
        geo = geometry.build_circular_geometry()
        dynamic_runtime_params_slice = (
            runtime_params_slice.DynamicRuntimeParamsSliceProvider(
                runtime_params=runtime_params,
                sources=source_models_builder.runtime_params,
                torax_mesh=geo.torax_mesh,
            )(
                t=runtime_params.numerics.t_initial,
            )
        )
        core_profiles = core_profile_setters.initial_core_profiles(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            geo=geo,
            source_models=source_models,
        )
        radiation_sink_dynamic_runtime_params_slice = dynamic_runtime_params_slice.sources[
            radiation_heat_sink.SOURCE_NAME
        ]
        heat_source_dynamic_runtime_params_slice = dynamic_runtime_params_slice.sources[
            generic_ion_el_heat_source.SOURCE_NAME
        ]
        radiation_heat_sink_power_density = radiation_sink.get_value(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            dynamic_source_runtime_params=radiation_sink_dynamic_runtime_params_slice,
            geo=geo,
            core_profiles=core_profiles,
        )

        # RadiationHeatSink provides TEMP_EL only
        chex.assert_rank(radiation_heat_sink_power_density, 1)

        # The value should be equal to fraction * sum of the TEMP_EL sources
        # In this case, that is the generic_ion_el_heat_source
        generic_ion_el_power_el = (
            heat_source_dynamic_runtime_params_slice.Ptot
            * heat_source_dynamic_runtime_params_slice.el_heat_fraction
        )
        radiation_heat_sink_power = math_utils.cell_integration(
            radiation_heat_sink_power_density * geo.vpr, geo
        )
        chex.assert_trees_all_close(
            radiation_heat_sink_power,
            generic_ion_el_power_el
            * -radiation_sink_dynamic_runtime_params_slice.fraction_of_total_power_density,
            rtol=1e-2, # TODO: this rtol seems v. high
        )

    def test_invalid_source_types_raise_errors(self):
        """Tests that using unsupported types raises an error."""
        runtime_params = general_runtime_params.GeneralRuntimeParams()
        geo = geometry.build_circular_geometry()
        source_builder = self._source_class_builder()
        source_models_builder = source_models_lib.SourceModelsBuilder(
            {"foo": source_builder},
        )
        source_models = source_models_builder()
        source = source_models.sources["foo"]
        self.assertIsInstance(source, source_lib.Source)
        dynamic_runtime_params_slice_provider = (
            runtime_params_slice.DynamicRuntimeParamsSliceProvider(
                runtime_params=runtime_params,
                sources=source_models_builder.runtime_params,
                torax_mesh=geo.torax_mesh,
            )
        )
        # This slice is needed to create the core_profiles
        dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
            t=runtime_params.numerics.t_initial,
        )
        core_profiles = core_profile_setters.initial_core_profiles(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            geo=geo,
            source_models=source_models,
        )

        for unsupported_mode in self._unsupported_modes:
            source_builder.runtime_params.mode = unsupported_mode
            # Construct a new slice with the given mode
            dynamic_runtime_params_slice = (
                runtime_params_slice.DynamicRuntimeParamsSliceProvider(
                    runtime_params=runtime_params,
                    sources=source_models_builder.runtime_params,
                    torax_mesh=geo.torax_mesh,
                )(
                    t=runtime_params.numerics.t_initial,
                )
            )
            with self.subTest(unsupported_mode.name):
                with self.assertRaises(RuntimeError):
                    source.get_value(
                        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
                        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
                            "foo"
                        ],
                        geo=geo,
                        core_profiles=core_profiles,
                    )

    def test_extraction_of_relevant_profile_from_output(self):
        """Tests that the relevant profile is extracted from the output."""
        geo = geometry.build_circular_geometry()
        source_builder = self._source_class_builder()
        source_models_builder = source_models_lib.SourceModelsBuilder(
            {'foo': source_builder},
        )
        source_models = source_models_builder()
        source = source_models.sources['foo']
        self.assertIsInstance(source, source_lib.Source)
        cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
        fake_profile = jnp.ones(cell)
        # Check TEMP_EL is modified
        np.testing.assert_allclose(
            source.get_source_profile_for_affected_core_profile(
                fake_profile,
                source_lib.AffectedCoreProfile.TEMP_EL.value,
                geo,
            ),
            jnp.ones(cell),
        )
        # For unrelated states, this should just return all 0s.
        np.testing.assert_allclose(
            source.get_source_profile_for_affected_core_profile(
                fake_profile,
                source_lib.AffectedCoreProfile.NE.value,
                geo,
            ),
            jnp.zeros(cell),
        )


if __name__ == "__main__":
    absltest.main()
