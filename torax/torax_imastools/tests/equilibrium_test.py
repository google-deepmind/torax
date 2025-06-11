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

"""Unit tests for torax.torax_imastools.equilibrium.py"""

import importlib
import os
from typing import Any, Optional

import numpy as np
import pytest
from absl.testing import absltest, parameterized

from torax.geometry import pydantic_model as geometry_pydantic_model

try:
    import imas
    from imas.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any

from torax._src.output_tools import post_processing
from torax._src.config import build_runtime_params
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config
from torax.torax_imastools.equilibrium as imas_equilibrium
from torax.torax_imastools.util as imas_util


@pytest.mark.skipif(
    importlib.util.find_spec("imas_core") is None,
    reason="IMAS-Python optional dependency"
)
class EquilibriumTest(sim_test_case.SimTestCase):
    """Unit tests for the `toraximastools.equilibrium` module."""

    @parameterized.parameters(
        [
            dict(config_name="test_imas.py", rtol=0.02, atol=1e-8),
        ]
    )
    def test_save_geometry_to_IMAS(
        self,
        config_name,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        """Test that the default IMAS geometry can be built and converted back
        to IDS."""
        if importlib.util.find_spec("imas") is None:
            self.skipTest("IMAS-Python optional dependency")

        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        # Input equilibrium reading
        config_module = self._get_config_module(config_name)
        geometry_dir = "torax/data/third_party/geo"
        path = os.path.join(
            geometry_dir, config_module.CONFIG["geometry"]["equilibrium_object"]
        )
        equilibrium_in = imas_util.load_IMAS_data(path, "equilibrium")
        # Build TORAXSimState object and write output to equilibrium IDS.
        # Improve resolution to compare the input without losing too much
        # information
        config_module.CONFIG["geometry"]["n_rho"] = len(
            equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
        )
        torax_config = model_config.ToraxConfig.from_dict(config_module.CONFIG)

        (
            _,
            dynamic_runtime_params_slice_provider,
            geometry_provider,
            initial_state,
            _,
            _,
            _,
        ) = run_simulation.prepare_simulation(torax_config)

        dynamic_runtime_params_slice_for_init, _ = (
            build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
                t=torax_config.numerics.t_initial,
                dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
                geometry_provider=geometry_provider,
            )
        )
        sim_state = post_processing.make_outputs(
            sim_state=initial_state,
            dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
        )

        equilibrium_out = imas_equilibrium.geometry_to_IMAS(sim_state)

        # Compare the output IDS with the input one.
        rhon_out = equilibrium_out.time_slice[0].profiles_1d.rho_tor_norm
        rhon_in = equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
        np.testing.assert_allclose(
            np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.psi),
            equilibrium_in.time_slice[0].profiles_1d.psi,
            rtol=rtol,
            atol=atol,
            err_msg="psi profile failed",
        )
        np.testing.assert_allclose(
            np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.q),
            equilibrium_in.time_slice[0].profiles_1d.q,
            rtol=rtol,
            atol=atol,
            err_msg="q profile failed",
        )
        np.testing.assert_allclose(
            np.interp(
                rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.j_phi
            ),
            equilibrium_in.time_slice[0].profiles_1d.j_phi,
            rtol=rtol,
            atol=atol,
            err_msg="jtot profile failed",
        )
        np.testing.assert_allclose(
            np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.f),
            equilibrium_in.time_slice[0].profiles_1d.f,
            rtol=rtol,
            atol=atol,
            err_msg="f profile failed",
        )

    @parameterized.parameters(
        [
            dict(rtol=0.02, atol=1e-8),
        ]
    )
    def test_geometry_from_IMAS(
        self,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        """Test to compare the outputs of CHEASE and IMAS methods for the same
        equilibrium."""
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol

        # Loading the equilibrium and constructing geometry object
        config = geometry_pydantic_model.IMASConfig(
            equilibrium_object="ITERhybrid_COCOS17_IDS_ddv4.nc",
            Ip_from_parameters=True
        )
        geo_IMAS = config.build_geometry()

        geo_CHEASE = geometry_pydantic_model.CheaseConfig().build_geometry()

        # Comparison of the fields
        diverging_fields = []
        for key in geo_IMAS:
            if (
                key != "geometry_type"
                and key != "Ip_from_parameters"
                and key != "torax_mesh"
                and key != "_z_magnetic_axis"
            ):
                try:
                    np.testing.assert_allclose(
                        geo_IMAS[key],
                        geo_CHEASE[key],
                        rtol=rtol,
                        atol=atol,
                        verbose=True,
                        err_msg=f"Value {key} failed",
                    )
                except AssertionError:
                    diverging_fields.append(key)
        if diverging_fields:
            raise AssertionError(f"Diverging profiles: {diverging_fields}")


if __name__ == "__main__":
    absltest.main()
