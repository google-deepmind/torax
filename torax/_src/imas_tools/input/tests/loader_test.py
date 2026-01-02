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
import pathlib

import imas
from absl.testing import absltest, parameterized
from imas import ids_toplevel

from torax._src.imas_tools.input import loader


class IMASLoaderTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            dict(
                ids_name="core_profiles",
                path="core_profiles_ddv4_iterhybrid_rampup_conditions.nc",
            ),
            dict(
                ids_name="equilibrium",
                path="ITERhybrid_COCOS17_IDS_ddv4.nc",
            ),
        ]
    )
    def test_load_imas_from_net_cdf(
        self,
        ids_name,
        path,
    ):
        ids_in = loader.load_imas_data(path, ids_name)
        assert isinstance(ids_in, ids_toplevel.IDSToplevel)

    def test_load_older_dd_version_data_explicit_convert(self):
        directory = pathlib.Path(__file__).parent
        ids = loader.load_imas_data(
            "core_profiles_ddv3.nc",
            "core_profiles",
            directory=directory,
            explicit_convert=True,
        )
        assert isinstance(ids, ids_toplevel.IDSToplevel)
        self.assertEqual(
            imas.util.get_data_dictionary_version(ids),
            loader._TORAX_IMAS_DD_VERSION,
        )

    def test_load_older_dd_version_without_explicit_convert_raises(self):
        directory = pathlib.Path(__file__).parent
        with self.assertRaises(RuntimeError):
            loader.load_imas_data(
                "core_profiles_ddv3.nc",
                "core_profiles",
                directory=directory,
                explicit_convert=False,
            )


if __name__ == "__main__":
    absltest.main()
