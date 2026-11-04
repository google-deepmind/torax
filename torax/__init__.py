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

"""Library functionality for TORAX."""

import os

# pylint: disable=g-importing-member
from torax import constants
from torax import experimental
from torax import math_utils
from torax import output
from torax import pedestal
from torax import sources
from torax import torax_pydantic
from torax import transport
from torax._src import jax_utils
from torax._src import version
from torax._src.config.config_loader import build_torax_config_from_file
from torax._src.config.config_loader import import_module
from torax._src.config.runtime_params import RuntimeParams
from torax._src.core_profiles.core_profiles import CoreProfiles
from torax._src.core_profiles.solver_profiles import SolverProfiles
from torax._src.geometry.geometry import Geometry
from torax._src.orchestration.run_simulation import run_simulation
from torax._src.output_tools.output import StateHistory
from torax._src.output_tools.post_processing import PostProcessedOutputs
from torax._src.plotting.plotruns_lib import plot_run
from torax._src.plotting.plotruns_lib import plot_run_from_data_tree
from torax._src.state import CoreTransport
from torax._src.state import SimError
from torax._src.torax_pydantic.config import ToraxConfig

# pylint: enable=g-importing-member


os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') + ' --xla_cpu_opt_preset=FAST_COMPILE'
)

__version__ = version.TORAX_VERSION
__version_info__ = version.TORAX_VERSION_INFO

__all__ = [
    'constants',
    'experimental',
    'math_utils',
    'output',
    'pedestal',
    'sources',
    'torax_pydantic',
    'transport',
    'build_torax_config_from_file',
    'import_module',
    'plot_run',
    'plot_run_from_data_tree',
    'run_simulation',
    'SolverProfiles',
    'CoreProfiles',
    'CoreTransport',
    'Geometry',
    'PostProcessedOutputs',
    'RuntimeParams',
    'SimError',
    'StateHistory',
    'ToraxConfig',
]

jax_utils.set_jax_precision()
