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

"""Basic post-run plotting tool. Plot a single run or comparison of two runs.

Includes a time slider. Reads output files with xarray data or legacy h5 data.

Plots:
(1) chi_i, chi_e (transport coefficients)
(2) Ti, Te (temperatures)
(3) ne (density)
(4) jtot, johm (total and ohmic plasma current)
(5) q (safety factor)
(6) s (magnetic shear)
"""
from absl import app
from absl import flags
import matplotlib
from torax.plotting import plotruns_lib


matplotlib.use('TkAgg')


_OUTFILES = flags.DEFINE_spaceseplist(
    'outfile',
    None,
    'Relative location of output files (if two are provided, a comparison is'
    ' done)',
    required=True,
)


def main(_):
  outfiles = _OUTFILES.value
  if len(outfiles) == 1:
    plotruns_lib.plot_run(outfiles[0])
  else:
    plotruns_lib.plot_run(outfiles[0], outfiles[1])


if __name__ == '__main__':
  app.run(main)
