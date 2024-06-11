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
from absl.flags import argparse_flags
import matplotlib
from torax.plotting import plotruns_lib


matplotlib.use('TkAgg')


def parse_flags(_):
  parser = argparse_flags.ArgumentParser(description='Plot finished run')
  parser.add_argument(
      '--outfile',
      nargs='*',
      help=(
          'Relative location of output files (if two are provided, a'
          ' comparison is done)'
      ),
  )
  parser.set_defaults(normalized=True)
  return parser.parse_args()


def main(args):
  if len(args.outfile) == 1:
    plotruns_lib.plot_run(args.outfile[0])
  else:
    plotruns_lib.plot_run(args.outfile[0], args.outfile[1])


if __name__ == '__main__':
  app.run(main, flags_parser=parse_flags)
