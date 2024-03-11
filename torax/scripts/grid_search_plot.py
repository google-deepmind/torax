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

"""Plots the output of `grid_search.py.

Invalid points (NaNs encountered during calculation, etc) are plotted as red
triangles.
Infeasible points (q < 1 for normalized radius > 0.1 at any time) are plotted
as yellow triangles.
Feasible points are plotted as cyan circles, with the brightness of the cyan
indicating fusion power.

Args:
  path: The filepath to save the output of the grid search.
"""

from typing import Sequence

from absl import app
from matplotlib import pyplot as plt
import numpy as np

# pylint:disable=invalid-name
# Names like "Ip" are chosen for consistency with standard physics notation


def main(argv: Sequence[str]) -> None:
  _, path = argv

  # Load file content
  with open(path) as f:
    lines = f.readlines()

  # Lists of values at valid points
  fext = []
  rext = []
  Ip = []
  fp = []
  qr_min = []

  # List of values at invalid points
  bad_fext = []
  bad_rext = []
  bad_Ip = []

  # Read values from the file content
  for line in lines[1:]:
    fexstr, rexstr, Ipstr, fpstr, qr_minstr = line.split('\t')
    if fpstr != 'None':
      fext.append(float(fexstr))
      rext.append(float(rexstr))
      Ip.append(float(Ipstr))
      fp.append(float(fpstr))
      qr_min.append(float(qr_minstr))
    else:
      bad_fext.append(float(fexstr))
      bad_rext.append(float(rexstr))
      bad_Ip.append(float(Ipstr))
  # Convert lists to numpy
  fext = np.array(fext)
  rext = np.array(rext)
  Ip = np.array(Ip)
  fp = np.array(fp)
  qr_min = np.array(qr_min)

  # Draw the 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.set_xlabel('fext')
  ax.set_ylabel('rext')
  ax.set_zlabel('Ip')

  # Plot invalid points, if any
  if bad_fext:
    ax.scatter(bad_fext, bad_rext, bad_Ip, marker='^', color='r')

  # Plot infeasible points, if any
  infeasible = qr_min < 1.0
  if infeasible.sum() > 0:
    ifext = fext[infeasible]
    irext = rext[infeasible]
    iIp = Ip[infeasible]
    ax.scatter(ifext, irext, iIp, marker='^', color='y')

  feasible = qr_min >= 1.0

  # Plot feasible points, if any
  if feasible.sum() > 0:
    # Color code the feasible points by fusion power.
    # Use the whole dynamic range of the color scheme.
    fp_remap = fp[feasible]
    fp_remap = fp_remap - fp_remap.min()
    assert fp_remap.min() == 0.0
    fp_remap = fp_remap / fp_remap.max()
    assert fp_remap.max() == 1.0
    fp_remap = np.expand_dims(fp_remap, 1)
    fp_remap = np.tile(fp_remap, (1, 3))
    fp_remap[:, 0] = 0.0
    fp_remap = list(fp_remap)
    fp_remap = [tuple(e) for e in fp_remap]

    ax.scatter(
        fext[feasible],
        rext[feasible],
        Ip[feasible],
        marker='o',
        color=fp_remap,
    )

  plt.show()


if __name__ == '__main__':
  app.run(main)
