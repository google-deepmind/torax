# Copyright 2026 DeepMind Technologies Limited
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

"""Test utilities for constructing transport objects."""

from torax._src.geometry import geometry
from torax._src.transport_model import transport_coeffs


def make_zero_core_transport(
    geo: geometry.Geometry,
) -> transport_coeffs.CoreTransport:
  """Returns a CoreTransport initialized with zeros on the given geometry.

  Args:
    geo: Geometry grid on which zero transport coefficients are created.

  Returns:
    A CoreTransport object with zeros for turbulent and neoclassical transport.
  """
  return transport_coeffs.CoreTransport(
      total=transport_coeffs.TransportCoeffs.zeros(geo),
      turbulent=transport_coeffs.TurbulentTransport.zeros(geo),
      neoclassical=transport_coeffs.NeoclassicalTransport.zeros(geo),
      pereverzev=None,
  )
