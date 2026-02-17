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

"""Turbulent transport dataclass.

Note: this needs to be separate from transport_model to avoid circular imports
since a) pedestal_model is imported in transport_model and b) pedestal_model
needs to create TurbulentTransport instances.
"""

import dataclasses
import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
  """Turbulent transport coefficients calculated by a transport model.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
    chi_face_el_bohm: (Optional) Bohm contribution for electron heat
      conductivity.
    chi_face_el_gyrobohm: (Optional) GyroBohm contribution for electron heat
      conductivity.
    chi_face_ion_bohm: (Optional) Bohm contribution for ion heat conductivity.
    chi_face_ion_gyrobohm: (Optional) GyroBohm contribution for ion heat
      conductivity.
    chi_face_ion_itg: (Optional) ITG contribution for ion heat conductivity.
    chi_face_ion_tem: (Optional) TEM contribution for ion heat conductivity.
    chi_face_el_itg: (Optional) ITG contribution for electron heat conductivity.
    chi_face_el_tem: (Optional) TEM contribution for electron heat conductivity.
    chi_face_el_etg: (Optional) ETG contribution for electron heat conductivity.
    d_face_el_itg: (Optional) ITG contribution for electron diffusivity.
    d_face_el_tem: (Optional) TEM contribution for electron diffusivity.
    v_face_el_itg: (Optional) ITG contribution for electron convection.
    v_face_el_tem: (Optional) TEM contribution for electron convection.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
  chi_face_ion_itg: jax.Array | None = None
  chi_face_ion_tem: jax.Array | None = None
  chi_face_el_itg: jax.Array | None = None
  chi_face_el_tem: jax.Array | None = None
  chi_face_el_etg: jax.Array | None = None
  d_face_el_itg: jax.Array | None = None
  d_face_el_tem: jax.Array | None = None
  v_face_el_itg: jax.Array | None = None
  v_face_el_tem: jax.Array | None = None
