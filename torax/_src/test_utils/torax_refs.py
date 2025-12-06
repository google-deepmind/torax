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

"""Reference values used throughout TORAX unit tests."""
import dataclasses
import json
from typing import Any, Callable, Final, Literal, Mapping

import immutabledict
import jax
import numpy as np
# set_jax_precision() is called in torax __init__.py, needed for tests and
# reference generation
import torax  # pylint: disable=unused-import
from torax._src import path_utils
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

JSON_FILENAME: Literal['references.json'] = 'references.json'


def _load_all_references():
  """Loads all reference data from the JSON file."""
  json_path = path_utils.torax_path() / '_src' / 'test_utils' / JSON_FILENAME
  try:
    with open(json_path, 'r') as f:
      return json.load(f)
  except FileNotFoundError:
    raise FileNotFoundError(
        f'Reference data file not found at {json_path}.'
    ) from None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class References:
  """Collection of reference values useful for unit tests."""

  config: model_config.ToraxConfig
  psi: cell_variable.CellVariable
  psi_face_grad: np.ndarray
  psidot: np.ndarray
  j_total: np.ndarray
  q: np.ndarray
  s: np.ndarray

  @property
  def geometry_provider(self) -> geometry_provider_lib.GeometryProvider:
    return self.config.geometry.build_provider

  def get_runtime_params_and_geo(
      self,
  ) -> tuple[runtime_params_lib.RuntimeParams, geometry.Geometry]:
    t = self.config.numerics.t_initial
    params = build_runtime_params.RuntimeParamsProvider.from_config(
        self.config
    )
    return build_runtime_params.get_consistent_runtime_params_and_geometry(
        t=t,
        runtime_params_provider=params,
        geometry_provider=self.config.geometry.build_provider,
    )


def _build_references_from_case_data(
    case_name: str, torax_config: model_config.ToraxConfig
) -> References:
  """Helper function to build a References object from a config and JSON data."""
  geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
  all_references = _load_all_references()
  case_data = all_references[case_name]

  psi_value = np.array(case_data['psi'])
  psi_face_grad = np.array(case_data['psi_face_grad'])
  psi_grad_bc = psi_face_grad[-1]

  psi = cell_variable.CellVariable(
      value=psi_value,
      right_face_grad_constraint=psi_grad_bc,
      dr=geo.drho_norm,
  )

  return References(
      config=torax_config,
      psi=psi,
      psi_face_grad=psi_face_grad,
      psidot=np.array(case_data['psidot']),
      j_total=np.array(case_data['j_total']),
      q=np.array(case_data['q']),
      s=np.array(case_data['s']),
  )


def circular_references() -> References:
  """Reference values for circular geometry."""
  torax_config = model_config.ToraxConfig.from_dict({
      'profile_conditions': {
          'Ip': 15e6,
          'current_profile_nu': 3,
          'n_e_nbar_is_fGW': True,
          'normalize_n_e_to_nbar': True,
          'nbar': 0.85,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
      },
      'numerics': {},
      'plasma_composition': {},
      'geometry': {
          'geometry_type': 'circular',
      },
      'transport': {},
      'solver': {},
      'pedestal': {},
      'sources': {'generic_current': {}},
  })
  return _build_references_from_case_data('circular_references', torax_config)


def chease_references_Ip_from_chease() -> References:
  """Reference values for CHEASE geometry where Ip comes from the file."""
  torax_config = model_config.ToraxConfig.from_dict({
      'profile_conditions': {
          'Ip': 15e6,
          'current_profile_nu': 3,
          'n_e_nbar_is_fGW': True,
          'normalize_n_e_to_nbar': True,
          'nbar': 0.85,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
      },
      'numerics': {},
      'plasma_composition': {},
      'geometry': {
          'geometry_type': 'chease',
          'Ip_from_parameters': False,
      },
      'transport': {},
      'solver': {},
      'pedestal': {},
      'sources': {'generic_current': {}},
  })
  return _build_references_from_case_data(
      'chease_references_Ip_from_chease', torax_config
  )


def chease_references_Ip_from_runtime_params() -> References:
  """Reference values for CHEASE geometry where Ip comes from runtime params."""
  torax_config = model_config.ToraxConfig.from_dict({
      'profile_conditions': {
          'Ip': 15e6,
          'current_profile_nu': 3,
          'n_e_nbar_is_fGW': True,
          'normalize_n_e_to_nbar': True,
          'nbar': 0.85,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
      },
      'numerics': {},
      'plasma_composition': {},
      'geometry': {
          'geometry_type': 'chease',
          'Ip_from_parameters': True,
      },
      'transport': {},
      'solver': {},
      'pedestal': {},
      'sources': {'generic_current': {}},
  })
  return _build_references_from_case_data(
      'chease_references_Ip_from_runtime_params', torax_config
  )


def sawtooth_references() -> dict[str, Any]:
  """Reference values for sawtooth post-crash state.

  Returns a dict with 'T_e', 'n_e', and 'psi' arrays representing the
  post-crash state after a sawtooth crash is triggered.
  """
  # This function will be implemented in regenerate_torax_refs.py
  # to calculate the actual values. Here we just load from JSON.
  all_references = _load_all_references()
  case_data = all_references.get('sawtooth_references', {})
  return {
      'T_e': np.array(case_data.get('T_e', [])),
      'n_e': np.array(case_data.get('n_e', [])),
      'psi': np.array(case_data.get('psi', [])),
  }


REFERENCES_REGISTRY: Final[Mapping[str, Callable[[], References]]] = (
    immutabledict.immutabledict({
        'circular_references': circular_references,
        'chease_references_Ip_from_chease': chease_references_Ip_from_chease,
        'chease_references_Ip_from_runtime_params': (
            chease_references_Ip_from_runtime_params
        ),
    })
)
