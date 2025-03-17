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
"""A wrapper for QLKNN transport surrogate models."""
from collections.abc import Mapping
from typing import Final
import immutabledict
import jax
import jax.numpy as jnp
from torax.transport_model import base_qlknn_model
from torax.transport_model import qualikiz_based_transport_model
# pylint: disable=g-import-not-at-top
try:
  from fusion_surrogates import qlknn_model
  _FUSION_SURROGATES_AVAILABLE = True
except ImportError:
  _FUSION_SURROGATES_AVAILABLE = False
# pylint: enable=g-import-not-at-top

# Convert flux names from Qualikiz to TORAX.
_FLUX_NAME_MAP: Final[Mapping[str, str]] = immutabledict.immutabledict({
    'efiITG': 'qi_itg',
    'efeITG': 'qe_itg',
    'pfeITG': 'pfe_itg',
    'efeTEM': 'qe_tem',
    'efiTEM': 'qi_tem',
    'pfeTEM': 'pfe_tem',
    'efeETG': 'qe_etg',
})


class QLKNNModelWrapper(base_qlknn_model.BaseQLKNNModel):
  """A TORAX wrapper for a QLKNNv2 Model."""

  def __init__(
      self,
      path: str,
      flux_name_map: Mapping[str, str] | None = None,
  ):
    if not _FUSION_SURROGATES_AVAILABLE:
      raise ImportError(
          'QLKNNModelWrapper requires fusion_surrogates to be installed.'
      )
    if flux_name_map is None:
      flux_name_map = _FLUX_NAME_MAP
    self._flux_name_map = flux_name_map
    self._model = qlknn_model.QLKNNModel.import_model(path)
    super().__init__(version=self._model.version)

  @property
  def inputs_and_ranges(self) -> base_qlknn_model.InputsAndRanges:
    return self._model.inputs_and_ranges

  def get_model_inputs_from_qualikiz_inputs(
      self, qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs
  ) -> jax.Array:
    """Converts QualikizInputs to model inputs."""
    # Non-trivial mappings from QualikizInputs to model inputs.
    input_map = {
        'Ani': lambda x: x.Ani0,
        'LogNuStar': lambda x: x.log_nu_star_face,
    }

    def _get_input(key: str) -> jax.Array:
      # If no complex mapping is defined, we use the trivial mapping.
      return jnp.array(
          input_map.get(key, lambda x: getattr(x, key))(qualikiz_inputs)
      )

    return jnp.array(
        [_get_input(key) for key in self.inputs_and_ranges.keys()]
    ).T

  def predict(self, inputs: jax.Array) -> dict[str, jax.Array]:
    """Predicts the fluxes given the inputs."""
    model_predictions = self._model.predict(inputs)

    return {
        self._flux_name_map.get(flux_name, flux_name): flux_value
        for flux_name, flux_value in model_predictions.items()
    }
