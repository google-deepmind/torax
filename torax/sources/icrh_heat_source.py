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
"""Surrogate model for ion-cyclotron resonance heating model."""
from typing import Sequence

import flax.linen as nn
from jax import numpy as jnp


class ToricNN(nn.Module):
  """Surrogate heating model trained on TORIC ICRF solver simulation."""

  hidden_sizes: Sequence[int]
  pca_coeffs: int
  input_dim: int
  radial_nodes: int

  def setup(self):
    self.scalar_mean = self.param(
        'scalar_mean',
        lambda rng, shape: jnp.zeros(self.input_dim, dtype=jnp.float64),
        (self.input_dim,),
    )
    self.scalar_scale = self.param(
        'scalar_scale',
        lambda rng, shape: jnp.zeros(self.input_dim, dtype=jnp.float64),
        (self.input_dim,),
    )
    self.pca_components = self.param(
        'pca_components',
        lambda rng, shape: jnp.zeros(
            (self.pca_coeffs, self.radial_nodes), dtype=jnp.float64
        ),
        (
            self.pca_coeffs,
            self.radial_nodes,
        ),
    )
    self.pca_mean = self.param(
        'pca_mean',
        lambda rng, shape: jnp.zeros(self.radial_nodes, dtype=jnp.float64),
        (self.radial_nodes,),
    )

  @nn.compact
  def __call__(
      self,
      x,
  ):
    x = (x - self.mean) / self.scale  # Scale and normalise inputs.

    for hidden_size in self.hidden_sizes:
      x = nn.Dense(hidden_size, dtype=jnp.float64)(x)
      x = nn.relu(x)

    x = nn.Dense(self.pca_coeffs, dtype=jnp.float64)(x)

    x = x @ self.pca_components + self.pca_mean  # Project back to true values.
    x = x * (x > 0)  # Eliminate non-physical values for power deposition.
    return x
