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

"""Spectators are "probes" into the simulation while it runs."""

import abc
from typing import Any
import jax


class Spectator(abc.ABC):
  """Abstract base class for Torax spectators.

  A spectator is a "probe" into the simulator which allows users to observe
  aspects of the simulation while the sim steps and state evolves. What the
  spectator does with that measured information, whether it is log it, plot it,
  or simply store it in memory, depends on the implementation of the spectator.

  This class is abstract, so it is an interface and cannot be instantiated.
  Subclasses can provide a proper implementation and be used.
  """

  def reset(self) -> None:
    pass

  def before_step(self) -> None:
    """Called before a simulation step update.

    Subclasses of Spectator may override this no-op method implementation.
    """
    pass

  def after_step(self) -> None:
    """Called after a simulation step update.

    Subclasses of Spectator may override this no-op method implementation.
    """
    pass

  @abc.abstractmethod
  def observe(self, key: str, data: Any) -> None:
    """Observes some piece of data associated with the key.

    Subclasses of Spectator MUST override this method.

    Args:
      key: Key associated with this piece of data.
      data: Any sort of data can be passed in.
    """


class InMemoryJaxArraySpectator(Spectator):
  """Collects a history of JAX arrays in memory."""

  def __init__(self):
    self._arrays: dict[str, list[jax.Array]] = {}

  @property
  def arrays(self) -> dict[str, list[jax.Array]]:
    """Returns a history of all data observed by this spectator."""
    return self._arrays

  def reset(self) -> None:
    """Deletes the history stored in memory of data observed thus far."""
    self._arrays = {}

  def observe(self, key: str, data: jax.Array) -> None:
    """Adds data to the stored history of observed data.

    If the key is already in the observed set, then it appends it to the list.
    Otherwise, it adds a new entry to the arrays dictionary.

    Args:
      key: String identifier for this piece of data.
      data: JAX array of data.
    """
    if key in self._arrays:
      self._arrays[key].append(data)
    else:
      self._arrays[key] = [data]


def get_data_at_index(
    spectator: InMemoryJaxArraySpectator,
    index: int,
) -> dict[str, jax.Array]:
  return {key: data[index] for key, data in spectator.arrays.items()}
