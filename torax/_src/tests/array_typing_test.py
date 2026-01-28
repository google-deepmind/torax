# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import dataclasses
import os
from unittest import mock

import jax
import jax.numpy as jnp
import jaxtyping as jt
from absl.testing import absltest

from torax._src import array_typing


def _f_invalid_shape(
    x: jt.Float[jax.Array, "size size"],
) -> jt.Float[jax.Array, "size"]:
    return x**2


@dataclasses.dataclass(frozen=True)
class TestClass:
    x: jt.Float[jax.Array, "size size"]


class ArrayTypingTest(absltest.TestCase):
    # This test ensures that runtime type checking is always turned on for tests,
    # despite being turned off by default.
    def test_invalid_shape(self):
        x = jnp.ones((3, 3))
        f = array_typing.jaxtyped(_f_invalid_shape)
        with self.assertRaises(jt.TypeCheckError):
            f(x)

    @mock.patch.dict(os.environ, {"TORAX_JAXTYPING": "false"})
    def test_invalid_shape_disabled(self):
        x = jnp.ones((3, 3))
        f = array_typing.jaxtyped(_f_invalid_shape)
        f(x)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_invalid_shape_default_enabled(self):
        x = jnp.ones((3, 3))
        f = array_typing.jaxtyped(_f_invalid_shape)
        with self.assertRaises(jt.TypeCheckError):
            f(x)

    def test_dataclass(self):
        test_class = array_typing.jaxtyped(TestClass)
        with self.assertRaises(jt.TypeCheckError):
            test_class(x=jnp.ones((3,)))

        test_class(x=jnp.ones((3, 3)))


if __name__ == "__main__":
    absltest.main()
