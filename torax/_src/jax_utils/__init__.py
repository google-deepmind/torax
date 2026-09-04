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
"""Commonly repeated JAX expressions."""

# pylint: disable=g-importing-member
from torax._src.jax_utils.common import assert_rank
from torax._src.jax_utils.common import batched_cond
from torax._src.jax_utils.common import enable_errors
from torax._src.jax_utils.common import env_bool
from torax._src.jax_utils.common import error_if
from torax._src.jax_utils.common import get_dtype
from torax._src.jax_utils.common import get_int_dtype
from torax._src.jax_utils.common import get_np_dtype
from torax._src.jax_utils.common import get_number_of_compiles
from torax._src.jax_utils.while_loop_bounded import while_loop_bounded

# pylint: enable=g-importing-member

__all__ = [
    'assert_rank',
    'batched_cond',
    'enable_errors',
    'env_bool',
    'error_if',
    'get_dtype',
    'get_int_dtype',
    'get_np_dtype',
    'get_number_of_compiles',
    'while_loop_bounded',
]
