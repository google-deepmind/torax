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

"""Backwards differentiable while loop with special structure."""

import dataclasses
import functools
from typing import Any, Callable

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing


# State and LoopStatistics are allowed to be any pytree.
# LoopStatistics must be integer dtypes.
_State = Any
_LoopStatistics = Any
_Counter = array_typing.IntScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WhileiLoopState:
  counter: _Counter
  state: _State
  loop_statistics: _LoopStatistics


@functools.partial(jax.custom_jvp, nondiff_argnums=[0, 1])
def whilei_loop(
    cond_fun: Callable[[_State, tuple[Any, ...]], jax.Array],
    compute_state: Callable[
        [_Counter, _LoopStatistics, tuple[Any, ...]],
        tuple[_State, _LoopStatistics],
    ],
    init_val: tuple[_State, _LoopStatistics],
    *args: tuple[Any, ...],
) -> WhileiLoopState:
  """Backward-mode differentiable while loop function.

  The while loop has a special structure such that we apply `compute_state`
  on incrementing iteration counters till `cond_fun` of the computed state
  returns False.

  i = 0
  state, loop_statistics = init_val
  while cond_fun(state, *args):
    state, loop_statistics = compute_state(i, loop_statistics, *args)
    i += 1

  Importantly any previously computed states are not used to compute future
  states.

  Functionality to accumulate statistics for a loop are exposed in the function
  signature but these are assumed to be integers that gradients will not be
  propagated through.

  Due to this special structure the gradient of the whilei_loop w.r.t any pytree
  args is the gradient of the computed final state w.r.t the pytree args.

  Args:
    cond_fun: A function that takes the current loop state and any other pytree
      args and returns a boolean that indicates whether the loop should
      continue. It is assumed that no pytrees are closed over into this
      function.
    compute_state: A function that takes the current counter, loop stats and
      any other pytree args and returns the next state and loop statistics.
    init_val: The initial state and initial loop statistics.
    *args: Additional pytree arguments to be passed to cond_fun and
      compute_state. The loop result state can be differentiated with respect to
      these args.
  """
  _, loop_statistics = init_val
  chex.assert_type(jax.tree_util.tree_leaves(loop_statistics), jnp.integer)

  def whilei_loop_body_fun(x: WhileiLoopState) -> WhileiLoopState:
    new_state, new_loop_statistics = compute_state(
        x.counter, x.loop_statistics, *args
    )
    return WhileiLoopState(
        counter=x.counter + 1,
        state=new_state,
        loop_statistics=new_loop_statistics,
    )

  def whilei_loop_cond_fun(x: WhileiLoopState) -> jax.Array:
    return cond_fun(x.state, *args)

  result = jax.lax.while_loop(
      whilei_loop_cond_fun,
      whilei_loop_body_fun,
      WhileiLoopState(
          counter=0,
          state=init_val[0],
          loop_statistics=init_val[1],
      ),
  )
  return result


@whilei_loop.defjvp
def whilei_loop_jvp(
    cond_fun,
    compute_state,
    primals,
    in_tangents,
):
  """Custom jvp for the whilei_loop function.

  This implicitly defines a vjp too because JAX automatically transposes the
  linear computation on the custom JVP rule:
  https://docs.jax.dev/en/latest/advanced-autodiff.html#basic-usage-of-jax-custom-jvp-and-jax-custom-vjp-apis.

  Args:
    cond_fun: Same as above.
    compute_state: Same as above.
    primals: The primal loop inputs.
    in_tangents: The input tangents.

  Returns:
    The loop result and the tangents of the loop result. The tangents of the
    counter and loop_statistics are set to zeros.
  """
  result = whilei_loop(
      cond_fun,
      compute_state,
      *primals,
  )
  final_counter = result.counter
  # JAX expects float0 tangent for non-differentiable integer types.
  counter_tangent = jnp.zeros_like(final_counter, dtype=jax.dtypes.float0)
  loop_statistics_tangent = jax.tree_util.tree_map(
      functools.partial(jnp.zeros_like, dtype=jax.dtypes.float0),
      result.loop_statistics,
  )

  def final_step():
    _, compute_state_tangents = jax.jvp(
        compute_state,
        # Compute the jvp for the final_counter - 1.
        (final_counter - 1, result.loop_statistics, *primals[1:]),
        (counter_tangent, loop_statistics_tangent, *in_tangents[1:]),
    )
    state_tangents, unused_loop_statistics_tangents = compute_state_tangents
    out_tangents_whilei_loop_state = WhileiLoopState(
        counter=counter_tangent,
        state=state_tangents,
        loop_statistics=loop_statistics_tangent,
    )
    return (
        result,
        out_tangents_whilei_loop_state,
    )

  def no_step():
    init_val_tangents = in_tangents[0]
    init_state_tangents, _ = init_val_tangents
    out_tangents_whilei_loop_state = WhileiLoopState(
        counter=counter_tangent,
        state=init_state_tangents,
        loop_statistics=loop_statistics_tangent,
    )

    return (result, out_tangents_whilei_loop_state)

  # If the loop ran at all compute final step result and tangents.
  return jax.lax.cond(final_counter == 0, no_step, final_step)
