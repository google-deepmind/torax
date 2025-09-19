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

import jax
from jax import numpy as jnp
from torax._src import array_typing


_State = Any
_AuxOutputs = Any
_Counter = array_typing.IntScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WhileiLoopState:
  counter: _Counter
  state: _State
  aux_outputs: _AuxOutputs


@functools.partial(jax.custom_jvp, nondiff_argnums=[0, 1])
def whilei_loop(
    compute_state: Callable[
        [_Counter, _AuxOutputs, tuple[Any, ...]], tuple[_State, _AuxOutputs]
    ],
    cond_fun: Callable[[_State], jax.Array],
    init_val: tuple[_State, _AuxOutputs],
    *args: tuple[Any, ...],
) -> WhileiLoopState:
  """Backward-mode differentiable while loop function.

  The while loop has a special structure such that we apply `compute_state`
  on incrementing iteration counters till cond_fun of the computed state
  returns False.

  Importantly any previously computed states are not used to compute future
  states.

  Due to this special structure the backward-mode grad can be computed as:
  grad(whilei_loop)(*args) = grad(compute_state)(final_i-1, *args).

  Args:
    compute_state: A function that takes the current counter, aux outputs and
      any other inputs and returns the next state and aux outputs. The
      aux_outputs are exposed in the function signature to allow for loop
      statistics to be accumulated but gradients are not propagated through
      them.
    cond_fun: A function that takes the current loop state and returns a boolean
      that indicates whether the loop should continue.
    init_val: The initial state and aux outputs.
    *args: Additional arguments to pass to the compute_state function.
  """

  def whilei_loop_body_fun(x: WhileiLoopState) -> WhileiLoopState:
    new_state, aux_outputs = compute_state(x.counter, x.aux_outputs, *args)
    return WhileiLoopState(
        counter=x.counter + 1.0,
        state=new_state,
        aux_outputs=aux_outputs,
    )

  def whilei_loop_cond_fun(x: WhileiLoopState) -> jax.Array:
    return cond_fun(x.state)

  result = jax.lax.while_loop(
      whilei_loop_cond_fun,
      whilei_loop_body_fun,
      WhileiLoopState(
          counter=0.0,
          state=init_val[0],
          aux_outputs=init_val[1],
      ),
  )
  return result


@whilei_loop.defjvp
def whilei_loop_jvp(
    compute_state,
    cond_fun,
    primals,
    in_tangents,
):
  """Custom jvp for the whilei_loop function.

  This implicitly defines a vjp too because JAX automatically transposes the
  linear computation on the custom JVP rule:
  https://docs.jax.dev/en/latest/advanced-autodiff.html#basic-usage-of-jax-custom-jvp-and-jax-custom-vjp-apis.

  Args:
    compute_state: Same as above.
    cond_fun: Same as above.
    primals: The primal loop inputs.
    in_tangents: The input tangents.

  Returns:
    The loop result and the tangents of the loop result. The tangents of the
    counter and aux_outputs are set to zeros.
  """
  result = whilei_loop(
      compute_state,
      cond_fun,
      *primals,
  )
  final_counter = result.counter

  def penultimate_step():
    _, out_tangents = jax.jvp(
        compute_state,
        # Compute the jvp for the final_counter - 1.
        (final_counter - 1, result.aux_outputs, *primals[1:]),
        (
            jnp.zeros_like(final_counter),
            jnp.zeros_like(result.aux_outputs),
            *in_tangents[1:],
        ),
    )
    state_tangents, _ = out_tangents
    out_tangents_whilei_loop_state = WhileiLoopState(
        counter=jnp.zeros_like(final_counter),
        state=state_tangents,
        aux_outputs=jnp.zeros_like(result.aux_outputs),
    )
    return (
        result,
        out_tangents_whilei_loop_state,
    )

  def no_step():
    init_val_tangents = in_tangents[0]
    init_state_tangents, _ = init_val_tangents
    out_tangents_whilei_loop_state = WhileiLoopState(
        counter=jnp.zeros_like(final_counter),
        state=init_state_tangents,
        aux_outputs=jnp.zeros_like(result.aux_outputs),
    )

    return (result, out_tangents_whilei_loop_state)

  # If the loop ran at all compute penultimate step result and tangents.
  return jax.lax.cond(final_counter == 0, no_step, penultimate_step)
