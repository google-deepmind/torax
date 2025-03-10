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

"""Test that torax uses consistent order of arguments.

Many argument names are used commonly throughout the whole library and it
is easiest to remember how to pass them if they are always used in the same
order.
"""
import inspect
from absl.testing import absltest
from absl.testing import parameterized
import torax
from torax import physics
from torax import sim
from torax import state
from torax.core_profiles import updaters
from torax.fvm import block_1d_coeffs
from torax.fvm import calc_coeffs
from torax.fvm import discrete_system
from torax.fvm import implicit_solve_block
from torax.fvm import newton_raphson_solve_block
from torax.fvm import optimizer_solve_block
from torax.fvm import residual_and_loss
from torax.sources import bootstrap_current_source
from torax.sources import formulas
from torax.sources import generic_current_source
from torax.sources import qei_source
from torax.sources import source
from torax.sources import source_models
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import predictor_corrector_method
from torax.stepper import stepper
from torax.transport_model import qlknn_transport_model


class ArgOrderTest(parameterized.TestCase):
  """Tests that argument order are correct."""

  counts = {}
  THRESHOLD = 3

  @parameterized.parameters([
      dict(module=calc_coeffs),
      dict(module=sim),
      dict(module=updaters),
      dict(module=physics),
      dict(module=sim),
      dict(module=state),
      dict(module=block_1d_coeffs),
      dict(module=discrete_system),
      dict(module=implicit_solve_block),
      dict(module=newton_raphson_solve_block),
      dict(module=optimizer_solve_block),
      dict(module=residual_and_loss),
      dict(module=bootstrap_current_source),
      dict(module=generic_current_source),
      dict(module=formulas),
      dict(module=qei_source),
      dict(module=source),
      dict(module=source_models),
      dict(module=linear_theta_method),
      dict(module=nonlinear_theta_method),
      dict(module=predictor_corrector_method),
      dict(module=stepper),
      dict(module=qlknn_transport_model),
  ])
  def test_arg_order(self, module):
    """Test that the functions in a module respect the canonical order."""
    fields = inspect.getmembers(module)
    print(module.__name__)
    for name, obj in fields:
      if name.startswith("_"):
        # Ignore private fields and methods.
        continue
      if inspect.isfunction(obj):
        print("\t", name)
        params = inspect.signature(obj).parameters.keys()
        print("\t\traw params: ", params)
        for param in params:
          if param not in self.counts:
            self.counts[param] = 1
          else:
            self.counts[param] += 1
            if self.counts[param] >= self.THRESHOLD:
              if param not in torax.CANONICAL_ORDER:
                for other in torax.CANONICAL_ORDER:
                  print(param, other, param == other)
                raise TypeError(
                    "Common param not in canonical order: param "
                    f'"{param}" of {module.__name__}.{name}'
                )
        params = [param for param in params if param in torax.CANONICAL_ORDER]
        print("\t\tFiltered params: ", params)
        idxs = [torax.CANONICAL_ORDER.index(param) for param in params]
        print("\t\tidxs: ", idxs)
        for i in range(1, len(idxs)):
          if idxs[i] <= idxs[i - 1]:
            first_arg_name = params[i - 1]
            second_arg_name = params[i]
            raise TypeError(
                f'Arguments of name "{first_arg_name}" need to '
                f'come after arguments of name "{second_arg_name}" '
                f" but function `{name}` in module `{module.__name__}`"
                " does not obey this."
            )


if __name__ == "__main__":
  absltest.main()
