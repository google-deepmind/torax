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
from torax._src import physics
from torax._src import state
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import discrete_system
from torax._src.fvm import implicit_solve_block
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import optimizer_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.orchestration import run_loop
from torax._src.solver import linear_theta_method
from torax._src.solver import nonlinear_theta_method
from torax._src.solver import predictor_corrector_method
from torax._src.solver import solver
from torax._src.sources import formulas
from torax._src.sources import generic_current_source
from torax._src.sources import qei_source
from torax._src.sources import source
from torax._src.sources import source_models
from torax._src.transport_model import qlknn_transport_model


# Throughout TORAX, we maintain the following canonical argument order for
# common argument names passed to many functions. This is a stylistic
# convention that helps to remember the order of arguments for a function.
# For each individual function only a subset of these are
# passed, but the order should be maintained.
_CANONICAL_ORDER = [
    't',
    'dt',
    'source_type',
    'static_runtime_params_slice',
    'static_source_runtime_params',
    'dynamic_runtime_params_slice',
    'dynamic_runtime_params_slice_t',
    'dynamic_runtime_params_slice_t_plus_dt',
    'dynamic_runtime_params_slice_provider',
    'unused_config',
    'dynamic_source_runtime_params',
    'geo',
    'geo_t',
    'geo_t_plus_dt',
    'geometry_provider',
    'source_name',
    'x_old',
    'state',
    'unused_state',
    'core_profiles',
    'core_profiles_t',
    'core_profiles_t_plus_dt',
    'T_i',
    'T_e',
    'n_e',
    'n_i',
    'psi',
    'transport_model',
    'source_profiles',
    'source_profile',
    'explicit_source_profiles',
    'model_func',
    'source_models',
    'pedestal_model',
    'time_step_calculator',
    'coeffs_callback',
    'evolving_names',
    'step_fn',
    'spectator',
    'explicit',
    'maxiter',
    'tol',
    'delta_reduction_factor',
    'file_restart',
    'ds',
]


class ArgOrderTest(parameterized.TestCase):
  """Tests that argument order are correct."""

  counts = {}
  THRESHOLD = 3

  @parameterized.parameters([
      dict(module=calc_coeffs),
      dict(module=run_loop),
      dict(module=updaters),
      dict(module=physics),
      dict(module=state),
      dict(module=block_1d_coeffs),
      dict(module=discrete_system),
      dict(module=implicit_solve_block),
      dict(module=newton_raphson_solve_block),
      dict(module=optimizer_solve_block),
      dict(module=residual_and_loss),
      dict(module=generic_current_source),
      dict(module=formulas),
      dict(module=qei_source),
      dict(module=source),
      dict(module=source_models),
      dict(module=linear_theta_method),
      dict(module=nonlinear_theta_method),
      dict(module=predictor_corrector_method),
      dict(module=solver),
      dict(module=qlknn_transport_model),
  ])
  def test_arg_order(self, module):
    """Test that the functions in a module respect the canonical order."""
    fields = inspect.getmembers(module)
    print(module.__name__)
    for name, obj in fields:
      if name.startswith('_'):
        # Ignore private fields and methods.
        continue
      if inspect.isfunction(obj):
        print('\t', name)
        params = inspect.signature(obj).parameters.keys()
        print('\t\traw params: ', params)
        for param in params:
          if param not in self.counts:
            self.counts[param] = 1
          else:
            self.counts[param] += 1
            if self.counts[param] >= self.THRESHOLD:
              if param not in _CANONICAL_ORDER:
                for other in _CANONICAL_ORDER:
                  print(param, other, param == other)
                raise TypeError(
                    'Common param not in canonical order: param '
                    f'`{param}` of {module.__name__}.{name}'
                )
        params = [param for param in params if param in _CANONICAL_ORDER]
        print('\t\tFiltered params: ', params)
        idxs = [_CANONICAL_ORDER.index(param) for param in params]
        print('\t\tidxs: ', idxs)
        for i in range(1, len(idxs)):
          if idxs[i] <= idxs[i - 1]:
            first_arg_name = params[i - 1]
            second_arg_name = params[i]
            raise TypeError(
                f'Arguments of name `{first_arg_name}` need to '
                f'come after arguments of name `{second_arg_name}` '
                f' but function `{name}` in module `{module.__name__}`'
                ' does not obey this.'
            )


if __name__ == '__main__':
  absltest.main()
