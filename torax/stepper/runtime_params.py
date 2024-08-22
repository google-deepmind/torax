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

"""Runtime parameters used inside the steppers."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, TypeAlias

import chex
from torax import interpolated_param


TimeInterpolated: TypeAlias = interpolated_param.TimeInterpolated


@dataclasses.dataclass(kw_only=True)
class RuntimeParams:
  """Runtime parameters used inside the stepper objects.

  These are general runtime parameters, some of which are time-dependent.
  Individual stepper classes can also define their own input runtime parameter
  classes which extend this one and add extra, stepper-specific parameters.

  Internally to TORAX, this class is interpolated down to a single "slice" of
  dynamic params for a particular time step.
  """

  # Most of the params below are used by one of the solve methods in fvm/, but
  # are accessed by both the nonlinear and linear steppers. That is why they
  # are located in this general stepper config class here.

  # theta value in the theta method.
  # 0 = explicit, 1 = fully implicit, 0.5 = Crank-Nicolson
  theta_imp: float = 1.0
  # Enables predictor_corrector iterations with the linear solver.
  # If False, compilation is faster
  predictor_corrector: bool = True
  # Number of corrector steps for the predictor-corrector linear solver.
  # 0 means a pure linear solve with no corrector steps.
  corrector_steps: int = 1
  # See `fvm.convection_terms` docstring, `dirichlet_mode` argument
  convection_dirichlet_mode: str = 'ghost'
  # See `fvm.convection_terms` docstring, `neumann_mode` argument
  convection_neumann_mode: str = 'ghost'
  # use pereverzev terms for linear solver. Is only applied in the nonlinear
  # solver for the optional initial guess from the linear solver
  use_pereverzev: bool = False
  # (deliberately) large heat conductivity for Pereverzev rule
  chi_per: float = 20.0
  # (deliberately) large particle diffusion for Pereverzev rule
  d_per: float = 10.0

  def __post_init__(self):
    assert self.theta_imp >= 0.0
    assert self.theta_imp <= 1.0
    assert self.corrector_steps >= 0
    _check_config_param_in_set(
        'convection_dirichlet_mode',
        self.convection_dirichlet_mode,
        ['ghost', 'direct', 'semi-implicit'],
    )
    _check_config_param_in_set(
        'convection_neumann_mode',
        self.convection_neumann_mode,
        ['ghost', 'semi-implicit'],
    )

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    del t  # Unused.
    return DynamicRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        corrector_steps=self.corrector_steps,
    )

  def build_static_params(self) -> StaticRuntimeParams:
    return StaticRuntimeParams(
        theta_imp=self.theta_imp,
        convection_dirichlet_mode=self.convection_dirichlet_mode,
        convection_neumann_mode=self.convection_neumann_mode,
        use_pereverzev=self.use_pereverzev,
        predictor_corrector=self.predictor_corrector,
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the stepper which can be used as compiled args."""

  chi_per: float
  d_per: float
  corrector_steps: int


@chex.dataclass(frozen=True)
class StaticRuntimeParams:
  """Static params for the stepper."""

  theta_imp: float
  convection_dirichlet_mode: str
  convection_neumann_mode: str
  use_pereverzev: bool
  predictor_corrector: bool


def _check_config_param_in_set(
    param_name: str,
    param_value: Any,
    valid_values: Iterable[Any],
) -> None:
  if param_value not in valid_values:
    raise ValueError(
        f'{param_name} invalid. Must give {" or ".join(valid_values)}. '
        f'Provided: {param_value}.'
    )
