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
"""Helpers for tests using core profiles."""

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import flux_profiles
from torax._src.core_profiles import initialization
from torax._src.core_profiles import ion_composition
from torax._src.core_profiles import kinetic_profiles
from torax._src.core_profiles import profiles
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.orchestration import physics_pipeline
from torax._src.output_tools import output_keys
from torax._src.physics import charge_states
from torax._src.torax_pydantic import config


def make_initial_inputs(
    torax_config: config.ToraxConfig,
    t_override: float | None = None,
) -> tuple[
    profiles.CoreProfiles,
    runtime_params_lib.RuntimeParams,
    geometry.Geometry,
]:
  """Builds initial CoreProfiles, RuntimeParams, and Geometry.

  Args:
    torax_config: The TORAX configuration.
    t_override: Optional timestamp override; defaults to
      `torax_config.time_step_calculator.t_initial`.

  Returns:
    A tuple of (core_profiles, runtime_params, geo).
  """
  models = torax_config.build_models()
  runtime_params, geo = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=(
              torax_config.time_step_calculator.t_initial
              if t_override is None
              else t_override
          ),
          runtime_params_provider=(
              build_runtime_params.RuntimeParamsProvider.from_config(
                  torax_config
              )
          ),
          geometry_provider=torax_config.geometry.build_provider,
          is_initialization=True,
      )
  )
  solver_state = initialization.initial_solver_state(
      runtime_params=runtime_params,
      geo=geo,
      source_models=models.source_models,
      neoclassical_model=models.neoclassical_model,
      source_states=models.source_models.initial_state(runtime_params),
  )
  physics_outputs, _ = physics_pipeline.evaluate_initial_physics(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      models=models,
  )
  return physics_outputs.core_profiles, runtime_params, geo


# pylint: disable=invalid-name
def make_zero_core_profiles(
    geo: geometry.Geometry,
    T_e: cell_variable.CellVariable | None = None,
    Z_impurity: jax.Array | None = None,
    Z_impurity_face: jax.Array | None = None,
    impurity_names: tuple[str, ...] = ("dummy_impurity",),
    main_ion_names: tuple[str, ...] = ("dummy_main_ion",),
) -> profiles.CoreProfiles:
  """Returns a dummy CoreProfiles object."""
  zero_cell_variable = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho),
      face_centers=geo.rho_face_norm,
      right_face_constraint=jnp.ones(()),
      right_face_grad_constraint=None,
  )
  impurity_fractions_dict = {
      name: jnp.zeros_like(geo.rho) for name in impurity_names
  }
  main_ion_fractions_dict = {name: jnp.array(0.0) for name in main_ion_names}
  zero_charge_state_info = charge_states.ChargeStateInfo(
      Z_avg=jnp.zeros_like(geo.rho),
      Z2_avg=jnp.zeros_like(geo.rho),
      Z_per_species={name: jnp.zeros_like(geo.rho) for name in impurity_names},
  )
  zero_charge_state_info_face = charge_states.ChargeStateInfo(
      Z_avg=jnp.zeros_like(geo.rho_face),
      Z2_avg=jnp.zeros_like(geo.rho_face),
      Z_per_species={
          name: jnp.zeros_like(geo.rho_face) for name in impurity_names
      },
  )
  kinetic = kinetic_profiles.KineticProfiles(
      T_i=zero_cell_variable,
      T_e=T_e if T_e is not None else zero_cell_variable,
      n_e=zero_cell_variable,
      n_i=zero_cell_variable,
      n_impurity=zero_cell_variable,
  )
  ions = ion_composition.IonComposition(
      impurity_fractions=impurity_fractions_dict,
      main_ion_fractions=main_ion_fractions_dict,
      Z_i=jnp.zeros_like(geo.rho),
      Z_i_face=jnp.zeros_like(geo.rho_face),
      Z_impurity=Z_impurity
      if Z_impurity is not None
      else jnp.zeros_like(geo.rho),
      Z_impurity_face=Z_impurity_face
      if Z_impurity_face is not None
      else jnp.zeros_like(geo.rho_face),
      A_i=jnp.zeros(()),
      A_impurity=jnp.zeros_like(geo.rho),
      A_impurity_face=jnp.zeros_like(geo.rho_face),
      Z_eff=jnp.zeros_like(geo.rho),
      Z_eff_face=jnp.zeros_like(geo.rho_face),
      charge_state_info=zero_charge_state_info,
      charge_state_info_face=zero_charge_state_info_face,
  )
  flux = flux_profiles.FluxProfiles(
      psi=zero_cell_variable,
      q_face=jnp.zeros_like(geo.rho_face),
      s_face=jnp.zeros_like(geo.rho_face),
      j_toroidal_total=jnp.zeros_like(geo.rho),
      j_toroidal_total_face=jnp.zeros_like(geo.rho_face),
      I_p_profile_face=jnp.zeros_like(geo.rho_face),
      v_loop_lcfs=jnp.array(0.0),
  )
  conductivity = profiles.Conductivity(
      sigma=jnp.zeros_like(geo.rho),
      sigma_face=jnp.zeros_like(geo.rho_face),
  )
  return profiles.CoreProfiles(
      kinetic=kinetic,
      ions=ions,
      flux=flux,
      toroidal_angular_velocity=zero_cell_variable,
      fast_ions=(),
      conductivity=conductivity,
      poloidal_velocity=zero_cell_variable,
      psidot=zero_cell_variable,
  )


def make_constant_core_profile(
    geo: geometry.Geometry,
    value: float,
) -> cell_variable.CellVariable:
  return cell_variable.CellVariable(
      value=value * np.ones_like(geo.rho_norm),
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=np.zeros(()),
      left_face_constraint=None,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(value),
  )


def verify_core_profiles(
    ref_profiles: dict[str, array_typing.Array],
    index: int,
    core_profiles: profiles.CoreProfiles,
):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.kinetic.T_e.value,
      ref_profiles[output_keys.T_E][index, 1:-1],
      err_msg=(
          "Mismatch for T_e.value. Core profile:"
          f" {core_profiles.kinetic.T_e.value}, Ref profile"
          f" ({output_keys.T_E}): {ref_profiles[output_keys.T_E][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.kinetic.T_i.value,
      ref_profiles[output_keys.T_I][index, 1:-1],
      err_msg=(
          "Mismatch for T_i.value. Core profile:"
          f" {core_profiles.kinetic.T_i.value}, Ref profile"
          f" ({output_keys.T_I}): {ref_profiles[output_keys.T_I][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.kinetic.n_e.value,
      ref_profiles[output_keys.N_E][index, 1:-1],
      err_msg=(
          "Mismatch for n_e.value. Core profile:"
          f" {core_profiles.kinetic.n_e.value}, Ref profile"
          f" ({output_keys.N_E}): {ref_profiles[output_keys.N_E][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
      core_profiles.kinetic.n_e.right_face_constraint,
      ref_profiles[output_keys.N_E][index, -1],
      err_msg=(
          "Mismatch for n_e.right_face_constraint. Core profile:"
          f" {core_profiles.kinetic.n_e.right_face_constraint}, Ref profile"
          f" ({output_keys.N_E} end val):"
          f" {ref_profiles[output_keys.N_E][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.kinetic.n_i.value,
      ref_profiles[output_keys.N_I][index, 1:-1],
      err_msg=(
          "Mismatch for n_i.value. Core profile:"
          f" {core_profiles.kinetic.n_i.value}, Ref profile"
          f" ({output_keys.N_I}): {ref_profiles[output_keys.N_I][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
      core_profiles.kinetic.n_i.right_face_constraint,
      ref_profiles[output_keys.N_I][index, -1],
      err_msg=(
          "Mismatch for n_i.right_face_constraint. Core profile:"
          f" {core_profiles.kinetic.n_i.right_face_constraint}, Ref profile"
          f" ({output_keys.N_I} end val):"
          f" {ref_profiles[output_keys.N_I][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.flux.psi.value,
      ref_profiles[output_keys.PSI][index, 1:-1],
      err_msg=(
          "Mismatch for psi.value. Core profile:"
          f" {core_profiles.flux.psi.value}, Ref profile"
          f" ({output_keys.PSI}): {ref_profiles[output_keys.PSI][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value,
      ref_profiles[output_keys.V_LOOP][index, 1:-1],
      err_msg=(
          "Mismatch for psidot.value. Core profile:"
          f" {core_profiles.psidot.value}, Ref profile ({output_keys.V_LOOP}):"
          f" {ref_profiles[output_keys.V_LOOP][index, 1:-1]}"
      ),
  )

  np.testing.assert_allclose(
      core_profiles.flux.q_face,
      ref_profiles[output_keys.Q][index, :],
      err_msg=(
          f"Mismatch for q_face. Core profile: {core_profiles.flux.q_face}, Ref"
          f" profile ({output_keys.Q}): {ref_profiles[output_keys.Q][index, :]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.flux.s_face,
      ref_profiles[output_keys.MAGNETIC_SHEAR][index, :],
      err_msg=(
          f"Mismatch for s_face. Core profile: {core_profiles.flux.s_face},"
          f" Ref profile ({output_keys.MAGNETIC_SHEAR}):"
          f" {ref_profiles[output_keys.MAGNETIC_SHEAR][index, :]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.flux.j_toroidal_total_face[0],
      ref_profiles[output_keys.J_TOROIDAL_TOTAL][index, 0],
      err_msg=(
          "Mismatch for j_toroidal_total_face[0]. Core profile:"
          f" {core_profiles.flux.j_toroidal_total_face[0]}, Ref profile"
          f" ({output_keys.J_TOROIDAL_TOTAL} start val):"
          f" {ref_profiles[output_keys.J_TOROIDAL_TOTAL][index, 0]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.flux.j_toroidal_total_face[-1],
      ref_profiles[output_keys.J_TOROIDAL_TOTAL][index, -1],
      err_msg=(
          "Mismatch for j_toroidal_total_face[-1]. Core profile:"
          f" {core_profiles.flux.j_toroidal_total_face[-1]}, Ref profile"
          f" ({output_keys.J_TOROIDAL_TOTAL} end val):"
          f" {ref_profiles[output_keys.J_TOROIDAL_TOTAL][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.flux.I_p_profile_face,
      ref_profiles[output_keys.I_P_PROFILE][index, :],
      err_msg=(
          "Mismatch for I_p_profile_face. Core profile:"
          f" {core_profiles.flux.I_p_profile_face}, Ref profile"
          f" ({output_keys.I_P_PROFILE}):"
          f" {ref_profiles[output_keys.I_P_PROFILE][index, :]}"
      ),
  )
