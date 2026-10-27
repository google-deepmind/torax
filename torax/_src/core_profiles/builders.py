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

"""Builders for core profiles."""

from collections.abc import Mapping
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources import source_state as source_state_lib


def _build_flux_profiles(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
    min_rho_norm: array_typing.FloatScalar,
    v_loop_lcfs: array_typing.FloatScalar | None,
) -> state.FluxProfiles:
  """Constructs FluxProfiles derived from poloidal flux and geometry."""
  j_toroidal_total, j_toroidal_total_face, i_p_profile_face = (
      psi_calculations.calc_j_toroidal_total(
          geo,
          psi,
          min_rho_norm,
      )
  )
  if v_loop_lcfs is None:
    v_loop_lcfs = jnp.asarray(0.0, dtype=jax_utils.get_dtype())
  return state.FluxProfiles(
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
      I_p_profile_face=i_p_profile_face,
      v_loop_lcfs=v_loop_lcfs,
  )


def build_base_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    v_loop_lcfs: array_typing.FloatScalar | None = None,
) -> state.BaseCoreProfiles:
  """Constructs purely algebraic BaseCoreProfiles from SolverState.

  Used by the unified physics pipeline and initial state convergence.

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry of the tokamak.
    solver_state: The current set of primary profiles.
    v_loop_lcfs: Loop voltage at the LCFS [V]. If None, defaults to 0.0.

  Returns:
    BaseCoreProfiles with kinetic, ions, flux, and toroidal angular velocity.
  """
  ion_composition, n_i, n_impurity = getters.get_updated_ions(
      runtime_params,
      geo,
      solver_state.n_e,
      solver_state.T_e,
  )
  flux = _build_flux_profiles(
      geo,
      solver_state.psi,
      runtime_params.numerics.min_rho_norm,
      v_loop_lcfs=v_loop_lcfs,
  )
  kinetic = state.KineticProfiles(
      T_i=solver_state.T_i,
      T_e=solver_state.T_e,
      n_e=solver_state.n_e,
      n_i=n_i,
      n_impurity=n_impurity,
  )
  return state.BaseCoreProfiles(
      kinetic=kinetic,
      ions=ion_composition,
      flux=flux,
      toroidal_angular_velocity=solver_state.toroidal_angular_velocity,
  )


def build_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
    *,
    explicit: bool,
    explicit_source_outputs: (
        Mapping[str, source_profiles_lib.SourceOutput] | None
    ) = None,
    calculate_anyway: bool = False,
    v_loop_lcfs: array_typing.FloatScalar | None = None,
) -> tuple[
    state.CoreProfiles,
    dict[str, source_profiles_lib.SourceOutput],
    dict[str, source_state_lib.SourceState],
    bootstrap_current_base.BootstrapCurrent,
    neoclassical_intermediates_lib.NeoclassicalIntermediates,
]:
  """Builds core profiles, basic source outputs, and neoclassical outputs.

  Args:
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the tokamak.
    solver_state: The current set of primary profiles.
    source_models: Source models.
    neoclassical_model: Neoclassical model.
    source_states: Source states mapping.
    explicit: Whether to evaluate explicit sources.
    explicit_source_outputs: Pre-evaluated explicit source outputs, if any.
    calculate_anyway: Whether to calculate sources even if inactive.
    v_loop_lcfs: Optional loop voltage at the LCFS [V].

  Returns:
    A tuple containing:
      - core_profiles: Fully constructed CoreProfiles.
      - basic_source_outputs: Fast-ion and psi source outputs.
      - updated_source_states: Source states updated by basic sources.
      - bootstrap_current: Bootstrap current output.
      - neoclassical_intermediates: Neoclassical intermediates for transport.
  """
  if (
      v_loop_lcfs is None
      and runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
  ):
    v_loop_lcfs = jnp.asarray(
        runtime_params.profile_conditions.v_loop_lcfs,
        dtype=jax_utils.get_dtype(),
    )

  # Construct profiles that depend only on the solver variables.
  base_profiles = build_base_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      v_loop_lcfs=v_loop_lcfs,
  )

  # Evaluate sources that only depend on the solver profiles. These sources
  # may produce fast ions and current drive.
  source_outputs_pass, updated_source_states = (
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=base_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=explicit,
          calculate_anyway=calculate_anyway,
      )
  )
  if explicit_source_outputs is not None:
    basic_source_outputs = {
        **explicit_source_outputs,
        **source_outputs_pass,
    }
  else:
    basic_source_outputs = source_outputs_pass

  if runtime_params.numerics.enable_fast_ions:
    fast_ions = profile_conditions.apply_prescribed_fast_ions(
        source_profiles_lib.extract_fast_ions(basic_source_outputs),
        runtime_params.profile_conditions.prescribed_fast_ions,
        geo.rho_face_norm,
    )
  else:
    fast_ions = ()

  # TODO(b/572586768): Pass BaseCoreProfiles and fast_ions directly once
  # neoclassical models migrate to BaseCoreProfiles.
  temp_core_profiles = state.CoreProfiles(
      kinetic=base_profiles.kinetic,
      ions=base_profiles.ions,
      flux=base_profiles.flux,
      toroidal_angular_velocity=base_profiles.toroidal_angular_velocity,
      fast_ions=fast_ions,
      conductivity=state.Conductivity.zeros(geo),
      poloidal_velocity=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
      psidot=cell_variable.CellVariable(
          value=jnp.zeros_like(geo.rho_norm),
          face_centers=geo.rho_face_norm,
      ),
  )
  (
      conductivity,
      bootstrap_current,
      poloidal_velocity,
      neoclassical_intermediates,
  ) = neoclassical_model.calculate_conductivity_current_and_rotation(
      runtime_params, geo, temp_core_profiles
  )

  total_psi_sources = source_profiles_lib.sum_total_psi_sources(
      basic_source_outputs=basic_source_outputs,
      source_models=source_models,
      bootstrap_current=bootstrap_current,
      geo=geo,
  )
  # Combine psi sources with core profiles to compute psidot.
  psidot = psi_calculations.calculate_psidot(
      psi=base_profiles.flux.psi,
      sigma=conductivity.sigma,
      psi_sources=total_psi_sources,
      geo=geo,
      evolve_current=runtime_params.numerics.evolve_current,
      prescribed_psidot=runtime_params.profile_conditions.psidot,
      v_loop_lcfs=v_loop_lcfs,
  )
  core_profiles = state.CoreProfiles(
      kinetic=base_profiles.kinetic,
      ions=base_profiles.ions,
      flux=base_profiles.flux,
      toroidal_angular_velocity=base_profiles.toroidal_angular_velocity,
      fast_ions=fast_ions,
      conductivity=conductivity,
      poloidal_velocity=poloidal_velocity.v_pol,
      psidot=psidot,
  )

  return (
      core_profiles,
      basic_source_outputs,
      updated_source_states,
      bootstrap_current,
      neoclassical_intermediates,
  )
