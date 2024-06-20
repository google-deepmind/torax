import jax
import dataclasses
import time
from absl import logging
import jax
import jax.numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.config import build_sim
from torax.sim import SimulationStepFn
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

CONFIG = {
    'runtime_params': {
        'plasma_composition': {
            'Ai': 2.5,
            'Zeff': 2.0,
            'Zimp': 10,
        },
        'profile_conditions': {
            'Ip': 11.5,
            'Ti_bound_left': 8,
            'Ti_bound_right': 0.2,
            'Te_bound_left': 8,
            'Te_bound_right': 0.2,
            'ne_bound_right': 0.5,
            'nbar_is_fGW': False,
            'nbar': 0.8,
            'npeak': 1.0,
            'set_pedestal': False,
            'Tiped': 4.0,
            'Teped': 4.0,
            'neped': 0.8,
            'Ped_top': 0.93,
            'nu': 2,
        },
        'numerics': {
            't_final': 3.0,
            'exact_t_final': True,
            'fixed_dt': 0.1,
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'current_eq': True,
            'dens_eq': True,
            'largeValue_T': 1.0e12,
            'largeValue_n': 1.0e8,
            'resistivity_mult': 1,
        },
    },
    'geometry': {
        'nr': 50,
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'Rmaj': 6.2,
        'Rmin': 2.0,
        'B0': 5.3,
    },
    'sources': {
        'j_bootstrap': {},
        'nbi_particle_source': {
            'S_nbi_tot': 3e+21,
            'nbi_deposition_location': 0.2,
            'nbi_particle_width': 0.25,
        },
        'generic_ion_el_heat_source': {
            'rsource': 0.11,
            'w': 0.2,
            'Ptot': 50e6,
            'el_heat_fraction': 0.5,
        },
        'ohmic_heat_source': {},
        'fusion_heat_source': {},
        'qei_source': {},
    },
    'transport': {
        'transport_model': 'constant',
        'chimin': 0.05,
        'chimax': 100.0,
        'Demin': 0.05,
        'Demax': 100.0,
        'Vemin': -50.0,
        'Vemax': 50.0,
        'apply_inner_patch': False,
        'apply_outer_patch': False,
        'constant_params': {
            'chii_const': 2.0,
            'chie_const': 1.0,
            'De_const': 1.0,
            'Ve_const': -0.15
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'theta_imp': 1.0,
        'predictor_corrector': True,
        'corrector_steps': 5,
        'convection_dirichlet_mode': 'ghost',
        'convection_neumann_mode': 'ghost',
        'use_pereverzev': False,
        'chi_per': 20.0,
        'd_per': 10.0,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed'
    }
}

# Helper functions from sim.py
def _get_initial_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> source_profiles_lib.SourceProfiles:
  """Returns the "implicit" profiles for the initial state in run_simulation().

  The source profiles returned as part of each time step's state in
  run_simulation() is computed based on the core profiles at that time step.
  However, for the first time step, only the explicit sources are computed.
  The
  implicit profiles are computed based on the "live" state that is evolving,
  "which depending on the stepper used is either consistent (within tolerance)
  with the state at the next timestep (nonlinear solver), or an approximation
  thereof (linear stepper or predictor-corrector)." So for the first time step,
  we need to prepopulate the state with the implicit profiles for the starting
  core profiles.

  Args:
    static_runtime_params_slice: Runtime parameters which, when they change,
      trigger recompilations. They should not change within a single run of the
      sim.
    dynamic_runtime_params_slice: Runtime parameters which may change from time
      step to time step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.

  Returns:
    SourceProfiles from implicit source models based on the core profiles from
    the starting state.
  """
  implicit_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=False,
  )
  qei = source_models.qei_source.get_qei(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.qei_source_name
      ],
      geo=geo,
      core_profiles=core_profiles,
  )
  implicit_profiles = dataclasses.replace(implicit_profiles, qei=qei)
  return implicit_profiles

def _log_timestep(t: jax.Array, dt: jax.Array, stepper_iterations: int) -> None:
  """Logs basic timestep info."""
  logging.info(
      '\nSimulation time: %.5f, previous dt: %.6f, previous stepper'
      ' iterations: %d',
      t,
      dt,
      stepper_iterations,
  )

# This function can be jitted if source_models is a static argument. However,
# in our tests, jitting this function actually slightly slows down runs, so this
# is left as pure python.
def _merge_source_profiles(
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    implicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    qei_core_profiles: state.CoreProfiles,
) -> source_profiles_lib.SourceProfiles:
  """Returns a SourceProfiles that merges the input profiles.

  Sources can either be explicit or implicit. The explicit_source_profiles
  contain the profiles for all source models that are set to explicit, and it
  contains profiles with all zeros for any implicit source. The opposite holds
  for the implicit_source_profiles.

  This function adds the two dictionaries of profiles and returns a single
  SourceProfiles that includes both.

  Args:
    explicit_source_profiles: Profiles from explicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the explicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.
    implicit_source_profiles: Profiles from implicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the implicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.
    source_models: Source models used to compute the profiles given.
    qei_core_profiles: The core profiles used to compute the Qei source.

  Returns:
    A SourceProfiles with non-zero profiles for all sources, both explicit and
    implicit (assuming the source model outputted a non-zero profile).
  """
  sum_profiles = lambda a, b: a + b
  summed_bootstrap_profile = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.j_bootstrap,
      implicit_source_profiles.j_bootstrap,
  )
  summed_qei_info = jax.tree_util.tree_map(
      sum_profiles, explicit_source_profiles.qei, implicit_source_profiles.qei
  )
  summed_other_profiles = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.profiles,
      implicit_source_profiles.profiles,
  )
  # For ease of comprehension, we convert the units of the Qei source and add it
  # to the list of other profiles before returning it.
  summed_other_profiles[source_models.qei_source_name] = (
      summed_qei_info.qei_coef
      * (qei_core_profiles.temp_el.value - qei_core_profiles.temp_ion.value)
  )
  # Also for better comprehension of the outputs, any known source model output
  # profiles that contain both the ion and electron heat profiles (i.e. they are
  # 2D with the separate profiles stacked) are unstacked here.
  for source_name in source_models.ion_el_sources:
    if source_name in summed_other_profiles:
      # The profile in summed_other_profiles is 2D. We want to split it up.
      if (
          f'{source_name}_ion' in summed_other_profiles
          or f'{source_name}_el' in summed_other_profiles
      ):
        raise ValueError(
            'Source model names are too close. Trying to save the output from '
            f'the source {source_name} in 2 components: {source_name}_ion and '
            f'{source_name}_el, but there is already a source output with that '
            'name. Rename your sources to avoid this collision.'
        )
      summed_other_profiles[f'{source_name}_ion'] = summed_other_profiles[
          source_name
      ][0, ...]
      summed_other_profiles[f'{source_name}_el'] = summed_other_profiles[
          source_name
      ][1, ...]
      del summed_other_profiles[source_name]
  return source_profiles_lib.SourceProfiles(
      profiles=summed_other_profiles,
      j_bootstrap=summed_bootstrap_profile,
      qei=summed_qei_info,
  )

# Single TORAX step in while loop
def run_single_step(
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geometry_provider,
        sim_state,
        step_fn):
  
  # Measure how long in wall clock time each simulation step takes.
  step_start_time = time.time()
  _log_timestep(sim_state.t, sim_state.dt, sim_state.stepper_iterations)
  explicit_source_profiles = source_models_lib.build_source_profiles(
    dynamic_runtime_params_slice=dynamic_runtime_params_slice,
    geo=geo,
    core_profiles=sim_state.core_profiles,
    source_models=step_fn.stepper.source_models,
    explicit=True,
  )

  sim_state.core_sources = _merge_source_profiles(
      explicit_source_profiles=explicit_source_profiles,
      implicit_source_profiles=sim_state.core_sources,
      source_models=step_fn.stepper.source_models,
      qei_core_profiles=sim_state.core_profiles,
  )  
  
  sim_state = step_fn(
      static_runtime_params_slice,
      dynamic_runtime_params_slice_provider,
      geometry_provider,
      sim_state,
      explicit_source_profiles,
  )

  wall_clock_step_time = (time.time() - step_start_time)

  return sim_state, wall_clock_step_time

# Build and launch TORAX sim

jax.config.update('jax_enable_x64', True)

runtime_params = build_sim.build_runtime_params_from_config(
    CONFIG['runtime_params']
)
geo = build_sim.build_geometry_from_config(
    CONFIG['geometry'], runtime_params
)
# Generate sim from config
sim = build_sim.build_sim_from_config(CONFIG)
# Build sim step function (not provided in build_sim_from_config)
sim._step_fn = SimulationStepFn(
      stepper=sim._stepper,
      time_step_calculator=sim.time_step_calculator,
      transport_model=sim._transport_model,
  )

# Initialize sim
if jax.config.read('jax_enable_x64'):
  logging.info('Precision is set at float64')
else:
  logging.info('Precision is set at float32')

logging.info('Starting simulation.')
running_main_loop_start_time = time.time()
wall_clock_step_times = []

torax_outputs = [
  sim.initial_state,
]
dynamic_runtime_params_slice = sim.dynamic_runtime_params_slice_provider(
  sim.initial_state.t
)
geo = sim.geometry_provider(sim.initial_state.t)

# Populate the starting state with source profiles from the implicit sources
# before starting the run-loop. The explicit source profiles will be computed
# inside the loop and will be merged with these implicit source profiles.
sim.initial_state.core_sources = _get_initial_source_profiles(
    static_runtime_params_slice=sim.static_runtime_params_slice,
    dynamic_runtime_params_slice=dynamic_runtime_params_slice,
    geo=geo,
    core_profiles=sim.initial_state.core_profiles,
    source_models=sim.step_fn.stepper.source_models,
)

sim_state = sim.initial_state

# Macro time loop
while sim.time_step_calculator.not_done(
  sim_state.t,
  dynamic_runtime_params_slice,
  sim_state.time_step_calculator_state,
):

  sim_state, wall_clock_step_time = run_single_step(
      static_runtime_params_slice = sim.static_runtime_params_slice,
      dynamic_runtime_params_slice_provider= sim.dynamic_runtime_params_slice_provider,
      geometry_provider=sim.geometry_provider,
      sim_state=sim_state,
      step_fn=sim.step_fn,
      )
  
  # append state, append 
  torax_outputs.append(sim_state)
  wall_clock_step_times.append(wall_clock_step_time)

# Update the final time step's source profiles based on the explicit source
# profiles computed based on the final state.
logging.info("Updating last step's source profiles.")
explicit_source_profiles = source_models_lib.build_source_profiles(
    dynamic_runtime_params_slice=dynamic_runtime_params_slice,
    geo=geo,
    core_profiles=sim_state.core_profiles,
    source_models=sim.step_fn.stepper.source_models,
    explicit=True,
)

sim_state.core_sources = _merge_source_profiles(
    explicit_source_profiles=explicit_source_profiles,
    implicit_source_profiles=sim_state.core_sources,
    source_models=sim.step_fn.stepper.source_models,
    qei_core_profiles=sim_state.core_profiles,
)
# If the first step of the simulation was very long, call it out. It might
# have to do with tracing the jitted step_fn.
std_devs = 2  # Check if the first step is more than 2 std devs longer.
if wall_clock_step_times and wall_clock_step_times[0] > (
    jnp.mean(jnp.array(wall_clock_step_times))
    + std_devs * jnp.std(jnp.array(wall_clock_step_times))
):
  long_first_step = True
  logging.info(
      'The first step took more than %.1f std devs longer than other steps.'
      ' It likely was tracing and compiling the step_fn. It took %.2f '
      'seconds of wall clock time.',
      std_devs,
      wall_clock_step_times[0],
  )
else:
  long_first_step = False

wall_clock_time_elapsed = time.time() - running_main_loop_start_time
simulation_time = torax_outputs[-1].t - torax_outputs[0].t
if long_first_step:
  # Don't include the long first step in the total time logged.
  wall_clock_time_elapsed -= wall_clock_step_times[0]
logging.info(
    'Simulated %.2f seconds of physics in %.2f seconds of wall clock time.',
    simulation_time,
    wall_clock_time_elapsed,
)
