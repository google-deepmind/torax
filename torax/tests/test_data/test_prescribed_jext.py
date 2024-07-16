"""Tests driving a prescribed, time-varying external current source.

Constant transport coefficient model, circular geometry.
"""
import jax.numpy as jnp

# Define the jext profile
def gaussian(r, center, width, amplitude):
    return amplitude * jnp.exp(-((r - center) ** 2) / (2 * width ** 2))

r = jnp.linspace(0, 1, 32)
jext_profile_0 = gaussian(r, center=0.35, width=0.05, amplitude=1e6)
jext_profile_1 = gaussian(r, center=0.15, width=0.1, amplitude=1e6)


# Create the config
CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': False,
            'nbar': 0.85,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            # 'nu': 0,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 5,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Only drive the external current source
        'jext': {
          'mode': 'prescribed',
          'prescribed_values': {
              0: {r_i.item(): jext_i.item() for r_i, jext_i in zip(r, jext_profile_0)},
              2.5: {r_i.item(): jext_i.item() for r_i, jext_i in zip(r, jext_profile_1)},
          },
          # Disable the formula-based jext term by setting its width to 0
          'wext': 0.0,
        },
        # Disable density sources/sinks
        'nbi_particle_source': {
            'S_nbi_tot': 0.0,
        },
        'gas_puff_source': {
            'S_puff_tot': 0.0,
        },
        'pellet_source': {
            'S_pellet_tot': 0.0,
        },
        # Use default heat sources
        'generic_ion_el_heat_source': {},
        'qei_source': {},
    },
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            # diffusion coefficient in electron density equation in m^2/s
            'De_const': 0.5,
            # convection coefficient in electron density equation in m^2/s
            'Ve_const': -0.2,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
