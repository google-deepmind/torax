"""Tests setting an externally-prescribed, time-evolving particle source.

Notable settings:
- Constant transport coefficient model
- Pedestal
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': True,
            'nbar': 0.85,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            't_final': 5,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_ion_el_heat_source': {},
        'qei_source': {},
        'gas_puff_source': {
            'S_puff_tot': 1.0e22,
        },
        'pellet_source': {
            'S_pellet_tot': {0: 5.0e22, 2.5: 5.0e22, 2.6: 0.5e22},
        },
        'j_bootstrap': {},
        'jext': {},
    },
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            'De_const': 0.5,
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
