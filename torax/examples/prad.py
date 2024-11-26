CONFIG = {
    'runtime_params': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'j_bootstrap': {},
        'generic_current_source': {},
        'generic_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        'generic_ion_el_heat_source': {},
        'fusion_heat_source': {},
        'qei_source': {},
        'ohmic_heat_source': {},
        'radiation_heat_sink': {},
    },
    'transport': {
        'transport_model': 'constant',
    },
    'stepper': {
        'stepper_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
