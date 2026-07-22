from pathlib import Path
import os
import numpy as np

RHO = np.array([0.015625, 0.046875, 0.078125, 0.109375, 0.140625, 0.171875, 0.203125, 0.234375, 0.265625, 0.296875, 0.328125, 0.359375, 0.390625, 0.421875, 0.453125, 0.484375, 0.515625, 0.546875, 0.578125, 0.609375, 0.640625, 0.671875, 0.703125, 0.734375, 0.765625, 0.796875, 0.828125, 0.859375, 0.890625, 0.921875, 0.953125, 0.984375], dtype=float)
TE_INIT_KEV = np.array([8.284316609233999, 8.262347073756615, 8.218459633434504, 8.15275785433791, 8.06539785421615, 7.956589244135078, 7.826596417036087, 7.675740213580269, 7.504400005959192, 7.313016252831933, 7.102093594001612, 6.872204572988716, 6.623994100844069, 6.358184807572755, 6.0755834715982555, 5.777088777486898, 5.463700734766083, 5.136532207032975, 4.7968231679285145, 4.445958546979978, 4.085490900747514, 3.7171697245913204, 3.3429801547960163, 2.9651953781541165, 2.5864498234034983, 2.209845347169251, 1.8391129104350301, 1.4788747861618219, 1.1351081553385514, 0.816075308109732, 0.5346253772634216, 0.3175523886779572], dtype=float)
TI_INIT_KEV = np.array([8.284316609233999, 8.262347073756615, 8.218459633434504, 8.15275785433791, 8.06539785421615, 7.956589244135078, 7.826596417036087, 7.675740213580269, 7.504400005959192, 7.313016252831933, 7.102093594001612, 6.872204572988716, 6.623994100844069, 6.358184807572755, 6.0755834715982555, 5.777088777486898, 5.463700734766083, 5.136532207032975, 4.7968231679285145, 4.445958546979978, 4.085490900747514, 3.7171697245913204, 3.3429801547960163, 2.9651953781541165, 2.5864498234034983, 2.209845347169251, 1.8391129104350301, 1.4788747861618219, 1.1351081553385514, 0.816075308109732, 0.5346253772634216, 0.3175523886779572], dtype=float)
NE_INIT_M3 = np.array([7.827728502565215e+19, 7.816982315719983e+19, 7.795476747889987e+19, 7.763188049624066e+19, 7.720073999675713e+19, 7.66608709914261e+19, 7.601156099671274e+19, 7.525203595643113e+19, 7.438131950425578e+19, 7.339824175981475e+19, 7.230146571696865e+19, 7.10894476613921e+19, 6.9760406384248095e+19, 6.831227480367641e+19, 6.674267797456106e+19, 6.504892429243725e+19, 6.322788234818906e+19, 6.12759105593053e+19, 5.918881318941439e+19, 5.696166882447039e+19, 5.458863685870656e+19, 5.20627463884028e+19, 4.937558834862988e+19, 4.651679654376112e+19, 4.347333956198413e+19, 4.0228364089068356e+19, 3.6759365321016476e+19, 3.303482245848341e+19, 2.9007443307335254e+19, 2.4598753523674382e+19, 1.9652587271984513e+19, 1.365437201095485e+19], dtype=float)

CONFIG = {
    "plasma_composition": {
        "main_ion": {"D": 0.5, "T": 0.5},
        "Z_eff": 1.0,
    },
    "profile_conditions": {
        "Ip": 15000000,
        "use_v_loop_lcfs_boundary_condition": False,

        "T_e": (RHO, TE_INIT_KEV),
        "T_i": (RHO, TI_INIT_KEV),
        "T_e_right_bc": 0.25,
        "T_i_right_bc": 0.25,

        "n_e": (RHO, NE_INIT_M3),
        "n_e_right_bc": 9.54929658551e+18,
        "n_e_right_bc_is_fGW": False,
        "normalize_n_e_to_nbar": False,
        "n_e_nbar_is_fGW": False,
        # Physical line-averaged density corresponding to f_GW=0.4 for Ip=15 MA, a=2 m.
        # normalize_n_e_to_nbar=False, so this is mainly bookkeeping.
        "nbar": 4.77464829276e19,
        "initial_psi_from_j": True,
        "initial_j_is_total_current": True,
        "current_profile_nu": 2.0,
    },
    "numerics": {
        "t_initial": 0.0,
        "t_final": 5.0,
        "fixed_dt": 1.0e-3,
        "min_dt": 1.0e-6,
        "dt_reduction_factor": 3.0,
        "adaptive_dt": False,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_current": True,
        "evolve_density": True,
        "resistivity_multiplier": 100.0,
    },
    "geometry": {
        "geometry_type": "circular",
        "R_major": 6.2,
        "a_minor": 2,
        "B_0": 5.3,
        "elongation_LCFS": 1.0,
        "n_rho": 32,
        "hires_factor": 4,
    },
    "sources": {
        "ohmic": {},
        "fusion": {},
        "ei_exchange": {"Qei_multiplier": 1.0},
        "bremsstrahlung": {"use_relativistic_correction": False},
        "cyclotron_radiation": {"wall_reflection_coeff": 0.8},
        "impurity_radiation": {"mode": "ZERO"},
        "generic_heat": {
            "gaussian_location": 0.2,
            "gaussian_width": 0.3,
            "P_total": 20000000,
            # TokaGrad's benchmark uses the default auxiliary partition model;
            # 0.6 is the default fixed electron fraction and is a practical
            # TORAX approximation for this benchmark.
            "electron_heat_fraction": 0.6,
        },
        "generic_current": {
            "fraction_of_total_current": 0.135122264671,
            "gaussian_location": 0.2,
            "gaussian_width": 0.3,
        },
        "generic_particle": {"S_total": 0.0},
        "gas_puff": {"S_total": 0.0},
        "pellet": {"S_total": 0.0},
    },
    "transport": {
        "model_name": "bohm-gyrobohm",
        "chi_e_bohm_multiplier": 1.0,
        "chi_i_bohm_multiplier": 1.0,
        "chi_e_gyrobohm_multiplier": 1.0,
        "chi_i_gyrobohm_multiplier": 1.0,
        "chi_e_bohm_coeff": 8e-5,
        "chi_e_gyrobohm_coeff": 5e-6,
        "chi_i_bohm_coeff": 8e-5,
        "chi_i_gyrobohm_coeff": 5e-6,
        "D_face_c1": 1.0,
        "D_face_c2": 0.3,
        "V_face_coeff": 0.0,
        "chi_min": 0.05,
        "chi_max": 100.0,
        "D_e_min": 0.05,
        "D_e_max": 50.0,
        "V_e_min": -10.0,
        "V_e_max": 10.0,
        "smooth_everywhere": False,
        "smoothing_width": 0.05,
    },
    "solver": {
        "solver_type": "linear",
        "use_predictor_corrector": True,
        "n_corrector_steps": 1,
        "use_pereverzev": False,
    },
    "time_step_calculator": {"calculator_type": "fixed"},
    "pedestal": {"model_name": "no_pedestal", "set_pedestal": False},
}
