"""WEST Nitrogen scenario 58656. Based on HFPS benchmark structure from 57757.
"""

#cd /home/mateo/torax_workspace/torax && PYTHONPATH=/home/mateo/torax_workspace:/home/mateo/torax_workspace/torax /home/mateo/torax_workspace/torax_env/bin/run_torax --config=torax/examples/new_qlknn_west_58656_benchmark_hfps.py --output_dir='/home/mateo/my_results' --quit

import imas
import numpy as np
from scipy.interpolate import interp1d
from torax._src.imas_tools.input import core_profiles, core_sources
from torax._src.sources.source import AffectedCoreProfile

from torax import sources
from hpi2nn.src_hpi2nn.interfaces.torax_interface import HPI2NNPelletConfig
sources.register_source_model_config(HPI2NNPelletConfig, 'pellet')

# Add gaussian anomalous ad-hoc transport in the core similar to what is used in HFPS. See Theo's paper.
def gaussian_profile(x, maximum, center, width):
    return maximum * np.exp(-((x - center) ** 2) / (2 * width**2))

def pos_gaussian_profile(x, maximum, center, width):
    return maximum * np.exp(((x - center) ** 2) / (2 * width**2))


# def V_ware_jintrac_fit(rho_norm):
#     """
#     Fit analytique du pinch Ware JINTRAC pour WEST 58656.
#     Forme: Beta distribution avec a=1, b=2.5.
#     - Vaut 0 à rho=0 et rho=1
#     - Minimum V_e = -0.062 m/s à rho ≈ 0.286
#     - Erreur RMS ~ 6e-3 m/s vs courbe JINTRAC digitalisée
#     """
#     return -0.504 * rho_norm * (1.0 - rho_norm)**2.5

# def V_neo_approx(rho_norm):
#     return -0.6 * np.sqrt(rho_norm * 0.2)

# Input shot info
Path = 'imas:hdf5?path=/home/mateo/torax_workspace/58656'
t_start = 34.8

# Load IDSs
# Load all core_profiles time slices so that T_e/T_i boundary conditions
# follow the IDS evolution throughout the simulation (not frozen at t_start).
with imas.DBEntry(Path, mode="r") as entry:
    #core_profiles_ids = entry.get('core_profiles', autoconvert=False)
    core_profiles_ids = entry.get_slice('core_profiles', time_requested=t_start, interpolation_method=imas.ids_defs.CLOSEST_INTERP, autoconvert=False)
    core_sources_ids = entry.get('core_sources', autoconvert=False)
core_profiles_ids = imas.convert_ids(core_profiles_ids, "4.0.0")
core_sources_ids = imas.convert_ids(core_sources_ids, "4.0.0")

###
# core_profiles
###

profile_conditions_from_ids = core_profiles.profile_conditions_from_IMAS(
    core_profiles_ids,
)
plasma_composition_from_ids = core_profiles.plasma_composition_from_IMAS(core_profiles_ids, excluded_impurities=['Cu'])
#del profile_conditions_from_ids["Ip"]
#del profile_conditions_from_ids["psi"]

# Filter out invalid n_e profiles (NaN or unphysically low values)
correct_index = []
for i, ne_profile in enumerate(profile_conditions_from_ids["n_e"][2]):
    if not np.isnan(ne_profile).any() and np.min(ne_profile) >= 1e18:
        correct_index.append(i)
profile_conditions_from_ids["n_e"] = (
    list([profile_conditions_from_ids["n_e"][0][i] for i in correct_index]),
    list([profile_conditions_from_ids["n_e"][1][i] for i in correct_index]),
    list([profile_conditions_from_ids["n_e"][2][i] for i in correct_index]),
)

profile_conditions_from_ids["psi"] = (
    profile_conditions_from_ids["psi"][0],  # times
    profile_conditions_from_ids["psi"][1],  # rho_norm
    [-p for p in profile_conditions_from_ids["psi"][2]],  # flip sign
)

# Setting internal boundary condition at customizable rhon.
# Use slice closest to t_start for pedestal values and rho_norm grid.
times = [float(p.time) for p in core_profiles_ids.profiles_1d]
idx_start = int(np.argmin(np.abs(np.array(times) - t_start)))
rho_norm = np.asarray(core_profiles_ids.profiles_1d[idx_start].grid.rho_tor_norm)
profiles_1d = core_profiles_ids.profiles_1d[idx_start]
T_e = np.asarray(profiles_1d.electrons.temperature)  # [eV]
n_e = np.asarray(profiles_1d.electrons.density)      # [m^-3]
T_i = np.asarray(profiles_1d.t_i_average)  

def get_pedestal_values(rho_norm_grid, T_e, n_e, T_i, rho_target):
    """
    Extract T_e, n_e, T_i at a specific rho_norm radius.
    
    Args:
        rho_norm_grid: rho_norm coordinate array
        T_e: electron temperature profile
        n_e: electron density profile
        T_i: ion temperature profile (average over all species)
        rho_target: target rho_norm value (e.g., 0.93 for pedestal top)
    
    Returns:
        Dictionary with T_e_ped, n_e_ped, T_i_ped values at rho_target
    """
    # Create interpolation functions
    f_Te = interp1d(rho_norm_grid, T_e, kind='cubic', bounds_error=False, fill_value='extrapolate')
    f_ne = interp1d(rho_norm_grid, n_e, kind='cubic', bounds_error=False, fill_value='extrapolate')
    f_Ti = interp1d(rho_norm_grid, T_i, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    T_e_ped = float(f_Te(rho_target))
    n_e_ped = float(f_ne(rho_target))
    T_i_ped = float(f_Ti(rho_target))
    
    return {
        'rho_norm_ped_top': rho_target,
        'n_e_ped': n_e_ped,      # [m^-3]
        'T_e_ped': T_e_ped / 1000,   # [keV]
        'T_i_ped': T_i_ped / 1000,   # [keV]
    }

rho_ped_top = 0.85  # → BC sur cellule rho=0.82 (côté pédestal, non 0.78 core)
pedestal_values = get_pedestal_values(rho_norm, T_e, n_e, T_i, rho_ped_top)

# # Print edge temperatures for diagnostic (not used as explicit BC — left to None)
print(f"Pedestal values at rho_norm = {pedestal_values['rho_norm_ped_top']}:")
print(f"  T_e = {pedestal_values['T_e_ped']:.3f} keV")
print(f"  n_e = {pedestal_values['n_e_ped']:.3e} m^-3")
print(f"  T_i = {pedestal_values['T_i_ped']:.3f} keV")
print(f"Edge T_e (rho=1, diagnostic only): {float(T_e[-1]):.1f} eV")
print(f"Edge T_i (rho=1, diagnostic only): {float(T_i[-1]):.1f} eV")


###
# core_sources58656
###
# Available sources in IDS 58656:
#   line_radiation : T_e (line radiation, < 0)           -> impurity_radiation prescribed
#   cold_neutrals  : n_e (particle source), T_e (< 0)    -> generic_particle prescribed (commented for now)
#   ic             : Te/Ti , 5e-17 W/m^3, negligible
#   lh, ec         : present but all zeros
additional_sources = core_sources._SourceCollection()
for source in core_sources_ids.source:
    if source.identifier.name == 'line_radiation':
        # Line radiation (impurities): replaces 'radiation' from 57757
        source_profiles = core_sources._extract_source_profiles(
            source, (AffectedCoreProfile.TEMP_EL,)
        )
        additional_sources.add('impurity_radiation', source_profiles)
    elif source.identifier.name == 'cold_neutrals':
        # Particle source from cold neutrals (gas puff + edge ionisation)
        dens_profile = core_sources._extract_source_profiles(
            source, (AffectedCoreProfile.NE,)
        )
        additional_sources.add('generic_particle', dens_profile)
additional_sources = additional_sources.to_dict()

# # Diagnostic: IDS radiation profile at t_start
# # Structure: {"mode": "PRESCRIBED", "prescribed_values": ((times, rhons, profiles),)}
# rad_times_raw, rad_rhons_raw, rad_profiles_raw = additional_sources['impurity_radiation']['prescribed_values'][0]
# rad_times = [float(t) for t in rad_times_raw]
# idx_rad = int(np.argmin(np.abs(np.array(rad_times) - t_start)))
# rad_profile_at_tstart = np.asarray(rad_profiles_raw[idx_rad], dtype=float)
# print(f"IDS line_radiation at t={rad_times[idx_rad]:.2f}s:")
# print(f"  Peak (most negative): {rad_profile_at_tstart.min():.3e} W/m^3")
# print(f"  Mean over profile: {rad_profile_at_tstart.mean():.3e} W/m^3")

# Non-uniform grid
core_faces = np.linspace(0, 0.8, 51)  # 50 cells from rho=0 to rho=0.8
edge_faces = np.linspace(0.8, 1.0, 51)[1:]  # 50 cells from rho=0.8 to rho=1.0
face_centers = np.concatenate([core_faces, edge_faces])


# # Comparaison Ip : core_profiles (racine IDS) vs equilibrium
# # core_profiles: global_quantities.ip est à la racine, avec time_array séparé
# cp_times = np.array([float(p.time) for p in core_profiles_ids.profiles_1d])
# cp_ip_array = np.asarray(core_profiles_ids.global_quantities.ip)  # même time_array que profiles_1d
# idx_cp = int(np.argmin(np.abs(cp_times - t_start)))
# Ip_cp = float(cp_ip_array[idx_cp])
# t_cp  = float(cp_times[idx_cp])

# with imas.DBEntry(Path, mode="r") as entry:
#     eq = entry.get('equilibrium', autoconvert=False)
#     eq = imas.convert_ids(eq, "4.0.0")
# eq_times = np.array([float(ts.time) for ts in eq.time_slice])
# idx_eq = int(np.argmin(np.abs(eq_times - t_start)))
# Ip_eq = float(eq.time_slice[idx_eq].global_quantities.ip)
# t_eq  = float(eq.time_slice[idx_eq].time)

# print(f"Ip comparison at t_start={t_start:.1f}s:")
# print(f"  core_profiles  (t={t_cp:.3f}s) : {Ip_cp/1e6:.4f} MA  ({Ip_cp:.1f} A)")
# print(f"  equilibrium    (t={t_eq:.3f}s) : {Ip_eq/1e6:.4f} MA  ({Ip_eq:.1f} A)")
# print(f"  différence absolue : {(Ip_cp - Ip_eq)/1e3:.1f} kA")
# print(f"  note: TORAX utilise -1 * core_profiles Ip → {-Ip_cp/1e6:.4f} MA")



CONFIG = {
    "plasma_composition": {
        "main_ion": {"D": 1},
        "impurity": {
            "species": {
                'N': None,
            },
            "impurity_mode": "n_e_ratios_Z_eff",
        },
        "Z_eff": 1.5,
    },
    "profile_conditions": {**profile_conditions_from_ids,
                           #"initial_psi_mode": "geometry",
                           #"use_v_loop_lcfs_boundary_condition": True,
                                                     
    },
    "geometry": {
        "geometry_type": "imas",
        "imas_filepath": Path,
        "Ip_from_parameters": True,
        #"face_centers": face_centers,
        'n_rho': 50,
    },
    # No pedestal. Prescribe transport at the edge when using other settings.
    "pedestal": {
        "set_pedestal": True,
        "model_name": "set_T_ped_n_ped",
        "rho_norm_ped_top": pedestal_values['rho_norm_ped_top'],  # [unitless],
        "T_i_ped": pedestal_values['T_i_ped'],  # [keV], 
        "T_e_ped": pedestal_values['T_e_ped'],  # [keV], 
        "n_e_ped": pedestal_values['n_e_ped'],  # [m^-3],
    },
    "sources": {
        # Physics-based sources (MODEL_BASED)
        "ohmic": {},
        "ei_exchange": {},
        "gas_puff": {
            "mode": "MODEL_BASED",
            # Gas puff schedule from HFPS ion source data:
            "S_total": (
                (
                    [35.0,   37.72,  38.05, 39.23,   39.52],
                    [11e21,  1.8e22,  11e21,  1.5e22,  11e21],
                ),
                "STEP",
            ),
            #"S_total": 8e21,
            "puff_decay_length": 0.09,
        },
        # Self-consistent radiation via mavrin_fit (like west_config_55797).
        # IDS-prescribed radiation is fixed amplitude — unsafe for ohmic shots
        # where radiation doesn't decrease when T drops, causing runaway cooling.
        "impurity_radiation": {
            **additional_sources['impurity_radiation'],
            # "model_name": "mavrin_fit",
            #"radiation_multiplier": 1.0,
        },
        # IC heating: 5e-17 W/m^3 in IDS, negligible, not included
        # LH, EC: present in IDS but all zeros, not included
        # Particle source from cold_neutrals: uncomment to enable when evolve_density=True
        "generic_particle": additional_sources['generic_particle'],
        
        "pellet": {
        "model_name": "hpi2_nn",
        "injection_line": "WEST_upHFS",
        "pellet_radii": [0.00080532, 0.0008634, 0.00083763],
        "pellet_velocities": [89.0, 58.0, 58.0],
        "injection_point_1": (1.8, 0.47),
        "injection_point_2": (2.6192, -0.136),
        "trigger_times": [36.186972564753056, 37.66997256476014, 39.17497256476732],
        "ablation_time": 1e-3,
        'use_model_ablation_time': True,
        "is_explicit": True,
        },
    },
    "transport": {
        'model_name': 'combined',
        'transport_models': [
            {
                # Inner patch: transport constant en dessous de rho=0.2
                'model_name': 'constant',
                'rho_max': 0.15,
                "chi_i": (rho_norm, gaussian_profile(rho_norm, 0.3, 0.0, 0.15)),
                "chi_e": (rho_norm, gaussian_profile(rho_norm, 0.3, 0.0, 0.1)),
                "D_e": (rho_norm, pos_gaussian_profile(rho_norm, 0.1, 0.0, 0.15)),
                # 'chi_i': 0.4,
                # 'chi_e': 0.6,
                # 'D_e': 0.1,
                "V_e": -0.1,
            },
            {
            'model_name': 'qlknn',
            'clip_inputs': False,
            'chi_min': 0.05,
            'chi_max': 3.0,   
            'D_e_min': 0.01,
            'D_e_max': 4.0,  
            'V_e_min': -4.0,
            'V_e_max': 1.0,
            'DV_effective': False,
            'include_ITG': True,
            'include_TEM': True,
            'include_ETG': True,
            'avoid_big_negative_s': True,
            'smag_alpha_correction': True,
            'An_min': 0.05,
            'ITG_flux_ratio_correction': 0.5,
            "ETG_correction_factor": 1/3,  # from west_config_55797
            #'rho_min': 0.15,
            "rho_max": 0.85,
            },

        ],
        # Transport prescrit pour la région pedestal/bord (rho > rho_ped_top = 0.85)
        'pedestal_transport_models': [
            {
                'model_name': 'constant',
                'chi_i': 1.0,
                'chi_e': 1.0,
                'D_e' : 2.0,
                'V_e' : -2.0, 
            }
        ],
        'smoothing_width': 0.1,
        'smooth_everywhere': False,
        'chi_min': 0.05,
        'chi_max': 4.0,
        'D_e_min': 0.05,
    },
    

    "mhd": {
        "sawtooth": {
            "trigger_model": {
                "model_name": "simple",
                "s_critical": 0.1,        # cisaillement magnétique critique à q=1 (défaut: 0.1)
                "minimum_radius": 0.05,   # rho_norm minimum de la surface q=1 (défaut: 0.05)
                "suppression_times": [36.186972564753056, 37.66997256476014, 39.17497256476732],  # [s] après lesquels les crashs sont supprimés
                "suppression_duration": 0.01,  # [s]
        },
            "redistribution_model": {
                "model_name": "simple",
                "mixing_radius_multiplier": 1.1,# r_mix = multiplier × rho_q1 (défaut: 1.1)
                "flattening_factor": 1.001, #default 1.01
        },
            "crash_step_duration": 1e-5,  # durée du crash [s] (défaut: 1e-5)
        }
    },

    "neoclassical": {
        "bootstrap_current": {"model_name": "sauter"},
        "transport": {
            "model_name": "angioni_sauter",
            "use_shaing_ion_correction": True,  # Shaing désactivé (near-axis instability)
            "shaing_ion_multiplier": 0.05,
            "chi_min": 0.0,
            "chi_max": 5.0,    # limité à la même amplitude que le transport turbulent
            "D_e_min": 0.0,
            "D_e_max": 2.0,    # limité pour éviter collapse n_e
            "V_e_min": -2.0,   # Ware pinch clipé pour éviter collapse n_e près de l'axe
            "V_e_max": 2.0,
        },
    },
    "numerics": {
        "t_initial": t_start,
        "t_final": t_start + 5.5,
        "fixed_dt": 0.01,
        "min_dt": 1e-4,  # was 1e-3
        "dt_reduction_factor": 3.0,
        "resistivity_multiplier": 1.0,
        "adaptive_T_source_prefactor": 1e8,
        "evolve_current": True,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_density": True,
        "min_dt": 1e-5
    },
    "solver": {
        #"solver_type": "newton_raphson",
        "solver_type": "linear",
        "use_predictor_corrector": True,
        "n_corrector_steps": 10,
        "use_pereverzev": True,
        "chi_pereverzev": 2.0,
        "D_pereverzev": 0.0,
        "log_iterations": False,
    },
    "time_step_calculator": {
        "calculator_type": "pellet_aware",
        "base_calculator_type": "fixed",
        "trigger_tolerance": 1e-8,
        # "window_after_pellet": 0.5,
        # "dt_after_pellet": 0.01,
    },
}
