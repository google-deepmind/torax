import chex
import jax
from jax import numpy as jnp
from tglfnn import TGLFInputs

from torax import geometry
from torax import physics
from torax import state
from torax.constants import CONSTANTS
from torax.transport_model import runtime_params as runtime_params_lib


def prepare_tglfnn_inputs(
    Zeff_face: chex.Array,
    transport,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    q_correction_factor: chex.Numeric,
) -> TGLFInputs:
    Te = core_profiles.temp_el
    Ti = core_profiles.temp_ion

    # Calculate q
    # Need to recalculate since in the nonlinear solver psi has intermediate
    # states in the iterative solve
    # Copied from QLKNN - is this correct?
    q, _ = physics.calc_q_from_psi(
        geo=geo,
        psi=core_profiles.psi,
        q_correction_factor=q_correction_factor,
    )

    # What are these variables?
    a = None
    dRmaj = None
    s_hat = None
    ei_collision_freq = None
    kappa = None
    kappa_shear = None
    delta = None
    delta_shear = None
    beta_e = None


    # Define radial coordinate as midplane average r
    # Copied from QLKNN - is this correct?
    rmid = (geo.Rout - geo.Rin) * 0.5

    return TGLFInputs(
        Te_grad_norm=a / Te.face_value() * Te.face_grad(rmid),
        Ti_grad_norm=a / Ti.face_value() * Ti.face_grad(rmid),
        Ti_over_Te=Ti.face_value() / Te.face_value(),
        Rmin=geo.Rmin,
        dRmaj=dRmaj,
        q=q,
        s_hat=s_hat,
        ei_collision_freq=ei_collision_freq,
        kappa=kappa,
        kappa_shear=kappa_shear,
        delta=delta,
        delta_shear=delta_shear,
        beta_e=beta_e,
        Zeff=Zeff_face,
    )
