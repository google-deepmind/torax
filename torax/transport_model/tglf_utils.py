import chex
from jax import numpy as jnp

from torax import geometry
from torax import physics
from torax import state
from torax.constants import CONSTANTS
from torax.quasilinear_utils import QuasilinearInputs


@chex.dataclass(frozen=True)
class TGLFInputs(QuasilinearInputs):
    r"""Dimensionless inputs to the TGLF model.

    See https://gafusion.github.io/doc/tglf/tglf_table.html for definitions.
    """
    # Ti/Te
    Ti_over_Te: chex.Array
    # dRmaj/dr
    dRmaj: chex.Array
    # q
    q: chex.Array
    # r/q dq/dr
    s_hat: chex.Array
    # nu_ei (see note in prepare_tglf_inputs)
    ei_collision_freq: chex.Array
    # Elongation kappa
    kappa: chex.Array
    # r/kappa dkappa/dr
    kappa_shear: chex.Array
    # Triangularity delta
    delta: chex.Array
    # r ddelta/dr
    delta_shear: chex.Array
    # Electron pressure defined w.r.t B_unit
    beta_e: chex.Array
    # Effective charge
    Zeff: chex.Array


def prepare_tglf_inputs(
    Zeff_face: chex.Array,
    transport,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    q_correction_factor: chex.Numeric,
    # TODO: Add Kappa or Zmax, Zmin to geo, rather than passing it in explicitly
    kappa: chex.Numeric,
    nref: chex.Numeric,
) -> TGLFInputs:
    # Shorthand for the appropriate variables
    Te = core_profiles.temp_el
    Ti = core_profiles.temp_ion
    ne = core_profiles.ne

    # Reference velocity and length, used for normalisation
    vref = (Te.face_value() / (core_profiles.Ai * CONSTANTS.mp)) ** 0.5
    lref = geo.Rmin[-1] # Minor radius at LCFS

    # Temperature gradients
    Ti_over_Te = Ti.face_value() / Te.face_value()
    Ate = -lref / Te.face_value() * Te.face_grad()
    Ati = -lref / Ti.face_value() * Ti.face_grad()

    # Density gradient
    # Note: nref cancels, as 1/(ne*nref) * (ne_grad * nref) = 1/ne * ne_grad
    Ane = -lref / ne.face_value() * core_profiles.ne.face_grad()

    # Electron-electron collision frequency
    # Note: In the docs, TGLF XNUE is mislabelled.
    # It is actually the electron-electron collision frequency
    # See https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html
    # Coulomb_logarithm_ee given by Wesson 3rd ed p727
    # ne in m^-3, Te in keV
    # TODO: Check definition of nu_ee
    # TODO: These should be in physics.py
    coulomb_logarithm_ee = 14.9 - 0.5 * jnp.log(ne / 1e20) + jnp.log(Te)
    normalised_nu_ee = (4 * jnp.pi * ne * CONSTANTS.qe**4 * coulomb_logarithm_ee) / (
        CONSTANTS.me**0.5 * (2 * Te) ** 1.5
    )
    nu_ee = normalised_nu_ee / (vref / lref)

    # Safety factor
    # Need to recalculate since in the nonlinear solver psi has intermediate
    # states in the iterative solve
    q, _ = physics.calc_q_from_psi(
        geo=geo,
        psi=core_profiles.psi,
        q_correction_factor=q_correction_factor,
    )
    # Shear uses rho_face_norm
    s_hat = physics.calc_s_from_psi(geo, core_profiles.psi) # = r/q dq/dr

    # Electron beta
    p_e = ne * (Te * 1e3)  # ne in m^-3, Te in eV
    # B_unit = q/r dpsi/dr
    B_unit = q / geo.rho_face_norm * jnp.gradient(core_profiles.psi, geo.rho_face_norm)
    beta_e = 8 * jnp.pi * p_e / B_unit**2

    # Geometry
    Rmaj = geo.Rmaj
    Rmin = geo.Rmin
    dRmaj = jnp.gradient(geo.Rmaj, geo.rho_face_norm)
    # Elongation
    kappa_shear = geo.rho_face_norm / kappa * jnp.gradient(kappa, geo.rho_face_norm)
    # Triangularity
    delta = geo.delta_face
    delta_shear = geo.delta_face * jnp.gradient(geo.delta_face, geo.rho_face_norm)

    # Gyrobohm diffusivity
    # Used to unnormalise the outputs
    # TODO: check this definition with Lorenzo/TGLF and ensure correct normalisation
    chiGB = (
      (core_profiles.Ai * CONSTANTS.mp) ** 0.5
      / (CONSTANTS.qe * geo.B0) ** 2
      * (Ti.face_value() * CONSTANTS.keV2J) ** 1.5
      / lref
    )

    return TGLFInputs(
        # From QuasilinearInputs
        chiGB=chiGB,
        Rmin=Rmin,
        Rmaj=Rmaj,
        Ati=Ati,
        Ate=Ate,
        Ane=Ane,
        # From TGLFInputs
        Ti_over_Te=Ti_over_Te,
        dRmaj=dRmaj,
        q=q,
        s_hat=s_hat,
        nu_ee=nu_ee,
        kappa=kappa,
        kappa_shear=kappa_shear,
        delta=delta,
        delta_shear=delta_shear,
        beta_e=beta_e,
        Zeff=Zeff_face,
    )
