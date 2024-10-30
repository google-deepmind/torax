import chex
from jax import numpy as jnp

from torax import geometry
from torax import physics
from torax import state
from torax.constants import CONSTANTS


@chex.dataclass(frozen=True)
class TGLFInputs:
    r"""Dimensionless inputs to the TGLF model.

    Attributes:
    -----------
      Te_grad: chex.Array
        Normalized electron temperature gradient: :math:`-{\frac {a}{T_{e}}}{\frac {dT_{e}}{dr}}`

      Ti_grad: chex.Array
        Normalized ion temperature gradient: :math:`-{\frac {a}{T_{i}}}{\frac {dT_{i}}{dr}}`

      Ti_over_Te: chex.Array
        Temperature ratio: :math:`{\frac {T_{i}}{T_{e}}}`

      rmin: chex.Array
        Flux surface centroid minor radius: :math:`\frac{r}{a}`

      dRmaj: chex.Array
        :math:`{\frac {\partial R_{maj}}{\partial x}}`

      q: chex.Array
        Safety factor, :math:`q`

      s_hat: chex.Array
        s_hat = r/q * dq/dr

      nu_ee: chex.Array
        Electron-electron collision frequency

      kappa: chex.Array
        Elongation of flux surface

      kappa_shear: chex.Array
        Shear in elongation: :math:`{\frac {r}{\kappa }}{\frac {\partial \kappa }{\partial r}}`

      delta: chex.Array
        Triangularity of flux surface

      delta_shear: chex.Array
        Shear in triangularity of flux surface: :math:`r{\frac {\partial \delta }{\partial r}}`

      beta_e: chex.Array
        :math:`\beta_e:math:` defined w.r.t :math:`B_\mathrm{unit}`

      Zeff: chex.Array
        Effective ion charge
    """
    Te_grad_norm: chex.Array
    Ti_grad_norm: chex.Array
    Ti_over_Te: chex.Array
    Rmin: chex.Array
    dRmaj: chex.Array
    q: chex.Array
    s_hat: chex.Array
    ei_collision_freq: chex.Array
    kappa: chex.Array
    kappa_shear: chex.Array
    delta: chex.Array
    delta_shear: chex.Array
    beta_e: chex.Array
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
    # Temperatures
    Te = core_profiles.temp_el
    Ti = core_profiles.temp_ion
    Ti_over_Te = Ti.face_value() / Te.face_value()

    # Reference velocity and length, used for normalisation
    vref = (Te.face_value() / (core_profiles.Ai * CONSTANTS.mp)) ** 0.5
    lref = geo.Rmin[-1] # Minor radius at LCFS

    # Temperature gradients
    normalised_dTe_drho = lref / Te.face_value() * Te.face_grad()
    normalised_dTi_drho = lref / Ti.face_value() * Ti.face_grad()

    # Density
    ne = core_profiles.ne.face_value() * nref

    # Electron-electron collision frequency
    # Note: In the docs, TGLF XNUE is mislabelled.
    # It is actually the electron-electron collision frequency
    # See https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html
    # Coulomb_logarithm_ee given by Wesson 3rd ed p727
    # ne in m^-3, Te in keV
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
    dRmaj = jnp.gradient(geo.Rmaj, geo.rho_face_norm)
    # Elongation
    kappa_shear = geo.rho_face_norm / kappa * jnp.gradient(kappa, geo.rho_face_norm)
    # Triangularity
    delta_shear = geo.delta_face * jnp.gradient(geo.delta_face, geo.rho_face_norm)

    return TGLFInputs(
        Te_grad_norm=normalised_dTe_drho,
        Ti_grad_norm=normalised_dTi_drho,
        Ti_over_Te=Ti_over_Te,
        Rmin=geo.Rmin,
        dRmaj=dRmaj,
        q=q,
        s_hat=s_hat,
        nu_ee=nu_ee,
        kappa=kappa,
        kappa_shear=kappa_shear,
        delta=geo.delta_face,
        delta_shear=delta_shear,
        beta_e=beta_e,
        Zeff=Zeff_face,
    )
