# Copyright 2024 DeepMind Technologies Limited
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
"""Base class and utils for TGLF-based models."""

import chex
from jax import numpy as jnp

from torax import geometry
from torax import physics
from torax import state
from torax.constants import CONSTANTS
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


@chex.dataclass
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
    pass


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(quasilinear_transport_model.DynamicRuntimeParams):
    pass


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    pass


@chex.dataclass(frozen=True)
class TGLFInputs(quasilinear_transport_model.QuasilinearInputs):
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


class TGLFBasedTransportModel(quasilinear_transport_model.QuasilinearTransportModel):
    """Base class for TGLF-based transport models."""

    def _prepare_tglf_inputs(
        Zeff_face: chex.Array,
        nref: chex.Numeric,
        q_correction_factor: chex.Numeric,
        transport: DynamicRuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> TGLFInputs:
        # Shorthand for the appropriate variables
        Te = core_profiles.temp_el
        Ti = core_profiles.temp_ion
        ne = core_profiles.ne

        # Reference velocity and length, used for normalisation
        vref = (Te.face_value() / (core_profiles.Ai * CONSTANTS.mp)) ** 0.5
        lref = geo.Rmin[-1]  # Minor radius at LCFS

        # Temperature gradients
        Ti_over_Te = Ti.face_value() / Te.face_value()
        Ate = -lref / Te.face_value() * Te.face_grad()
        Ati = -lref / Ti.face_value() * Ti.face_grad()

        # Density gradient
        # Note: nref cancels, as 1/(ne*nref) * (ne_grad * nref) = 1/ne * ne_grad
        Ane = -lref / ne.face_value() * core_profiles.ne.face_grad()

        # Electron-electron collision frequency
        # Note: In the TGLF docs, XNUE is mislabelled.
        # It is actually the electron-electron collision frequency
        # See https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html
        Lambda_ee = physics._calculate_lambda_ee(Te, ne)
        normalised_nu_ee = (4 * jnp.pi * ne * CONSTANTS.qe**4 * Lambda_ee) / (
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
        # TODO: check whether this should be midplane R
        s_hat = physics.calc_s_from_psi(geo, core_profiles.psi)  # = r/q dq/dr

        # Electron beta
        p_e = ne * (Te * 1e3)  # ne in m^-3, Te in eV
        # B_unit = q/r dpsi/dr
        B_unit = (
            q / geo.rho_face_norm * jnp.gradient(core_profiles.psi, geo.rho_face_norm)
        )
        beta_e = 8 * jnp.pi * p_e / B_unit**2

        # Geometry
        Rmaj = geo.Rmaj
        Rmin = geo.Rmin
        dRmaj = jnp.gradient(geo.Rmaj, geo.rho_face_norm)
        kappa = geo.elongation_face
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
