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
    """Shared parameters for TGLF-based models."""

    def make_provider(
        self, torax_mesh: geometry.Grid1D | None = None
    ) -> "RuntimeParamsProvider":
        return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(quasilinear_transport_model.DynamicRuntimeParams):
    """Shared parameters for TGLF-based models."""

    pass


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    """Provides a RuntimeParams to use during time t of the sim."""

    runtime_params_config: RuntimeParams

    def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
        return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


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
    # nu_ee (see note in prepare_tglf_inputs)
    nu_ee: chex.Array
    # Elongation, kappa
    kappa: chex.Array
    # Shear in elongation, r/kappa dkappa/dr
    kappa_shear: chex.Array
    # Triangularity, delta
    delta: chex.Array
    # Shear in triangularity, r ddelta/dr
    delta_shear: chex.Array
    # Electron pressure defined w.r.t B_unit
    beta_e: chex.Array
    # Effective charge
    Zeff: chex.Array


class TGLFBasedTransportModel(quasilinear_transport_model.QuasilinearTransportModel):
    """Base class for TGLF-based transport models."""

    def _prepare_tglf_inputs(
        Zeff_face: chex.Array,
        q_correction_factor: chex.Numeric,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> TGLFInputs:
        # Note: TGLF uses geo.rmid = (Rmax - Rmin)/2 as the radial coordinate
        # This means all gradients are calculated w.r.t. rmid

        ## Shorthand for commonly used variables
        Te = core_profiles.temp_el
        Ti = core_profiles.temp_ion
        ne = core_profiles.ne

        ## Reference velocity and length, used for normalisation
        # https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization
        vref = (Te.face_value() / (core_profiles.Ai * CONSTANTS.mp)) ** 0.5
        lref = geo.Rmin[-1]  # Minor radius at LCFS

        ## Temperature gradients, At = -1/T * dT/drho
        # Note: RLTS = a_minor / T * dT/dr = 1 / T * dT/drho
        # https://gafusion.github.io/doc/tglf/tglf_table.html#id2
        Ti_over_Te = Ti.face_value() / Te.face_value()
        Ate = -1 / Te.face_value() * Te.face_grad(geo.rmid)
        Ati = -1 / Ti.face_value() * Ti.face_grad(geo.rmid)

        # Density gradient, Ane = -1/ne * dne/drho
        # Note: RLNS = a_minor / ne * dne/dr = 1 / ne * dne/drho
        # https://gafusion.github.io/doc/tglf/tglf_table.html#id2
        # Note: nref cancels, as 1/(ne*nref) * (ne_grad * nref) = 1/ne * ne_grad
        Ane = -1 / ne.face_value() * core_profiles.ne.face_grad(geo.rmid)

        ## Electron-electron collision frequency = nu_ee / (vref/lref)
        # https://gafusion.github.io/doc/tglf/tglf_list.html#xnue
        # https://gafusion.github.io/doc/cgyro/cgyro_list.html#cgyro-nu-ee
        # Note: In the TGLF docs, XNUE is mislabelled as electron-ion collision frequency.
        # It is actually the electron-electron collision frequency, and is defined as in CGYRO
        # See https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html#tglf
        Lambda_ee = physics._calculate_lambda_ee(Te, ne)
        normalised_nu_ee = (4 * jnp.pi * ne * CONSTANTS.qe**4 * Lambda_ee) / (
            CONSTANTS.me**0.5 * (2 * Te) ** 1.5
        )
        nu_ee = normalised_nu_ee / (vref / lref)

        ## Safety factor, q
        # https://gafusion.github.io/doc/tglf/tglf_list.html#q-sa
        # Need to recalculate since in the nonlinear solver psi has intermediate
        # states in the iterative solve
        q, _ = physics.calc_q_from_psi(
            geo=geo,
            psi=core_profiles.psi,
            q_correction_factor=q_correction_factor,
        )

        ## Safety factor shear, s_hat = r/q dq/dr
        # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-shat-sa
        # calc_s_from_psi_rmid gives rq dq/dr
        s_hat = physics.calc_s_from_psi_rmid(geo, core_profiles.psi) / q**2

        ## Electron beta
        # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-betae
        p_e = ne * (Te * 1e3)  # ne in m^-3, Te in eV
        # B_unit = q/r dpsi/dr
        B_unit = (
            q / geo.rmid * jnp.gradient(core_profiles.psi, geo.rmid)
        )
        beta_e = 8 * jnp.pi * p_e / B_unit**2

        ## Major radius shear = dRmaj/dr
        # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-drmajdx-loc
        dRmaj = jnp.gradient(geo.Rmaj, geo.rmid)

        ## Elongation shear = r/kappa dkappa/dr
        # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-s-kappa-loc
        kappa = geo.elongation_face
        kappa_shear = geo.rmid_face / kappa * jnp.gradient(kappa, geo.rmid_face)

        ## Triangularity shear = r ddelta/dr
        # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-s-delta-loc
        delta = geo.delta_face
        delta_shear = geo.rmid_face * jnp.gradient(delta, geo.rmid_face)

        ## Gyrobohm diffusivity
        # https://gafusion.github.io/doc/tglf/tglf_table.html#id7
        # https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization
        # Note: TGLF uses the same normalisation as CGYRO, ie
        # chiGB = ne * vref * T_e * (rho_s_unit / lref)**2
        # where rho_s_unit = vref / (e*B_unit/ m_D / c)
        # TODO: Check if this code implementation is correct/equivalent to the above
        chiGB = (
            (core_profiles.Ai * CONSTANTS.mp) ** 0.5
            / (CONSTANTS.qe * geo.B0) ** 2
            * (Ti.face_value() * CONSTANTS.keV2J) ** 1.5
            / lref
        )

        return TGLFInputs(
            # From QuasilinearInputs
            chiGB=chiGB,
            Rmin=geo.Rmin,
            Rmaj=geo.Rmaj,
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
