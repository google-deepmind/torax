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
  r"""Dimensionless inputs to TGLF-based models.

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


class TGLFBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for TGLF-based transport models."""

  def _prepare_tglf_inputs(
      Zeff_face: chex.Array,
      q_correction_factor: chex.Numeric,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> TGLFInputs:
    ## Shorthand 'standard' variables
    Te_keV = core_profiles.temp_el.face_value()
    Te_eV = Te_keV * 1e3
    Te_J = Te_keV * CONSTANTS.keV2J
    Ti_keV = core_profiles.temp_ion.face_value()
    ne = core_profiles.ne * core_profiles.nref
    # q must be recalculated since in the nonlinear solver psi has intermediate
    # states in the iterative solve
    q, _ = physics.calc_q_from_psi(
        geo=geo,
        psi=core_profiles.psi,
        q_correction_factor=q_correction_factor,
    )

    ## Reference values used for TGLF-specific normalisation
    # https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization
    # https://gafusion.github.io/doc/geometry.html#effective-field
    # B_unit = 1/r d(psi_tor)/dr = q/r dpsi/dr
    # Note: TGLF uses geo.rmid = (Rmax - Rmin)/2 as the radial coordinate
    # This means all gradients are calculated w.r.t. rmid
    m_D_amu = 2.014  # Mass of deuterium
    m_D = m_D_amu * CONSTANTS.mp  # Mass of deuterium
    c_s = (Te_J / m_D) ** 0.5
    a = geo.Rmin[-1]  # Minor radius at LCFS
    B_unit = q / (geo.rmid) * jnp.gradient(core_profiles.psi, geo.rmid)

    ## Dimensionless gradients, eg lref_over_lti where lref=amin, lti = -ti / (dti/dr)
    normalized_log_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.rmid,
        reference_length=a,
    )

    ## Dimensionless temperature ratio
    Ti_over_Te = Ti_keV / Te_keV

    ## Dimensionless electron-electron collision frequency = nu_ee / (c_s/a)
    # https://gafusion.github.io/doc/tglf/tglf_list.html#xnue
    # https://gafusion.github.io/doc/cgyro/cgyro_list.html#cgyro-nu-ee
    # Note: In the TGLF docs, XNUE is mislabelled as electron-ion collision frequency.
    # It is actually the electron-electron collision frequency, and is defined as in CGYRO
    # See https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html#tglf
    # Lambda_ee is computed with keV and m^-3 units
    # normalised_nu_ee is computed with SI units (ie J rather than keV)
    Lambda_ee = physics._calculate_lambda_ee(Te_keV, ne)
    normalised_nu_ee = (4 * jnp.pi * ne * CONSTANTS.qe**4 * Lambda_ee) / (
        CONSTANTS.me**0.5 * (2 * Te_J) ** 1.5
    )
    nu_ee = normalised_nu_ee / (c_s / a)

    ## Safety factor, q
    # https://gafusion.github.io/doc/tglf/tglf_list.html#q-sa
    # defined before

    ## Safety factor shear, s_hat = r/q dq/dr
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-shat-sa
    # Note: calc_s_from_psi_rmid gives rq dq/dr, so we divide by q**2
    s_hat = physics.calc_s_from_psi_rmid(geo, core_profiles.psi) / q**2

    ## Electron beta
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-betae
    # Note: Te in eV
    beta_e = 8 * jnp.pi * ne * Te_eV / B_unit**2

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
    # Note: TGLF uses the same normalisation as CGYRO
    # This has an extra c^2 factor compared to TORAX's calculate_chiGB
    chiGB = (
        quasilinear_transport_model.calculate_chiGB(
            reference_temperature=Te_keV,  # conversion to J done internally
            reference_magnetic_field=B_unit,
            reference_mass=m_D_amu,
            reference_length=a,
        )
        * CONSTANTS.c**2
    )

    return TGLFInputs(
        # From QuasilinearInputs
        chiGB=chiGB,
        Rmin=geo.Rmin,
        Rmaj=geo.Rmaj,
        lref_over_lti=normalized_log_gradients.lref_over_lti,
        lref_over_lte=normalized_log_gradients.lref_over_lte,
        lref_over_lne=normalized_log_gradients.lref_over_lne,
        lref_over_lni0=normalized_log_gradients.lref_over_lni0,
        lref_over_lni1=normalized_log_gradients.lref_over_lni1,
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
