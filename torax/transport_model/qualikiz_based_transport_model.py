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
"""Base class and utils for Qualikiz-based models."""

import chex
from jax import numpy as jnp
from torax import constants as constants_module
from torax import geometry
from torax import physics
from torax import state
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


@chex.dataclass
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  """Shared parameters for Qualikiz-based models."""
  # Collisionality multiplier.
  coll_mult: float = 1.0
  # ensure that smag - alpha > -0.2 always, to compensate for no slab modes
  avoid_big_negative_s: bool = True
  # reduce magnetic shear by 0.5*alpha to capture main impact of alpha
  smag_alpha_correction: bool = True
  # if q < 1, modify input q and smag as if q~1 as if there are sawteeth
  q_sawtooth_proxy: bool = True

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(quasilinear_transport_model.DynamicRuntimeParams):
  """Shared parameters for Qualikiz-based models."""
  coll_mult: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class QualikizInputs(quasilinear_transport_model.QuasilinearInputs):
  """Inputs to Qualikiz-based models."""

  Zeff_face: chex.Array
  Ani0: chex.Array
  Ani1: chex.Array
  q: chex.Array
  smag: chex.Array
  x: chex.Array
  Ti_Te: chex.Array
  log_nu_star_face: chex.Array
  normni: chex.Array
  alpha: chex.Array
  epsilon_lcfs: chex.Array


class QualikizBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for Qualikiz-based transport models."""

  def _prepare_qualikiz_inputs(
      self,
      Zeff_face: chex.Array,
      nref: chex.Numeric,
      q_correction_factor: chex.Numeric,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> QualikizInputs:
    """Prepare Qualikiz inputs."""
    constants = constants_module.CONSTANTS

    Rmin = geo.Rmin
    Rmaj = geo.Rmaj

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5
    rmid_face = (geo.Rout_face - geo.Rin_face) * 0.5

    temp_ion_var = core_profiles.temp_ion
    temp_ion_face = temp_ion_var.face_value()
    temp_ion_face_grad = temp_ion_var.face_grad(rmid)
    temp_el_var = core_profiles.temp_el
    temp_electron_face = temp_el_var.face_value()
    temp_electron_face_grad = temp_el_var.face_grad(rmid)
    # Careful, these are in n_ref units, not postprocessed to SI units yet
    raw_ne = core_profiles.ne
    raw_ne_face = raw_ne.face_value()
    raw_ne_face_grad = raw_ne.face_grad(rmid)
    raw_ni = core_profiles.ni
    raw_ni_face = raw_ni.face_value()
    raw_ni_face_grad = raw_ni.face_grad(rmid)
    raw_nimp = core_profiles.nimp
    raw_nimp_face = raw_nimp.face_value()
    raw_nimp_face_grad = raw_nimp.face_grad(rmid)

    # True SI value versions
    true_ne_face = raw_ne_face * nref
    true_ni_face = raw_ni_face * nref
    true_nimp_face = raw_nimp_face * nref

    # gyrobohm diffusivity
    # (defined here with Lref=Rmin due to QLKNN training set normalization)
    chiGB = (
        (core_profiles.Ai * constants.mp) ** 0.5
        / (constants.qe * geo.B0) ** 2
        * (temp_ion_face * constants.keV2J) ** 1.5
        / Rmin
    )

    # transport coefficients from the qlknn-hyper-10D model
    # (K.L. van de Plassche PoP 2020)

    # TODO(b/335581689): make a unit test that tests this function directly
    # with set_pedestal = False. Currently this is tested only via
    # sim test7, which has set_pedestal=True. With set_pedestal=True,
    # mutants of Ati[-1], Ate[-1], An[-1] all affect only chi[-1], but
    # chi[-1] remains above config.transport.chimin for all mutants.
    # The pedestal feature then clips chi[-1] to config.transport.chimin, so the
    # mutants have no effect.

    # set up input vectors (all as jax.numpy arrays on face grid)

    # R/LTi profile from current timestep temp_ion
    Ati = -Rmaj * temp_ion_face_grad / temp_ion_face
    # to avoid divisions by zero
    Ati = jnp.where(jnp.abs(Ati) < constants.eps, constants.eps, Ati)

    # R/LTe profile from current timestep temp_el
    Ate = -Rmaj * temp_electron_face_grad / temp_electron_face
    # to avoid divisions by zero
    Ate = jnp.where(jnp.abs(Ate) < constants.eps, constants.eps, Ate)

    # R/Ln profiles from current timestep
    # OK to use normalized version here, because nref in numer and denom
    # cancels.
    Ane = -Rmaj * raw_ne_face_grad / raw_ne_face
    Ani0 = -Rmaj * raw_ni_face_grad / raw_ni_face
    # To avoid divisions by zero in cases where Zeff=1.
    Ani1 = jnp.where(
        jnp.abs(raw_nimp_face) < constants.eps,
        0.0,
        -Rmaj * raw_nimp_face_grad / raw_nimp_face,
    )
    # to avoid divisions by zero
    Ane = jnp.where(jnp.abs(Ane) < constants.eps, constants.eps, Ane)
    Ani0 = jnp.where(jnp.abs(Ani0) < constants.eps, constants.eps, Ani0)
    Ani1 = jnp.where(jnp.abs(Ani1) < constants.eps, constants.eps, Ani1)

    # Calculate q and s.
    # Need to recalculate since in the nonlinear solver psi has intermediate
    # states in the iterative solve.
    q, _ = physics.calc_q_from_psi(
        geo=geo,
        psi=core_profiles.psi,
        q_correction_factor=q_correction_factor,
    )
    smag = physics.calc_s_from_psi_rmid(
        geo,
        core_profiles.psi,
    )

    # Inverse aspect ratio at LCFS.
    epsilon_lcfs = rmid_face[-1] / Rmaj
    # Local normalized radius.
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)

    # Ion to electron temperature ratio
    Ti_Te = temp_ion_face / temp_electron_face

    # logarithm of normalized collisionality
    nu_star = physics.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        nref=nref,
        Zeff_face=Zeff_face,
        coll_mult=transport.coll_mult,
    )
    log_nu_star_face = jnp.log10(nu_star)

    # calculate alpha for magnetic shear correction (see S. van Mulders NF 2021)
    factor_0 = 2 / geo.B0**2 * constants.mu0 * q**2
    alpha = factor_0 * (
        temp_electron_face * constants.keV2J * true_ne_face * (Ate + Ane)
        + true_ni_face * temp_ion_face * constants.keV2J * (Ati + Ani0)
        + true_nimp_face * temp_ion_face * constants.keV2J * (Ati + Ani1)
    )

    # to approximate impact of Shafranov shift. From van Mulders Nucl. Fusion
    # 2021.
    smag = jnp.where(
        transport.smag_alpha_correction,
        smag - alpha / 2,
        smag,
    )

    # very basic ad-hoc sawtooth model
    smag = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        0.1,
        smag,
    )

    q = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        1,
        q,
    )

    smag = jnp.where(
        jnp.logical_and(
            transport.avoid_big_negative_s,
            smag - alpha < -0.2,
        ),
        alpha - 0.2,
        smag,
    )
    normni = raw_ni_face / raw_ne_face
    return QualikizInputs(
        Zeff_face=Zeff_face,
        Ati=Ati,
        Ate=Ate,
        Ane=Ane,
        Ani0=Ani0,
        Ani1=Ani1,
        q=q,
        smag=smag,
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=Rmaj,
        Rmin=Rmin,
        alpha=alpha,
        epsilon_lcfs=epsilon_lcfs,
    )
