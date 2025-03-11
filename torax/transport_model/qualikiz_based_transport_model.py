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
from torax import state
from torax.geometry import geometry
from torax.physics import collisions
from torax.physics import psi_calculations
from torax.torax_pydantic import torax_pydantic
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
      self, torax_mesh: torax_pydantic.Grid1D | None = None
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
  q: chex.Array
  smag: chex.Array
  x: chex.Array
  Ti_Te: chex.Array
  log_nu_star_face: chex.Array
  normni: chex.Array
  alpha: chex.Array
  epsilon_lcfs: chex.Array

  # Also define the logarithmic gradients using standard QuaLiKiz notation.
  @property
  def Ati(self) -> chex.Array:
    return self.lref_over_lti

  @property
  def Ate(self) -> chex.Array:
    return self.lref_over_lte

  @property
  def Ane(self) -> chex.Array:
    return self.lref_over_lne

  @property
  def Ani0(self) -> chex.Array:
    return self.lref_over_lni0

  @property
  def Ani1(self) -> chex.Array:
    return self.lref_over_lni1


class QualikizBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for Qualikiz-based transport models."""

  def _prepare_qualikiz_inputs(
      self,
      Zeff_face: chex.Array,
      nref: chex.Numeric,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> QualikizInputs:
    """Prepare Qualikiz inputs."""
    constants = constants_module.CONSTANTS

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5
    rmid_face = (geo.Rout_face - geo.Rin_face) * 0.5

    # gyrobohm diffusivity
    # (defined here with Lref=Rmin due to QLKNN training set normalization)
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.temp_ion.face_value(),
        reference_magnetic_field=geo.B0,
        reference_mass=core_profiles.Ai,
        reference_length=geo.Rmin,
    )

    # transport coefficients from the qlknn-hyper-10D model
    # (K.L. van de Plassche PoP 2020)

    # set up input vectors (all as jax.numpy arrays on face grid)

    # Calculate normalized logarithmic gradients
    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=rmid,
        reference_length=geo.Rmaj,
    )

    q = core_profiles.q_face

    # Due to QuaLikiz geometry assumptions, we need to calculate s with respect
    # to the midplane average, and not use the standard s_face from CoreProfiles
    smag = psi_calculations.calc_s_rmid(
        geo,
        core_profiles.psi,
    )

    # Inverse aspect ratio at LCFS.
    epsilon_lcfs = rmid_face[-1] / geo.Rmaj
    # Local normalized radius.
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)

    # Ion to electron temperature ratio
    Ti_Te = (
        core_profiles.temp_ion.face_value() / core_profiles.temp_el.face_value()
    )

    # logarithm of normalized collisionality
    nu_star = collisions.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        nref=nref,
        Zeff_face=Zeff_face,
        coll_mult=transport.coll_mult,
    )
    log_nu_star_face = jnp.log10(nu_star)

    # calculate alpha for magnetic shear correction (see S. van Mulders NF 2021)
    alpha = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        nref=nref,
        q=q,
        reference_magnetic_field=geo.B0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
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
    normni = core_profiles.ni.face_value() / core_profiles.ne.face_value()
    return QualikizInputs(
        Zeff_face=Zeff_face,
        lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
        lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
        lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
        lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
        lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
        q=q,
        smag=smag,
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=geo.Rmaj,
        Rmin=geo.Rmin,
        alpha=alpha,
        epsilon_lcfs=epsilon_lcfs,
    )
