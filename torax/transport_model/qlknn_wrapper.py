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

"""A wrapper around qlknn_10d.

The wrapper calls the pretrained models trained on QuaLiKiz heat and
particle transport. The wrapper calculates qlknn_10d inputs, infers the
model, carries out post-processing, and returns a CoreTransport object
with turbulent transport coefficients.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Callable

import chex
import jax
from jax import numpy as jnp
from torax import constants as constants_module
from torax import geometry
from torax import physics
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.transport_model import base_qlknn_model
from torax.transport_model import qlknn_10d
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# Environment variable for the QLKNN model. Used if the model path
# is not set in the config.
MODEL_PATH_ENV_VAR = 'TORAX_QLKNN_MODEL_PATH'
# If no path is set in either the config or the environment variable, use
# this path.
DEFAULT_MODEL_PATH = '~/qlknn_hyper'


def get_default_model_path() -> str:
  return os.environ.get(MODEL_PATH_ENV_VAR, DEFAULT_MODEL_PATH)


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # Collisionality multiplier in QLKNN for sensitivity testing.
  # Default is 0.25 (correction factor to a more recent QLK collision operator)
  coll_mult: float = 0.25
  include_ITG: bool = True  # to toggle ITG modes on or off
  include_TEM: bool = True  # to toggle TEM modes on or off
  include_ETG: bool = True  # to toggle ETG modes on or off
  # The QLK version this specific QLKNN was trained on tends to underpredict
  # ITG electron heat flux in shaped, high-beta scenarios.
  # This is a correction factor
  ITG_flux_ratio_correction: float = 2.0
  # effective D / effective V approach for particle transport
  DVeff: bool = False
  # minimum |R/Lne| below which effective V is used instead of effective D
  An_min: float = 0.05
  # ensure that smag - alpha > -0.2 always, to compensate for no slab modes
  avoid_big_negative_s: bool = True
  # reduce magnetic shear by 0.5*alpha to capture main impact of alpha
  smag_alpha_correction: bool = True
  # if q < 1, modify input q and smag as if q~1 as if there are sawteeth
  q_sawtooth_proxy: bool = True

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  coll_mult: float
  include_ITG: bool
  include_TEM: bool
  include_ETG: bool
  ITG_flux_ratio_correction: float
  DVeff: bool
  An_min: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool


_EPSILON_NN: float = 1 / 3  # fixed inverse aspect ratio used to train QLKNN10D


# Memoize, but evict the old model if a new path is given.
@functools.lru_cache(maxsize=1)
def _get_model(path: str) -> base_qlknn_model.BaseQLKNNModel:
  """Load the model."""
  logging.info('Loading model from %s', path)
  try:
    return qlknn_10d.QLKNN10D(path)
  except FileNotFoundError as fnfe:
    raise FileNotFoundError(
        f'Failed to load model from {path}. Check that the path exists.'
    ) from fnfe


@chex.dataclass(frozen=True)
class QLKNNRuntimeConfigInputs:
  """Runtime config inputs for QLKNN.

  The runtime DynamicRuntimeParamsSlice contains global runtime parameters, not
  all of which are cacheable. This set of inputs IS cacheable, and using this
  added layer allows the global config to change without affecting how
  QLKNNTransportModel works.
  """

  # pylint: disable=invalid-name
  nref: float
  Ai: float
  Zeff: float
  transport: DynamicRuntimeParams
  Ped_top: float
  set_pedestal: bool
  q_correction_factor: float
  # pylint: enable=invalid-name

  @staticmethod
  def from_runtime_params_slice(
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
  ) -> 'QLKNNRuntimeConfigInputs':
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )
    return QLKNNRuntimeConfigInputs(
        nref=dynamic_runtime_params_slice.numerics.nref,
        Ai=dynamic_runtime_params_slice.plasma_composition.Ai,
        Zeff=dynamic_runtime_params_slice.plasma_composition.Zeff,
        transport=dynamic_runtime_params_slice.transport,
        Ped_top=dynamic_runtime_params_slice.profile_conditions.Ped_top,
        set_pedestal=dynamic_runtime_params_slice.profile_conditions.set_pedestal,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
    )


def filter_model_output(
    model_output: base_qlknn_model.ModelOutput,
    include_ITG: bool,
    include_TEM: bool,
    include_ETG: bool,
    zeros_shape: tuple[int, ...],
) -> base_qlknn_model.ModelOutput:
  """Potentially filtering out some fluxes."""
  filter_map = {
      'qi_itg': include_ITG,
      'qe_itg': include_ITG,
      'pfe_itg': include_ITG,
      'qe_tem': include_TEM,
      'qi_tem': include_TEM,
      'pfe_tem': include_TEM,
      'qe_etg': include_ETG,
  }
  zeros = jnp.zeros(zeros_shape)

  def filter_flux(flux_name: str, value: jax.Array) -> jax.Array:
    return jax.lax.cond(
        filter_map.get(flux_name, True),
        lambda: value,
        lambda: zeros,
    )

  return {k: filter_flux(k, v) for k, v in model_output.items()}


def prepare_qualikiz_inputs(
    runtime_config_inputs: QLKNNRuntimeConfigInputs,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> dict[str, chex.Array]:
  """Prepare Qualikiz inputs."""
  constants = constants_module.CONSTANTS

  # pylint: disable=invalid-name
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

  # True SI value versions
  true_ne_face = raw_ne_face * runtime_config_inputs.nref
  true_ni_face = raw_ni_face * runtime_config_inputs.nref

  # pylint: disable=invalid-name
  # gyrobohm diffusivity
  # (defined here with Lref=Rmin due to QLKNN training set normalization)
  chiGB = (
      (runtime_config_inputs.Ai * constants.mp) ** 0.5
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
  Zeff = runtime_config_inputs.Zeff * jnp.ones_like(geo.r_face)

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
  Ani = -Rmaj * raw_ni_face_grad / raw_ni_face
  # to avoid divisions by zero
  Ane = jnp.where(jnp.abs(Ane) < constants.eps, constants.eps, Ane)
  Ani = jnp.where(jnp.abs(Ani) < constants.eps, constants.eps, Ani)

  # Calculate q and s.
  # Need to recalculate since in the nonlinear solver psi has intermediate
  # states in the iterative solve.
  # To avoid unnecessary complexity for the Jacobian, we still use the
  # old jtot_face in the q calculation. It only modifies the r=0 value
  # of the q-profile. This does not impact qlknn output, which is
  # always stable at r=0 due to the zero gradient boundary conditions.

  q, _ = physics.calc_q_from_jtot_psi(
      geo=geo,
      psi=core_profiles.psi,
      jtot_face=core_profiles.currents.jtot_face,
      q_correction_factor=runtime_config_inputs.q_correction_factor,
  )
  smag = physics.calc_s_from_psi(
      geo,
      core_profiles.psi,
  )

  # local r/Rmin
  # epsilon = r/R
  epsilon = rmid_face[-1] / Rmaj  # inverse aspect ratio at LCFS

  x = rmid_face / rmid_face[-1] * epsilon

  # Ion to electron temperature ratio
  Ti_Te = temp_ion_face / temp_electron_face

  # logarithm of normalized collisionality
  nu_star = physics.calc_nu_star(
      geo=geo,
      core_profiles=core_profiles,
      nref=runtime_config_inputs.nref,
      Zeff=runtime_config_inputs.Zeff,
      coll_mult=runtime_config_inputs.transport.coll_mult,
  )
  log_nu_star_face = jnp.log10(nu_star)

  # calculate alpha for magnetic shear correction (see S. van Mulders NF 2021)
  factor_0 = 2 / geo.B0**2 * constants.mu0 * q**2
  alpha = factor_0 * (
      temp_electron_face * constants.keV2J * true_ne_face * (Ate + Ane)
      + true_ni_face * temp_ion_face * constants.keV2J * (Ati + Ani)
  )

  # to approximate impact of Shafranov shift. From van Mulders Nucl. Fusion
  # 2021.
  smag = jax.lax.cond(
      runtime_config_inputs.transport.smag_alpha_correction,
      lambda: smag - alpha / 2,
      lambda: smag,
  )

  # very basic ad-hoc sawtooth model
  smag = jnp.where(
      jnp.logical_and(
          runtime_config_inputs.transport.q_sawtooth_proxy,
          q < 1,
      ),
      0.1,
      smag,
  )

  q = jnp.where(
      jnp.logical_and(
          runtime_config_inputs.transport.q_sawtooth_proxy,
          q < 1,
      ),
      1,
      q,
  )

  smag = jnp.where(
      jnp.logical_and(
          runtime_config_inputs.transport.avoid_big_negative_s,
          smag - alpha < -0.2,
      ),
      alpha - 0.2,
      smag,
  )
  normni = raw_ni_face / raw_ne_face
  return {
      'Zeff': Zeff,
      'Ati': Ati,
      'Ate': Ate,
      'Ane': Ane,
      'Ani': Ani,
      'q': q,
      'smag': smag,
      'x': x,
      'Ti_Te': Ti_Te,
      'log_nu_star_face': log_nu_star_face,
      'normni': normni,
      'chiGB': chiGB,
      'Rmaj': Rmaj,
      'Rmin': Rmin,
  }


def make_core_transport(
    qi: jax.Array,
    qe: jax.Array,
    pfe: jax.Array,
    prepared_data: dict[str, jax.Array],
    runtime_config_inputs: QLKNNRuntimeConfigInputs,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> state.CoreTransport:
  """Converts model output to CoreTransport."""
  constants = constants_module.CONSTANTS

  # conversion to SI units (note that n is normalized here)
  pfe_SI = (
      pfe
      * core_profiles.ne.face_value()
      * prepared_data['chiGB']
      / prepared_data['Rmin']
  )

  # chi outputs in SI units.
  # chi in GB units is Q[GB]/(a/LT) , Lref=Rmin in Q[GB].
  # max/min clipping included
  chi_face_ion = (
      ((prepared_data['Rmaj'] / prepared_data['Rmin']) * qi)
      / prepared_data['Ati']
  ) * prepared_data['chiGB']
  chi_face_el = (
      ((prepared_data['Rmaj'] / prepared_data['Rmin']) * qe)
      / prepared_data['Ate']
  ) * prepared_data['chiGB']

  # Effective D / Effective V approach.
  # For small density gradients or up-gradient transport, set pure effective
  # convection. Otherwise pure effective diffusion.
  def DVeff_approach() -> tuple[jax.Array, jax.Array]:
    # The geo.rmax is to unnormalize the face_grad.
    Deff = -pfe_SI / (
        core_profiles.ne.face_grad() * geo.g1_over_vpr2_face * geo.rmax
        + constants.eps
    )
    Veff = pfe_SI / (
        core_profiles.ne.face_value() * geo.g0_over_vpr_face * geo.rmax
    )
    Deff_mask = (
        ((pfe >= 0) & (prepared_data['Ane'] >= 0))
        | ((pfe < 0) & (prepared_data['Ane'] < 0))
    ) & (abs(prepared_data['Ane']) >= runtime_config_inputs.transport.An_min)
    Veff_mask = jnp.invert(Deff_mask)
    # Veff_mask is where to use effective V only, so zero out D there.
    d_face_el = jnp.where(Veff_mask, 0.0, Deff)
    # And vice versa
    v_face_el = jnp.where(Deff_mask, 0.0, Veff)
    return d_face_el, v_face_el

  # Scaled D approach. Scale electron diffusivity to electron heat
  # conductivity (this has some physical motivations),
  # and set convection to then match total particle transport
  def Dscaled_approach() -> tuple[jax.Array, jax.Array]:
    chex.assert_rank(pfe, 1)
    d_face_el = jnp.where(jnp.abs(pfe_SI) > 0.0, chi_face_el, 0.0)
    v_face_el = (
        pfe_SI / core_profiles.ne.face_value()
        - prepared_data['Ane']
        * d_face_el
        / prepared_data['Rmaj']
        * geo.g1_over_vpr2_face
        * geo.rmax**2
    ) / (geo.g0_over_vpr_face * geo.rmax)
    return d_face_el, v_face_el

  d_face_el, v_face_el = jax.lax.cond(
      runtime_config_inputs.transport.DVeff,
      DVeff_approach,
      Dscaled_approach,
  )

  # pylint: enable=invalid-name

  return state.CoreTransport(
      chi_face_ion=chi_face_ion,
      chi_face_el=chi_face_el,
      d_face_el=d_face_el,
      v_face_el=v_face_el,
  )


class QLKNNTransportModel(transport_model.TransportModel):
  """Calculates turbulent transport coefficients."""

  def __init__(
      self,
      model_path: str,
  ):
    super().__init__()
    self._model_path = model_path
    self._frozen = True

  @property
  def model_path(self) -> str:
    return self._model_path

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Calculates several transport coefficients simultaneously.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.

    Returns:
      coeffs: transport coefficients
    """

    # lru_cache is important: there's no global coordination of calls to
    # transport model so it is called 2-4X with the same args. Caching prevents
    # construction of multiple copies of identical expressions, and these are
    # expensive expressions because they have all branches of dynamic config
    # compiled in and selected using cond, they're qlknn neural networks,
    # they're expensive to trace because it's numpy / filesystem access.
    # Caching this one function reduces trace time for a whole
    # end to end sim by 35% and compile time by 30%.
    # This only works for tracers though, since concrete (numpy) arrays aren't
    # hashable. We assume that either we're running a whole sim in uncompiled
    # mode and everything is concrete or we're running a whole sim in compiled
    # mode and everything is a tracer, so we can just test one value.
    runtime_config_inputs = QLKNNRuntimeConfigInputs.from_runtime_params_slice(
        dynamic_runtime_params_slice
    )
    return self._combined(runtime_config_inputs, geo, core_profiles)

  # Wrap in JIT here in order to cache the tracing/compilation of this function.
  # We mark self as static because it is a singleton. Other args are pytrees.
  @functools.partial(jax.jit, static_argnames=['self'])
  def _combined(
      self,
      runtime_config_inputs: QLKNNRuntimeConfigInputs,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Actual implementation of `__call__`.

    `__call__` itself is just a cache dispatch wrapper.

    Args:
      runtime_config_inputs: Input runtime parameters that can change without
        triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.

    Returns:
      chi_face_ion: Chi for ion temperature, along faces.
      chi_face_el: Chi for electron temperature, along faces.
      d_face_ne: Diffusivity for electron density, along faces.
      v_face_ne: Convectivity for electron density, along faces.
    """
    prepared_data = prepare_qualikiz_inputs(
        runtime_config_inputs=runtime_config_inputs,
        geo=geo,
        core_profiles=core_profiles,
    )
    model = _get_model(self._model_path)
    version = model.version
    if version == '10D':
      # To take into account a different aspect ratio compared to the qlknn10D
      # training set, the qlknn input normalized radius needs to be rescaled by
      # the aspect ratio ratio. This ensures that the model is evaluated with
      # the correct trapped electron fraction.
      prepared_data['x'] /= _EPSILON_NN
      keys = [
          'Zeff',
          'Ati',
          'Ate',
          'Ane',
          'q',
          'smag',
          'x',
          'Ti_Te',
          'log_nu_star_face',
      ]
    else:
      raise ValueError(f'Unknown model version: {version}')
    feature_scan = jnp.array([prepared_data[key] for key in keys]).T
    model_output = model.predict(feature_scan)
    model_output = filter_model_output(
        model_output=model_output,
        include_ITG=runtime_config_inputs.transport.include_ITG,
        include_TEM=runtime_config_inputs.transport.include_TEM,
        include_ETG=runtime_config_inputs.transport.include_ETG,
        zeros_shape=(feature_scan.shape[0], 1),
    )

    # combine fluxes
    qi_itg_squeezed = model_output['qi_itg'].squeeze()
    qi = qi_itg_squeezed + model_output['qi_tem'].squeeze()
    qe = (
        model_output['qe_itg'].squeeze()
        * runtime_config_inputs.transport.ITG_flux_ratio_correction
        + model_output['qe_tem'].squeeze()
        + model_output['qe_etg'].squeeze()
    )

    pfe = model_output['pfe_itg'].squeeze() + model_output['pfe_tem'].squeeze()

    return make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        prepared_data=prepared_data,
        runtime_config_inputs=runtime_config_inputs,
        geo=geo,
        core_profiles=core_profiles,
    )

  def __hash__(self) -> int:
    return hash(('QLKNNTransportModel' + self._model_path))

  def __eq__(self, other: QLKNNTransportModel) -> bool:
    return (
        isinstance(other, QLKNNTransportModel)
        and self.model_path == other.model_path
    )


def _default_qlknn_builder(model_path: str) -> QLKNNTransportModel:
  return QLKNNTransportModel(model_path)


@dataclasses.dataclass(kw_only=True)
class QLKNNTransportModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class QLKNNTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )
  model_path: str | None = None

  _builder: Callable[
      [str],
      QLKNNTransportModel,
  ] = _default_qlknn_builder

  def __call__(
      self,
  ) -> QLKNNTransportModel:
    if not self.model_path:
      self.model_path = get_default_model_path()
    return self._builder(self.model_path)
