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

"""A wrapper around qlknn.

The wrapper provides specifically the pretrained models used for heat
diffusion. The role of the wrapper is send JAX tracers through the
network.
"""

from __future__ import annotations

import functools
import os
import warnings

from absl import flags
import chex
import jax
from jax import numpy as jnp
from qlknn.models import ffnn
from torax import config_slice
from torax import constants as constants_module
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state as state_module
from torax.transport_model import _qlknn_np
from torax.transport_model import transport_model


# Env variable name. See _MODEL_PATH description for how this is used.
_MODEL_PATH_ENV_VAR = 'TORAX_QLKNN_MODEL_PATH'

# Path to the QKLNN models.
_MODEL_PATH = flags.DEFINE_string(
    'torax_qlknn_model_path',
    None,
    'Path to the qlknn model network parameters. If None, then it defaults to '
    f'the {_MODEL_PATH_ENV_VAR} environment variable, if set. If that env '
    'variable is also not set, then it defaults to "third_party/qlknn_hyper". '
    'Users may set the model path by flag or env variable.',
)

# Singleton instance for storing QLKNN networks. These params will be lazily
# loaded once they are needed.
_NETWORKS: Networks | None = None

# TODO( b/320469912)
# make epsilonNN a QLKNN version specific constant
_EPSILON_NN = 1 / 3  # fixed inverse aspect ratio used to train QLKNN10D


def _get_model_path() -> str:
  try:
    if _MODEL_PATH.value is not None:
      return _MODEL_PATH.value
  except flags.Error:
    # This was likely called outside the context of an absl app, so ignore the
    # error and use the environment variable.
    pass
  return os.environ.get(_MODEL_PATH_ENV_VAR, 'third_party/qlknn_hyper')


def _get_networks() -> Networks:
  """Gets the Networks singleton instance."""
  # The advantags of lazily loading the networks is we don't need to do file
  # I/O operations at import time and only need to read these files when needed.
  global _NETWORKS
  if _NETWORKS is None:
    path = _get_model_path()
    _NETWORKS = Networks(path)
  return _NETWORKS


class WrappedQLKNN:
  """A wrapper around a QLKNN network, to be called with JAX tracers.

  Attributes:
    network: The raw `qlknn` network.
  """

  def __init__(self, network: ffnn.QuaLiKizNDNN):
    self.network = network

  def __call__(self, x: jnp.ndarray):
    """Call the network, with a JAX argument."""
    return self.network.get_output(x, safe=False, output_pandas=False)


class Networks:
  """Class holding QLKNN networks.

  Attributes:
    model_path: Path to qlknn-hyper
    net_itgleading: ITG Qi net
    net_itgqediv: ITG Qe/Qi net
    net_temleading: TEM Qe net
    net_temqediv: TEM Qi/Qe net
    net_etgleading: ETG Qe net
    net_temqidiv: TEM Qi/Qe net
    net_tempfediv: Tem pfe/Qe net
    net_etgleading: ITG Qe/Qi net
    net_itgpfediv: ITG pfe/Qi net
  """

  def __init__(self, model_path: str):
    self.model_path = model_path
    self.net_itgleading = self._load('efiitg_gb.json')
    self.net_itgqediv = self._load('efeitg_gb_div_efiitg_gb.json')
    self.net_temleading = self._load('efetem_gb.json')
    self.net_temqidiv = self._load('efitem_gb_div_efetem_gb.json')
    self.net_tempfediv = self._load('pfetem_gb_div_efetem_gb.json')
    self.net_etgleading = self._load('efeetg_gb.json')
    self.net_itgpfediv = self._load('pfeitg_gb_div_efiitg_gb.json')

  def _load(self, path):
    full_path = os.path.join(self.model_path, path)
    try:
      raw = ffnn.QuaLiKizNDNN.from_json(full_path, np=_qlknn_np)
    except FileNotFoundError as fnfe:
      raise FileNotFoundError(
          f'Failed to find file: {full_path}. Check that the file exists. If '
          'the path to the file is not correct, make sure you set '
          f'--{_MODEL_PATH.name} or export the environment variable '
          f'{_MODEL_PATH_ENV_VAR} with the correct path.'
      ) from fnfe
    return WrappedQLKNN(raw)


@chex.dataclass(frozen=True)
class _QLKNNRuntimeConfigInputs:
  """Runtime config inputs for QLKNN.

  The runtime DynamicConfigSlice contains global config parameters, not all of
  which are cacheable. This set of inputs IS cacheable, and using this added
  layer allows the global config to change without affecting how
  QLKNNTransportModel works.
  """

  # pylint: disable=invalid-name
  Rmin: float
  Rmaj: float
  nref: float
  Ai: float
  Zeff: float
  transport: config_slice.DynamicTransportConfigSlice
  Ped_top: float
  set_pedestal: bool
  q_correction_factor: float
  # pylint: enable=invalid-name

  @staticmethod
  def from_config_slice(
      dynamic_config_slice: config_slice.DynamicConfigSlice,
  ) -> _QLKNNRuntimeConfigInputs:
    return _QLKNNRuntimeConfigInputs(
        Rmin=dynamic_config_slice.Rmin,
        Rmaj=dynamic_config_slice.Rmaj,
        nref=dynamic_config_slice.nref,
        Ai=dynamic_config_slice.Ai,
        Zeff=dynamic_config_slice.Zeff,
        transport=dynamic_config_slice.transport,
        Ped_top=dynamic_config_slice.Ped_top,
        set_pedestal=dynamic_config_slice.set_pedestal,
        q_correction_factor=dynamic_config_slice.q_correction_factor,
    )


class QLKNNTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(self):
    super().__init__()
    self._cached_combined = functools.lru_cache(maxsize=10)(self._combined)

  def _call_implementation(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      state: state_module.State,
  ) -> transport_model.TransportCoeffs:
    """Calculates several transport coefficients simultaneously.

    Args:
      dynamic_config_slice: Input config parameters that can change without
        triggering a JAX recompilation.
      geo: Geometry of the torus.
      state: Current simulator state.

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
    runtime_config_inputs = _QLKNNRuntimeConfigInputs.from_config_slice(
        dynamic_config_slice
    )
    try:
      return self._cached_combined(runtime_config_inputs, geo, state)
    except TypeError as e:
      if jax_utils.env_bool('TORAX_COMPILATION_ENABLED', True):
        raise
      warnings.warn(
          "Couldn't cache QLKNN call. This happens when compilation is off "
          ' because concrete arrays are not hashable, but it causes repeat '
          ' computation of QLKNN when normally it would only be traced once.'
          f' Original exception {e}.'
      )

    return self._combined(runtime_config_inputs, geo, state)

  def _combined(
      self,
      runtime_config_inputs: _QLKNNRuntimeConfigInputs,
      geo: geometry.Geometry,
      state: state_module.State,
  ) -> transport_model.TransportCoeffs:
    """Actual implementation of `__call__`.

    `__call__` itself is just a cache dispatch wrapper.

    Args:
      runtime_config_inputs: Input config parameters that can change without
        triggering a JAX recompilation.
      geo: Geometry of the torus.
      state: Current simulator state.

    Returns:
      chi_face_ion: Chi for ion temperature, along faces.
      chi_face_el: Chi for electron temperature, along faces.
      d_face_ne: Diffusivity for electron density, along faces.
      v_face_ne: Convectivity for electron density, along faces.
    """
    constants = constants_module.CONSTANTS

    # pylint: disable=invalid-name
    Rmin = runtime_config_inputs.Rmin
    Rmaj = runtime_config_inputs.Rmaj

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5
    rmid_face = (geo.Rout_face - geo.Rin_face) * 0.5

    temp_ion_var = state.temp_ion
    temp_ion_face = temp_ion_var.face_value()
    temp_ion_face_grad = temp_ion_var.face_grad(rmid)
    temp_el_var = state.temp_el
    temp_electron_face = temp_el_var.face_value()
    temp_electron_face_grad = temp_el_var.face_grad(rmid)
    # Careful, these are in n_ref units, not postprocessed to SI units yet
    raw_ne = state.ne
    raw_ne_face = raw_ne.face_value()
    raw_ne_face_grad = raw_ne.face_grad(rmid)
    raw_ni = state.ni
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

    # TODO(b/323504363): make a unit test that tests this function directly
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
    # Need to recalculate since in the nonlinear solver psi has
    # intermediate states in the iterative solve.
    # To avoid unnecessary complexity for the Jacobian, we still use the
    # old jtot_face in the q calculation. It only modifies the r=0 value
    # of the q-profile. This does not impact qlknn output, which is
    # always stable at r=0 due to the zero gradient boundary conditions.

    q, _ = physics.calc_q_from_jtot_psi(
        geo,
        state.currents.jtot_face,
        state.psi,
        Rmaj,
        runtime_config_inputs.q_correction_factor,
    )
    smag = physics.calc_s_from_psi(
        geo,
        state.psi,
    )

    # local r/Rmin
    # epsilon = r/R

    epsilon = rmid_face[-1] / Rmaj  # inverse aspect ratio at LCFS

    # to take into account a different aspect ratio compared to the qlknn
    # training set, the qlknn input normalized radius needs to be rescaled by
    # the aspect ratio ratio. This ensures that the model is evaluated with
    # the correct trapped electron fraction
    x = rmid_face / rmid_face[-1] * epsilon / _EPSILON_NN

    # Ion to electron temperature ratio
    Ti_Te = temp_ion_face / temp_electron_face

    # logarithm of normalized collisionality
    nu_star = physics.calc_nu_star(
        geo=geo,
        state=state,
        nref=runtime_config_inputs.nref,
        Zeff=runtime_config_inputs.Zeff,
        Rmaj=Rmaj,
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

    # Shape: (num_face, 9)
    feature_scan = jnp.array(
        [Zeff, Ati, Ate, Ane, q, smag, x, Ti_Te, log_nu_star_face]
    ).T
    zeros = jnp.expand_dims(jnp.zeros_like(feature_scan[..., 0]), axis=-1)

    networks = _get_networks()

    # ITG Qi net
    # propagate inputs through net. Clip negative output to zero
    qi_itg = jax.lax.cond(
        runtime_config_inputs.transport.include_ITG,
        lambda: networks.net_itgleading(feature_scan).clip(0),
        lambda: zeros,
    )
    # ITG Qe/Qi net
    # propagate inputs through net and multiply by Qi
    qe_itg = jax.lax.cond(
        runtime_config_inputs.transport.include_ITG,
        lambda: networks.net_itgqediv(feature_scan) * qi_itg,
        lambda: zeros,
    )
    # ITG pfe/Qi net
    # propagate inputs through net and multiply by Qi
    pfe_itg = jax.lax.cond(
        runtime_config_inputs.transport.include_ITG,
        lambda: networks.net_itgpfediv(feature_scan) * qi_itg,
        lambda: zeros,
    )

    # TEM Qe net
    # Clip negative output to zero
    qe_tem = jax.lax.cond(
        runtime_config_inputs.transport.include_TEM,
        lambda: networks.net_temleading(feature_scan).clip(0),
        lambda: zeros,
    )
    # TEM Qi/Qe net
    qi_tem = jax.lax.cond(
        runtime_config_inputs.transport.include_TEM,
        lambda: networks.net_temqidiv(feature_scan) * qe_tem,
        lambda: zeros,
    )
    # TEM pfe/Qe net
    pfe_tem = jax.lax.cond(
        runtime_config_inputs.transport.include_TEM,
        lambda: networks.net_tempfediv(feature_scan) * qe_tem,
        lambda: zeros,
    )

    qe_etg = jax.lax.cond(
        runtime_config_inputs.transport.include_ETG,
        lambda: networks.net_etgleading(feature_scan).clip(0),
        lambda: zeros,
    )

    # combine fluxes
    qi_itg_squeezed = qi_itg.squeeze()
    qi = qi_itg_squeezed + qi_tem.squeeze()
    qe = (
        qe_itg.squeeze()
        * runtime_config_inputs.transport.ITG_flux_ratio_correction
        + qe_tem.squeeze()
        + qe_etg.squeeze()
    )

    pfe = pfe_itg.squeeze() + pfe_tem.squeeze()

    # conversion to SI units (note that n is normalized here)
    pfe_SI = pfe * state.ne.face_value() * chiGB / Rmin

    # chi outputs in SI units.
    # chi in GB units is Q[GB]/(a/LT) , Lref=Rmin in Q[GB].
    # max/min clipping included
    chi_face_ion = (((Rmaj / Rmin) * qi) / Ati) * chiGB
    chi_face_el = (((Rmaj / Rmin) * qe) / Ate) * chiGB
    # enforce chi bounds (sometimes needed for PDE stability)
    chi_face_ion = jnp.clip(
        chi_face_ion,
        runtime_config_inputs.transport.chimin,
        runtime_config_inputs.transport.chimax,
    )
    chi_face_el = jnp.clip(
        chi_face_el,
        runtime_config_inputs.transport.chimin,
        runtime_config_inputs.transport.chimax,
    )

    # Effective D / Effective V approach.
    # For small density gradients or up-gradient transport, set pure effective
    # convection. Otherwise pure effective diffusion.
    def DVeff_approach() -> tuple[jnp.ndarray, jnp.ndarray]:
      Deff = -pfe_SI / (
          state.ne.face_grad() * geo.g1_over_vpr2_face / geo.rmax
          + constants.eps
      )
      Veff = pfe_SI / (state.ne.face_value() * geo.g0_over_vpr_face)
      Deff_mask = (((pfe >= 0) & (Ane >= 0)) | ((pfe < 0) & (Ane < 0))) & (
          abs(Ane) >= runtime_config_inputs.transport.An_min
      )
      Veff_mask = jnp.invert(Deff_mask)
      # Veff_mask is where to use effective V only, so zero out D there.
      d_face_el = jnp.where(Veff_mask, 0.0, Deff)
      # And vice versa
      v_face_el = jnp.where(Deff_mask, 0.0, Veff)
      return d_face_el, v_face_el

    # Scaled D approach. Scale electron diffusivity to electron heat
    # conductivity (this has some physical motivations),
    # and set convection to then match total particle transport
    def Dscaled_approach() -> tuple[jnp.ndarray, jnp.ndarray]:
      chex.assert_rank(pfe, 1)
      d_face_el = jnp.where(jnp.abs(pfe_SI) > 0.0, chi_face_el, 0.0)
      v_face_el = (
          pfe_SI / state.ne.face_value()
          - Ane * d_face_el / Rmaj * geo.g1_over_vpr2_face
      ) / geo.g0_over_vpr_face
      return d_face_el, v_face_el

    d_face_el, v_face_el = jax.lax.cond(
        runtime_config_inputs.transport.DVeff,
        DVeff_approach,
        Dscaled_approach,
    )

    # enforce D and V bounds (sometimes needed for PDE stability)
    d_face_el = jnp.clip(
        d_face_el,
        runtime_config_inputs.transport.Demin,
        runtime_config_inputs.transport.Demax,
    )
    v_face_el = jnp.clip(
        v_face_el,
        runtime_config_inputs.transport.Vemin,
        runtime_config_inputs.transport.Vemax,
    )

    # set low transport in pedestal region to facilitate PDE solver
    # (more consistency between desired profile and transport coefficients)
    # if config.set_pedestal:
    mask = geo.r_face_norm >= runtime_config_inputs.Ped_top
    chi_face_ion = jnp.where(
        jnp.logical_and(runtime_config_inputs.set_pedestal, mask),
        runtime_config_inputs.transport.chimin,
        chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(runtime_config_inputs.set_pedestal, mask),
        runtime_config_inputs.transport.chimin,
        chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(runtime_config_inputs.set_pedestal, mask),
        runtime_config_inputs.transport.Demin,
        d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(runtime_config_inputs.set_pedestal, mask),
        0.0,
        v_face_el,
    )

    # pylint: enable=invalid-name

    return transport_model.TransportCoeffs(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
