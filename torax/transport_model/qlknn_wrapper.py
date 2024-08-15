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
from typing import Callable, Final

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.transport_model import base_qlknn_model
from torax.transport_model import qlknn_10d
from torax.transport_model import qualikiz_utils
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# Environment variable for the QLKNN model. Used if the model path
# is not set in the config.
MODEL_PATH_ENV_VAR: Final[str] = 'TORAX_QLKNN_MODEL_PATH'
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
  # Correction factor to account for multiscale correction in Qualikiz ETG.
  # https://gitlab.com/qualikiz-group/QuaLiKiz/-/commit/5bcd3161c1b08e0272ab3c9412fec7f9345a2eef
  ETG_correction_factor: float = 1.0 / 3.0
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
    return DynamicRuntimeParams(**self._get_interpolation(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(qualikiz_utils.QualikizDynamicRuntimeParams):
  include_ITG: bool
  include_TEM: bool
  include_ETG: bool
  ITG_flux_ratio_correction: float
  ETG_correction_factor: float


_EPSILON_NN: Final[float] = (
    1 / 3
)  # fixed inverse aspect ratio used to train QLKNN10D


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
    qualikiz_inputs = qualikiz_utils.prepare_qualikiz_inputs(
        zeff=runtime_config_inputs.Zeff,
        nref=runtime_config_inputs.nref,
        Ai=runtime_config_inputs.Ai,
        q_correction_factor=runtime_config_inputs.q_correction_factor,
        transport=runtime_config_inputs.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    model = _get_model(self._model_path)
    version = model.version

    # To take into account a different aspect ratio compared to the qlknn
    # training set, the qlknn input normalized radius needs to be rescaled by
    # the inverse aspect ratio. This ensures that the model is evaluated with
    # the correct trapped electron fraction.
    qualikiz_inputs = dataclasses.replace(
        qualikiz_inputs,
        x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
    )
    if version == '10D':
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

    feature_scan = jnp.array([getattr(qualikiz_inputs, key) for key in keys]).T
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
        * runtime_config_inputs.transport.ETG_correction_factor
    )

    pfe = model_output['pfe_itg'].squeeze() + model_output['pfe_tem'].squeeze()

    return qualikiz_utils.make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        qualikiz_inputs=qualikiz_inputs,
        transport=runtime_config_inputs.transport,
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
