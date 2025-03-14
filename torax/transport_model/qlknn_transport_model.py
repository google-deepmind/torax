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

"""A transport model that uses a QLKNN model."""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Callable, Final

import chex
import jax
from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import base_qlknn_model
from torax.transport_model import qlknn_10d
from torax.transport_model import qlknn_model_wrapper
from torax.transport_model import qualikiz_based_transport_model
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


def get_default_runtime_params_from_model_path(
    model_path: str,
) -> RuntimeParams:
  """Returns default runtime params for the model version given the path."""
  version = _get_model(model_path).version
  if version == '10D':
    return RuntimeParams(
        # Correction factor to a more recent QLK collision operator.
        coll_mult=0.25,
        # The QLK version this specific QLKNN was trained on tends to
        # underpredict ITG electron heat flux in shaped, high-beta scenarios.
        ITG_flux_ratio_correction=2.0,
    )
  elif version == '11D':
    return RuntimeParams()
  else:
    raise ValueError(f'Unknown model version: {version}')


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(qualikiz_based_transport_model.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  include_ITG: bool = True  # to toggle ITG modes on or off
  include_TEM: bool = True  # to toggle TEM modes on or off
  include_ETG: bool = True  # to toggle ETG modes on or off
  # This is a correction factor for ITG electron heat flux.
  ITG_flux_ratio_correction: float = 1.0
  # Correction factor to account for multiscale correction in Qualikiz ETG.
  # https://gitlab.com/qualikiz-group/QuaLiKiz/-/commit/5bcd3161c1b08e0272ab3c9412fec7f9345a2eef
  ETG_correction_factor: float = 1.0 / 3.0
  # clip inputs within desired margin of the QLKNN training set boundaries
  clip_inputs: bool = False
  clip_margin: float = 0.95

  def make_provider(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(qualikiz_based_transport_model.DynamicRuntimeParams):
  include_ITG: bool
  include_TEM: bool
  include_ETG: bool
  ITG_flux_ratio_correction: float
  ETG_correction_factor: float
  clip_inputs: bool
  clip_margin: float


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


_EPSILON_NN: Final[float] = (
    1 / 3
)  # fixed inverse aspect ratio used to train QLKNN10D


# Memoize, but evict the old model if a new path is given.
@functools.lru_cache(maxsize=1)
def _get_model(path: str) -> base_qlknn_model.BaseQLKNNModel:
  """Load the model."""
  logging.info('Loading model from %s', path)
  try:
    # New QLKNN models are encapsulated in a single `.qlknn` file.
    if path.endswith('.qlknn'):
      return qlknn_model_wrapper.QLKNNModelWrapper(path)
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
  Zeff_face: chex.Array
  transport: DynamicRuntimeParams
  Ped_top: float
  set_pedestal: bool
  # pylint: enable=invalid-name

  @staticmethod
  def from_runtime_params_slice(
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> 'QLKNNRuntimeConfigInputs':
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )
    return QLKNNRuntimeConfigInputs(
        nref=dynamic_runtime_params_slice.numerics.nref,
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        transport=dynamic_runtime_params_slice.transport,
        Ped_top=pedestal_model_output.rho_norm_ped_top,
        set_pedestal=dynamic_runtime_params_slice.profile_conditions.set_pedestal,
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


def clip_inputs(
    feature_scan: jax.Array,
    clip_margin: float,
    inputs_and_ranges: base_qlknn_model.InputsAndRanges,
) -> jax.Array:
  """Clip input values according to the training set limits + optional user-defined margin for qlknn."""
  for i, key in enumerate(inputs_and_ranges.keys()):
    bounds = inputs_and_ranges[key]
    # set min or max if present in the bounds dict
    min_val = bounds.get('min', -jnp.inf)
    max_val = bounds.get('max', jnp.inf)
    # increase/decrease bounds based on clip_margin
    min_val += jnp.where(
        jnp.isfinite(min_val), jnp.abs(min_val) * (1 - clip_margin), 0.0
    )
    max_val -= jnp.where(
        jnp.isfinite(max_val), jnp.abs(max_val) * (1 - clip_margin), 0.0
    )
    feature_scan = feature_scan.at[:, i].set(
        jnp.clip(
            feature_scan[:, i],
            min_val,
            max_val,
        )
    )
  return feature_scan


class QLKNNTransportModel(
    qualikiz_based_transport_model.QualikizBasedTransportModel
):
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
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    """Calculates several transport coefficients simultaneously.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: transport coefficients
    """

    runtime_config_inputs = QLKNNRuntimeConfigInputs.from_runtime_params_slice(
        dynamic_runtime_params_slice,
        pedestal_model_output,
    )
    return self._combined(runtime_config_inputs, geo, core_profiles)

  # Wrap in JIT here in order to cache the tracing/compilation of this function.
  # We mark self as static because it is a singleton. Other args are pytrees.
  # There's no global coordination of calls to transport model so it is called
  # 2-4X with the same args. Caching prevents construction of multiple copies of
  # identical expressions saving ~30% in compile time.
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
    qualikiz_inputs = self._prepare_qualikiz_inputs(
        Zeff_face=runtime_config_inputs.Zeff_face,
        nref=runtime_config_inputs.nref,
        transport=runtime_config_inputs.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    model = _get_model(self._model_path)

    # To take into account a different aspect ratio compared to the qlknn
    # training set, the qlknn input normalized radius needs to be rescaled by
    # the inverse aspect ratio. This ensures that the model is evaluated with
    # the correct trapped electron fraction.
    qualikiz_inputs = dataclasses.replace(
        qualikiz_inputs,
        x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
    )

    feature_scan = model.get_model_inputs_from_qualikiz_inputs(qualikiz_inputs)
    # Clip inputs if requested.
    # TODO(b/364218524): Consider better clipping of out-of-distribution inputs.
    feature_scan = jax.lax.cond(
        runtime_config_inputs.transport.clip_inputs,
        lambda: clip_inputs(
            feature_scan,
            runtime_config_inputs.transport.clip_margin,
            model.inputs_and_ranges,
        ),  # Called when True
        lambda: feature_scan,  # Called when False
    )
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

    return self._make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        quasilinear_inputs=qualikiz_inputs,
        transport=runtime_config_inputs.transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.Rmaj,
        gyrobohm_flux_reference_length=geo.Rmin,
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

  runtime_params: RuntimeParams | None = None
  model_path: str = dataclasses.field(default_factory=get_default_model_path)

  def __post_init__(self):
    if self.runtime_params is None:
      self.runtime_params = get_default_runtime_params_from_model_path(
          self.model_path
      )

  _builder: Callable[
      [str],
      QLKNNTransportModel,
  ] = _default_qlknn_builder

  def __call__(
      self,
  ) -> QLKNNTransportModel:
    return self._builder(self.model_path)
