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
import dataclasses
import functools
import logging
import os
from typing import Final

import chex
import jax
from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import base_qlknn_model
from torax.transport_model import qlknn_10d
from torax.transport_model import qlknn_model_wrapper
from torax.transport_model import qualikiz_based_transport_model
import typing_extensions


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(qualikiz_based_transport_model.DynamicRuntimeParams):
  include_ITG: bool
  include_TEM: bool
  include_ETG: bool
  ITG_flux_ratio_correction: float
  ETG_correction_factor: float
  clip_inputs: bool
  clip_margin: float


_EPSILON_NN: Final[float] = (
    1 / 3
)  # fixed inverse aspect ratio used to train QLKNN models


# Memoize, but evict the old model if a new path is given.
@functools.lru_cache(maxsize=1)
def get_model(path: str, name: str) -> base_qlknn_model.BaseQLKNNModel:
  """Load the model."""
  logging.info('Loading model from %s', path)
  try:
    if path:
      if not path.endswith('.qlknn'):
        return qlknn_10d.QLKNN10D(path, name)
      else:
        return qlknn_model_wrapper.QLKNNModelWrapper(path, name)
    elif name == qlknn_10d.QLKNN10D_NAME:
      raise ValueError(
          'To use QLKNN10D, please provide a path to the qlknn-hyper directory.'
      )
    return qlknn_model_wrapper.QLKNNModelWrapper(path, name)
  except FileNotFoundError as fnfe:
    raise FileNotFoundError(
        f'Failed to load model with path "{path}" and name "{name}". Check that'
        ' the path exists.'
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
        set_pedestal=dynamic_runtime_params_slice.pedestal.set_pedestal,
    )


def _filter_model_output(
    model_output: base_qlknn_model.ModelOutput,
    include_ITG: bool,
    include_TEM: bool,
    include_ETG: bool,
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

  def filter_flux(flux_name: str, value: jax.Array) -> jax.Array:
    return jax.lax.cond(
        filter_map.get(flux_name, True),
        lambda: value,
        lambda: jnp.zeros_like(value),
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
      path: str,
      name: str,
  ):
    super().__init__()
    self._path = path
    self._name = name
    self._frozen = True

  @property
  def path(self) -> str:
    return self._path

  @property
  def name(self) -> str:
    return self._name

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
    model = get_model(self.path, self.name)

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
    model_output = _filter_model_output(
        model_output=model_output,
        include_ITG=runtime_config_inputs.transport.include_ITG,
        include_TEM=runtime_config_inputs.transport.include_TEM,
        include_ETG=runtime_config_inputs.transport.include_ETG,
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
        gradient_reference_length=geo.R_major,
        gyrobohm_flux_reference_length=geo.a_minor,
    )

  def __hash__(self) -> int:
    return hash(('QLKNNTransportModel' + self.path + self.name))

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return (
        isinstance(other, QLKNNTransportModel)
        and self.path == other.path and self.name == other.name
    )
