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
"""TGLFNN-ukaea transport model."""

from __future__ import annotations

import dataclasses
from typing import Literal

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_coeffs


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  # Left blank for future extension: bool
  pass


@dataclasses.dataclass(frozen=True, eq=False)
class TGLFNNukaeaTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """TGLFNN-ukaea transport model."""

  machine: Literal["step", "multimachine"]

  # The following fields are set by __post_init__
  model: tglfnn_ukaea_model.TGLFNNukaeaModel = dataclasses.field(init=False)

  def __post_init__(self):
    # Load weights in post-init, so that they are not reloaded on every call.
    # Use __setattr__ as this is a frozen dataclass, so we can't just do
    # self.model = ...
    object.__setattr__(
        self, "model", tglfnn_ukaea_model.TGLFNNukaeaModel(self.machine)
    )
    super().__post_init__()

  def _make_input_tensor_step(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as S_HAT_LOC in s-alpha geometry.
    s_hat = (
        tglf_inputs.RMIN_LOC / tglf_inputs.Q_LOC
    ) ** 2 * tglf_inputs.Q_PRIME_LOC
    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.S_KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.S_DELTA_LOC,
            tglf_inputs.BETAE,
            tglf_inputs.ZEFF,
        ],
        axis=-1,
    )

  def _make_input_tensor_multimachine(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as S_HAT_LOC in s-alpha geometry.
    s_hat = (
        tglf_inputs.RMIN_LOC / tglf_inputs.Q_LOC
    ) ** 2 * tglf_inputs.Q_PRIME_LOC

    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.ZEFF,
            tglf_inputs.VEXB_SHEAR,
        ],
        axis=-1,
    )

  def _prepare_tglfnn_inputs(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    """Prepares the input tensor for the surrogate model.

    If the model exposes `input_labels`, dynamically stacks features from the
    canonical physics dictionary. Otherwise, falls back to legacy machine-specific
    dispatch.
    """
    if hasattr(self.model, "input_labels") and self.model.input_labels:
      feature_dict = tglf_based_transport_model.get_canonical_physics_dict(
          tglf_inputs
      )
      missing_labels = [
          col for col in self.model.input_labels if col not in feature_dict
      ]
      if missing_labels:
        raise ValueError(
            f"Surrogate model requested features not found in canonical physics dictionary: {missing_labels}"
        )
      return jnp.stack(
          [feature_dict[col] for col in self.model.input_labels],
          axis=-1,
      )

    match self.machine:
      case "step":
        return self._make_input_tensor_step(tglf_inputs)
      case "multimachine":
        return self._make_input_tensor_multimachine(tglf_inputs)
      case _:
        raise ValueError(f"Unsupported machine: {self.machine}")

  def predict_with_uncertainty(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Evaluates the surrogate model and returns predictive (means, variances).

    Returns:
      means: Dict of {channel: mean_profile} across faces.
      variances: Dict of {channel: variance_profile} across faces.
    """
    tglfnn_inputs = self._prepare_tglfnn_inputs(tglf_inputs)
    predictions = self.model.predict(tglfnn_inputs)
    means = {k: predictions[k][..., 0] for k in predictions}
    variances = {k: predictions[k][..., 1] for k in predictions}
    return means, variances

  def compute_relative_uncertainty(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      eps: float = 1e-4,
  ) -> jax.Array:
    """Computes max relative uncertainty across all predicted flux channels.

    Returns:
      Array of shape (n_faces,) with the maximum relative standard deviation.
    """
    means, variances = self.predict_with_uncertainty(tglf_inputs)
    rel_uncs = []
    for k in ["efi_gb", "efe_gb", "pfi_gb"]:
      if k in means:
        sigma = jnp.sqrt(jnp.maximum(variances[k], 0.0))
        rel_unc = sigma / (jnp.abs(means[k]) + eps)
        rel_uncs.append(rel_unc)
    if not rel_uncs:
      return jnp.zeros_like(tglf_inputs.RLTS_1)
    return jnp.max(jnp.stack(rel_uncs, axis=0), axis=0)

  def call_implementation(
      self,
      transport: tglf_based_transport_model.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    means, _ = self.predict_with_uncertainty(tglf_inputs)

    return self._make_core_transport(
        ion_heat_flux_GB=means["efi_gb"],
        electron_heat_flux_GB=means["efe_gb"],
        # TODO(b/323504363): Convert pfi to pfe for multi-ion plasmas
        electron_particle_flux_GB=means["pfi_gb"],
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
