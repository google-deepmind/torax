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
import logging
from typing import Callable, Literal

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib
import typing_extensions


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  # Left blank for future extensions
  pass


@dataclasses.dataclass(frozen=True, eq=False)
class TGLFNNukaeaTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """TGLFNN-ukaea transport model."""

  PrepareTGLFNNInputsFunc = Callable[
      [typing_extensions.Self, tglf_based_transport_model.TGLFInputs],
      jax.Array,
  ]

  machine: Literal["step", "multimachine"]
  # The remaining fields are set by __post_init__
  model: tglfnn_ukaea_model.TGLFNNukaeaModel = dataclasses.field(init=False)
  _prepare_tglfnn_inputs: PrepareTGLFNNInputsFunc = dataclasses.field(
      init=False, metadata={"hash_by_id": True}
  )

  def __post_init__(self):

    object.__setattr__(
        self, "model", tglfnn_ukaea_model.TGLFNNukaeaModel(self.machine)
    )

    # We need to install <class>.<method>, not self.<method>, as a field.
    # We let this field hash by id. Installing self.<method> will make `self`
    # become part of the hash, which we don't want. We want <class>.<method>
    # to be a singleton with a unique id / unique hash.
    if self.machine == "step":
      logging.info("Using STEP version of TGLFNNukaea")
      object.__setattr__(
          self,
          "_prepare_tglfnn_inputs",
          TGLFNNukaeaTransportModel._make_input_tensor_step,
      )
    else:
      logging.info("Using Multimachine version of TGLFNNukaea")
      object.__setattr__(
          self,
          "_prepare_tglfnn_inputs",
          TGLFNNukaeaTransportModel._make_input_tensor_multimachine,
      )

    super().__post_init__()

  def _make_input_tensor_step(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as s_hat in s-alpha geometry.
    s_hat = (tglf_inputs.r_minor / tglf_inputs.q) ** 2 * tglf_inputs.q_prime
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
    # to TGLF. This is not the same as s_hat in s-alpha geometry.
    s_hat = (tglf_inputs.r_minor / tglf_inputs.q) ** 2 * tglf_inputs.q_prime

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

  def _call_implementation(
      self,
      transport: tglf_based_transport_model.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,  # unused
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,  # unused
  ) -> transport_model_lib.TurbulentTransport:
    del runtime_params
    del pedestal_model_output
    tglf_inputs = self._prepare_tglf_inputs(transport, geo, core_profiles)
    tglfnn_inputs = self._prepare_tglfnn_inputs(self, tglf_inputs)
    predictions = self.model.predict(tglfnn_inputs)

    # TODO(b/323504363): expose variance outputs
    return self._make_core_transport(
        ion_heat_flux_GB=predictions["efi_gb"][..., 0],
        electron_heat_flux_GB=predictions["efe_gb"][..., 0],
        # TODO(b/323504363): Convert pfi to pfe for multi-ion plasmas
        electron_particle_flux_GB=predictions["pfi_gb"][..., 0],
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )
