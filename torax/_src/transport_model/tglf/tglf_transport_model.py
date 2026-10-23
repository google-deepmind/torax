# Copyright 2026 DeepMind Technologies Limited
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

"""A TORAX transport model that calls TGLF in-memory using a threadpool."""

from concurrent import futures
import dataclasses
from typing import Annotated, Any, Literal, TypeAlias

import chex
import jax
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib
from torax._src.transport_model.tglf import defaults as tglf_defaults
from torax._src.transport_model.tglf import tglf2py

import multiprocessing

# pylint: disable=invalid-name


TGLFSettingsValueTypes: TypeAlias = str | float | int | bool | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for the TGLF transport model."""

  n_processes: int
  n_cores_per_process: int
  tglf_settings: dict[str, TGLFSettingsValueTypes] = jax.tree.static()


def _run_single_tglf(
    i: int,
    local_tglf_settings: dict[str, Any],
    global_tglf_settings: dict[str, Any],
) -> tuple[int, float, float, float]:
  """Runs a single tglf2py evaluation in a worker process.

  Merges the TGLF input parameters for the i-th face with the global TGLF
  settings and then calls tglf2py.run_tglf() to compute the turbulent fluxes.

  This function must be at the top level of the module to be pickleable, which
  is required for passing to the executor.

  Args:
    i: The index of the run
    local_tglf_settings: Dictionary of TGLF input arrays, whose values are
      length n_faces.
    global_tglf_settings: Dictionary of global TGLF settings (scalars).

  Returns:
    A tuple of the index, electron particle flux, electron heat flux and total
    ion heat flux.
  """
  tglf_inputs_i = jax.tree.map(
      lambda x: float(x[i]),
      local_tglf_settings,
  )
  run_inputs = global_tglf_settings | tglf_inputs_i
  (
      electron_particle_flux,
      _,
      electron_heat_flux,
      ion_heat_flux,
  ) = tglf2py.run_tglf(**run_inputs)
  return (
      i,
      float(electron_particle_flux),
      float(electron_heat_flux),
      float(np.sum(ion_heat_flux)),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class TGLFTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculates turbulent transport coefficients with tglf2py in-memory."""

  # Hash by id as should be unique per instance
  executor: futures.Executor = dataclasses.field(metadata={'hash_by_id': True})

  def call_implementation(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      component_state: transport_state_lib.ComponentTransportState,
  ) -> tuple[
      transport_coeffs.TransportCoeffs,
      transport_state_lib.ComponentTransportState,
  ]:
    """Calculates several transport coefficients and state simultaneously.

    Args:
      transport_runtime_params: Input runtime parameters for this transport
        model.
      runtime_params: Input runtime parameters for all components of the
        simulation at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      two_point_mask: Boolean mask on the face grid indicating where to use
        2-point central differencing instead of 3-point polynomial interpolation
        for gradients.
      component_state: Component transport state carried over from the previous
        timestep. Unused by this model and passed through unchanged.

    Returns:
      A tuple of (transport_coeffs, component_state), where transport_coeffs
      contains the transport coefficients and component_state is passed
      through unchanged.
    """
    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    n_faces = len(geo.rho_face_norm)
    # Discard fields that aren't needed for the TGLF calculation.
    local_settings_dict = dataclasses.asdict(tglf_inputs)
    valid_keys = tglf_defaults.TGLF_DEFAULTS.keys()
    local_settings_dict = {
        key: val
        for key, val in local_settings_dict.items()
        if key in valid_keys
    }
    global_settings_dict = dict(transport_runtime_params.tglf_settings)

    def callback(local_settings_dict):
      electron_heat_flux_array = np.zeros((n_faces,))
      electron_particle_flux_array = np.zeros((n_faces,))
      total_ion_heat_flux_array = np.zeros((n_faces,))

      future_list = [
          self.executor.submit(
              _run_single_tglf,
              i,
              local_settings_dict,
              global_settings_dict,
          )
          for i in range(n_faces)
      ]
      for future in futures.as_completed(future_list):
        i, electron_particle_flux, electron_heat_flux, total_ion_heat_flux = (
            future.result()
        )
        electron_particle_flux_array[i] = electron_particle_flux
        electron_heat_flux_array[i] = electron_heat_flux
        total_ion_heat_flux_array[i] = total_ion_heat_flux

      return (
          electron_heat_flux_array,
          total_ion_heat_flux_array,
          electron_particle_flux_array,
      )

    face_array_shape_dtype = jax.ShapeDtypeStruct(
        shape=(n_faces,), dtype=jax_utils.get_dtype()
    )
    result_shape_dtypes = (
        face_array_shape_dtype,
        face_array_shape_dtype,
        face_array_shape_dtype,
    )

    # We must pass local_settings_dict explicitly as an argument (*args) to
    # jax.pure_callback so JAX concretizes the abstract tracer arrays into
    # real NumPy arrays before invoking callback().
    electron_heat_flux, ion_heat_flux, electron_particle_flux = (
        jax.pure_callback(
            callback,
            result_shape_dtypes,
            local_settings_dict,
        )
    )

    return (
        self._make_transport_coeffs(
            electron_heat_flux_GB=electron_heat_flux,
            ion_heat_flux_GB=ion_heat_flux,
            electron_particle_flux_GB=electron_particle_flux,
            tglf_inputs=tglf_inputs,
            transport=transport_runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            two_point_mask=two_point_mask,
        ),
        component_state,
    )


class TGLFTransportModelConfig(
    tglf_based_transport_model.TGLFBasedTransportModelConfig
):
  r"""Model for the TGLF transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'tglf'.
    n_processes: Set number of parallel TGLF calculations to run.
    n_cores_per_process: Number of cores to use for each parallel TGLF
      calculation.
    tglf_settings: Dictionary of TGLF namelist parameters.
  """

  model_name: Annotated[Literal['tglf'], torax_pydantic.JAX_STATIC] = 'tglf'
  n_processes: pydantic.PositiveInt = 8
  n_cores_per_process: pydantic.PositiveInt = 2
  tglf_settings: Annotated[
      dict[str, TGLFSettingsValueTypes], torax_pydantic.JAX_STATIC
  ] = pydantic.Field(default_factory=dict)

  def build_transport_model(self) -> TGLFTransportModel:
    mp_context = multiprocessing.get_context('spawn')
    return TGLFTransportModel(
        executor=futures.ProcessPoolExecutor(
            max_workers=self.n_processes,
            mp_context=mp_context,
        )
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        n_processes=self.n_processes,
        n_cores_per_process=self.n_cores_per_process,
        tglf_settings=self.tglf_settings,
        **vars(base_params),
    )
