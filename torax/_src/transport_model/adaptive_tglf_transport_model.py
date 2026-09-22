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
"""Active Learning Adaptive TGLF transport model coupling surrogate with TGLF."""

from collections.abc import Callable, Mapping
from concurrent import futures
import dataclasses
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.data_harvesting import compute_solver_fingerprint
from torax._src.data_harvesting import StagingSink
from torax._src.geometry import geometry
from torax._src.physics import adaptive_physics_module
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model.tglf import defaults as tglf_defaults
from torax._src.transport_model.tglf import tglf_transport_model
from torax._src.transport_model.tglf import tglf2py
from typing_extensions import override

# Type alias for high-fidelity evaluation worker:
# (i, local_tglf_settings, global_tglf_settings) -> (i, Gamma_e, Q_e, Q_i)
HighFidelitySolverFn = Callable[
    [int, dict[str, Any], dict[str, Any]],
    tuple[int, float, float, float],
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for AdaptiveTGLFTransportModel."""

  tglf_settings: tuple[tuple[str, Any], ...] = ()
  output_directory: str = "/tmp/torax_tglf_runs"
  uncertainty_threshold: float = 0.20
  fallback_mode: str = "full_profile"  # 'full_profile' or 'per_face'
  smoothing_sigma: float = 0.05
  enable_data_harvesting: bool = True
  harvest_output_dir: str = "/tmp/torax_harvest"


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class AdaptiveTGLFTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Adaptive TGLF transport model coupling surrogate NN inference with TGLF fallback.

  Evaluates fast neural network surrogate models (such as TGLFNN) and calculates
  predictive uncertainty. If the uncertainty exceeds a user-configured threshold,
  it falls back to the high-fidelity TGLF solver (either full profile or per-face
  spliced with Gaussian smoothing) and optionally stages the evaluation pairs into
  a local Parquet dataset for active learning surrogate retraining.
  """

  surrogate_model: tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel
  executor: futures.Executor = dataclasses.field(metadata={"hash_by_id": True})
  sink: StagingSink | None = dataclasses.field(
      default=None, metadata={"hash_by_id": True}
  )
  high_fidelity_solver_fn: HighFidelitySolverFn | None = dataclasses.field(
      default=None, metadata={"hash_by_id": True}
  )

  @override
  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.ComponentRuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    assert isinstance(transport_runtime_params, RuntimeParams)

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    n_faces = len(geo.rho_face_norm)

    # 1. Fast surrogate prediction with uncertainty
    surrogate_means, _ = self.surrogate_model.predict_with_uncertainty(
        tglf_inputs
    )
    relative_uncertainty = self.surrogate_model.compute_relative_uncertainty(
        tglf_inputs
    )

    # 2. Extract settings for TGLF solver fallback
    local_settings_dict = dataclasses.asdict(tglf_inputs)
    valid_keys = tglf_defaults.TGLF_DEFAULTS.keys()
    filtered_local_settings = {
        k: v for k, v in local_settings_dict.items() if k in valid_keys
    }
    global_settings_dict = dict(transport_runtime_params.tglf_settings)

    # 3. Solver fingerprint for data harvesting
    fingerprint = compute_solver_fingerprint(
        model_name="tglf",
        settings=global_settings_dict,
    )

    # 4. Active learning engine configuration
    engine_config = adaptive_physics_module.AdaptivePhysicsConfig(
        uncertainty_threshold=float(
            transport_runtime_params.uncertainty_threshold
        ),
        fallback_mode=transport_runtime_params.fallback_mode,  # type: ignore
        smoothing_sigma=float(transport_runtime_params.smoothing_sigma),
        enable_data_harvesting=bool(
            transport_runtime_params.enable_data_harvesting
        ),
    )
    sink = self.sink or StagingSink(
        output_dir=transport_runtime_params.harvest_output_dir
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(
        config=engine_config,
        sink=sink,
    )

    solver_fn = (
        self.high_fidelity_solver_fn
        or tglf_transport_model._run_single_tglf
    )

    def callback(
        local_dict: dict[str, np.ndarray],
        rel_unc: np.ndarray,
        surr_efe: np.ndarray,
        surr_efi: np.ndarray,
        surr_pfi: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      needs_fallback, run_mask = engine.decide_fallback(rel_unc)

      if not needs_fallback:
        # Zero expensive TGLF runs executed
        return surr_efe, surr_efi, surr_pfi

      # Fallback required on indicated faces
      faces_to_run = [i for i in range(n_faces) if run_mask[i]]

      tglf_efe = np.copy(surr_efe)
      tglf_efi = np.copy(surr_efi)
      tglf_pfi = np.copy(surr_pfi)

      submitted_futures = [
          self.executor.submit(
              solver_fn,
              i,
              local_dict,
              global_settings_dict,
          )
          for i in faces_to_run
      ]

      for fut in futures.as_completed(submitted_futures):
        i, pfi, efe, efi = fut.result()
        tglf_pfi[i] = pfi
        tglf_efe[i] = efe
        tglf_efi[i] = efi

      # Fuse and smooth
      final_efe = engine.fuse_and_smooth(surr_efe, tglf_efe, run_mask, coords)
      final_efi = engine.fuse_and_smooth(surr_efi, tglf_efi, run_mask, coords)
      final_pfi = engine.fuse_and_smooth(surr_pfi, tglf_pfi, run_mask, coords)

      # Data harvesting dispatch
      if engine_config.enable_data_harvesting:
        engine.harvest_if_enabled(
            fingerprint=fingerprint,
            inputs=local_dict,
            high_fidelity_outputs={
                "efe_gb": tglf_efe,
                "efi_gb": tglf_efi,
                "pfi_gb": tglf_pfi,
            },
            uncertainties={"rel_unc": rel_unc},
        )

      return final_efe, final_efi, final_pfi

    face_struct = jax.ShapeDtypeStruct(
        shape=(n_faces,), dtype=jax_utils.get_dtype()
    )
    result_shape_dtypes = (face_struct, face_struct, face_struct)

    electron_heat_flux, ion_heat_flux, electron_particle_flux = (
        jax.pure_callback(
            callback,
            result_shape_dtypes,
            filtered_local_settings,
            relative_uncertainty,
            surrogate_means["efe_gb"],
            surrogate_means["efi_gb"],
            surrogate_means["pfi_gb"],
            geo.rho_face_norm,
        )
    )

    return self._make_core_transport(
        electron_heat_flux_GB=electron_heat_flux,
        ion_heat_flux_GB=ion_heat_flux,
        electron_particle_flux_GB=electron_particle_flux,
        tglf_inputs=tglf_inputs,
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
