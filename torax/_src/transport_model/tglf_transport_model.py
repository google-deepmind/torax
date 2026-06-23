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

"""A transport model that calls TGLF.

Used for generating ground truth for surrogate model evaluations.
"""

import dataclasses
import datetime
from multiprocessing import pool
import os
import subprocess
from typing import Annotated
from typing import Any, Literal, TypeAlias
import uuid

from absl import logging
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import quasilinear_transport_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import tglf_defaults
from torax._src.transport_model import transport_model

# Internal import.

# pylint: disable=invalid-name


# TODO(b/434175938): remove support for parsing via the legacy config kwargs.
# Most of these are just lower->upper but some have _ in different places.
_OLD_CONFIG_MAPPING = {
    'kygrid_model': 'KYGRID_MODEL',
    'ky': 'KY',
    'n_ky': 'NKY',
    'n_modes': 'NMODES',
    'geometry_flag': 'GEOMETRY_FLAG',
    'sat_rule': 'SAT_RULE',
    'xnu_model': 'XNU_MODEL',
    'n_width': 'NWIDTH',
    'width_min': 'WIDTH_MIN',
    'width': 'WIDTH',
    'filter': 'FILTER',
    'theta_trapped': 'THETA_TRAPPED',
    'w_dia_trapped': 'WDIA_TRAPPED',
    'sign_bt': 'SIGN_BT',
    'sign_it': 'SIGN_IT',
    'xnu_factor': 'XNU_FACTOR',
    'debye_factor': 'DEBYE_FACTOR',
    'etg_factor': 'ETG_FACTOR',
    'find_width': 'FIND_WIDTH',
    'use_mhd_rule': 'USE_MHD_RULE',
    'use_bpar': 'USE_BPAR',
    'use_bper': 'USE_BPER',
    'use_inboard_detrapped': 'USE_INBOARD_DETRAPPED',
    'use_ave_ion_grid': 'USE_AVE_ION_GRID',
    'alpha_e': 'ALPHA_E',
    'alpha_zf': 'ALPHA_ZF',
    'alpha_quench': 'ALPHA_QUENCH',
    'n_xgrid': 'NXGRID',
    'n_basis_min': 'NBASIS_MIN',
    'n_basis_max': 'NBASIS_MAX',
}

TGLFSettingsValueTypes: TypeAlias = str | float | int | bool | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for the TGLF transport model."""

  n_processes: int
  n_cores_per_process: int
  verbose: bool
  tglf_settings: dict[str, TGLFSettingsValueTypes] = dataclasses.field(
      metadata={'static': True}
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class TGLFTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculates turbulent transport coefficients with TGLF."""

  tglf_exec_path: str = '~/tglf'
  output_directory: str = '/tmp/torax_tglf_runs'

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model.TurbulentTransport:
    """Calculates several transport coefficients simultaneously.

    Args:
      transport_runtime_params: Input runtime parameters for this transport
        model.
      runtime_params: Input runtime parameters for all components of the
        simulation at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: transport coefficients
    """
    del pedestal_model_output  # Unused.

    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
    )

    def callback(
        tglf_inputs: tglf_based_transport_model.TGLFInputs,
        transport_runtime_params: RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> transport_model.TurbulentTransport:
      return self._run_tglf(
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          n_processes=transport_runtime_params.n_processes,
          n_cores_per_process=transport_runtime_params.n_cores_per_process,
          verbose=transport_runtime_params.verbose,
          core_profiles=core_profiles,
      )

    face_array_shape_dtype = jax.ShapeDtypeStruct(
        shape=(geo.torax_mesh.nx + 1,), dtype=jax_utils.get_dtype()
    )
    result_shape_dtypes = transport_model.TurbulentTransport(
        chi_face_ion=face_array_shape_dtype,
        chi_face_el=face_array_shape_dtype,
        d_face_el=face_array_shape_dtype,
        v_face_el=face_array_shape_dtype,
    )
    # Even though TGLF has side-effects (writing and reading from disk) we
    # still use a pure_callback here as:
    # 1. Nothing outside of this method depends on the side-effect.
    # 2. We don't mind if results are cached or recomputed.
    # 3. DCE will not happen here as we make use of the `core_transport` result.
    # This is based on the current implementation of pure_callback and JAX
    # may change the implementation making this not appropriate down the line.
    core_transport = jax.pure_callback(
        callback,
        result_shape_dtypes,
        tglf_inputs,
        transport_runtime_params,
        geo,
        core_profiles,
    )

    return core_transport

  def _run_tglf(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      n_processes: int,
      n_cores_per_process: int,
      verbose: bool = True,
  ) -> transport_model.TurbulentTransport:
    """Runs TGLF using command line tools. Loose coupling with TORAX.

    Args:
      tglf_inputs: Precomputed physics data.
      transport: Runtime parameters for the transport model.
      geo: TORAX geometry object.
      core_profiles: TORAX core profiles object.
      n_processes: Number of processes to run in parallel.
      n_cores_per_process: Number of cores to use for each TGLF process.
      verbose: If True, print the output of each TGLF process.

    Returns:
      core_transport: The core transport coefficients calculated by TGLF.
    """
    # Generate a unique directory for this TGLF plan.
    # Include UUID to prevent collisions when multiple simulations start
    # simultaneously.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:8]
    unique_suffix = f'uuid_{short_uuid}'
    # Add SLURM job ID to the unique suffix if running on SLURM.
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
      unique_suffix = f'job_{slurm_job_id}_{unique_suffix}'
    plan_directory = os.path.join(
        self.output_directory,
        f'torax_tglf_run_{timestamp}_{unique_suffix}',
    )
    if not os.path.exists(plan_directory):
      os.makedirs(plan_directory)

    def _run_tglf_single(
        face_index: int,
    ) -> tuple[float, float, float]:
      """Execute a single TGLF run.

      Args:
        face_index: The index of the face to run TGLF for.

      Returns:
        A tuple of (electron_heat_flux_GB, ion_heat_flux_GB,
        electron_particle_flux_GB).
      """
      # Create a unique directory for this TGLF run.
      label = f'tglf_run_{face_index:04d}'
      output_directory = os.path.join(plan_directory, label)
      if not os.path.exists(output_directory):
        os.makedirs(output_directory)

      # Extract the TGLFInputs for this face.
      tglf_inputs_i = jax.tree.map(
          lambda x: x[face_index] if jnp.ndim(x) > 0 else x, tglf_inputs
      )

      # Create the TGLF input namelist, merging the TGLF settings from the
      # transport runtime params with the data from tglf_inputs.
      tglf_namelist = transport.tglf_settings.copy()
      tglf_namelist.update(dataclasses.asdict(tglf_inputs_i))
      # Drop fields that are inherited from the QuasilinearInputs.
      excluded_fields = {
          f.name
          for f in dataclasses.fields(
              quasilinear_transport_model.QuasilinearInputs
          )
      }
      # Drop fields that are used for denormalization of the outputs
      excluded_fields.update({'Q_GB', 'GAMMA_GB'})
      tglf_namelist = {
          k: v for k, v in tglf_namelist.items() if k not in excluded_fields
      }
      namelist_str = '\n'.join([f'{k}={v}' for k, v in tglf_namelist.items()])
      with open(output_directory + '/input.tglf', 'w+') as f:
        f.write(namelist_str)

      # Run TGLF in the given working directory.
      result = subprocess.run(
          [
              str(self.tglf_exec_path),
              '-n',
              str(n_cores_per_process),
              '-e',
              label,
          ],
          capture_output=verbose,
          text=verbose,
          stdout=None if verbose else subprocess.DEVNULL,
          stderr=None if verbose else subprocess.DEVNULL,
          cwd=plan_directory,
          check=True,  # Raise error if the command fails.
      )

      if verbose:
        subprocess_output = result.stdout
        if result.stderr:
          subprocess_output += result.stderr
        logging.info('TGLF face %s output:\n%s', face_index, subprocess_output)

      # Read the TGLF output.
      gbfluxes = np.fromfile(
          os.path.join(output_directory, 'out.tglf.gbflux'),
          sep=' ',
      )
      nspecies = len(gbfluxes) // 4
      tglf_elec_eflux_out = float(gbfluxes[1 * nspecies + 0])
      tglf_ion1_eflux_out = float(
          sum(gbfluxes[1 * nspecies + 1 : 1 * nspecies + nspecies])
      )
      tglf_elec_pflux_out = float(gbfluxes[0 * nspecies + 0])

      return (
          tglf_elec_eflux_out,
          tglf_ion1_eflux_out,
          tglf_elec_pflux_out,
      )

    cell_indices = range(len(geo.torax_mesh.face_centers))
    with pool.ThreadPool(processes=n_processes) as thread_pool:
      subprocess_outputs = thread_pool.map(_run_tglf_single, cell_indices)

    tglf_elec_eflux_out, tglf_ion1_eflux_out, tglf_elec_pflux_out = np.array(
        subprocess_outputs
    ).T

    return self._make_core_transport(
        electron_heat_flux_GB=tglf_elec_eflux_out,
        ion_heat_flux_GB=tglf_ion1_eflux_out,
        electron_particle_flux_GB=tglf_elec_pflux_out,
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


class TGLFTransportModelConfig(pydantic_model_base.TransportBase):
  r"""Model for the TGLF transport model.

  TGLF settings used to be passed as kwargs to the constructor of this
  class. This behavior is now deprecated in favour of setting TGLF parameters
  via the tglf_settings dictionary. The following options can still be parsed
  from kwargs for backwards compatibility:
    kygrid_model: 0 = user-defined with n_ky points equally spaced with kymin =
      ky/n_ky. 1 = standard ky spectrum for SAT0 and SAT1 with kymin=0.1/rho_i.
      4 = standard ky spectrum with more low ky points and
      kymin=0.05*grad_r0/rho_i.
    ky: Specify wavenumber for single wavenumber call, or set user-defined ky
      grid with kygrid_model=0.
    n_ky: Number of ky points with kygrid_model=0, else number of additional
      logarithmically equally spaced points within 1 < ky < 24 when using
      kygrid_model=4.
    n_modes: Number of eigenmodes to track, advise to use num_species+2 for
      efficiency.
    geometry_flag: 0 = s-alpha, 1 = Miller/MXH, 2 = Fourier, 3 = ELITE.
    sat_rule: Specify quasilinear saturation rule used to compute transport
      fluxes.
    xnu_model: Specify collision model. 2 = default for SAT0 and SAT1, 3 =
      default for SAT2 and SAT3.
    n_width: Maximum number of mode widths in mode width scan.
    width_min: Minimum value for mode width scan, set negative for
      electromagnetic search.
    width: Maximum value for mode width scan.
    filter: Set frequency threshold to filter out non-drift-wave instabilities.
    theta_trapped: Adjustment parameter for trapped fraction model. Set to 0.4
      with n_basis_max = 8 for low aspect ratio
      (https://eprints.whiterose.ac.uk/159700/).
    w_dia_trapped: Non-standard option. Set to 1.0 for SAT2 and SAT3.
    sign_bt: Sign of toroidal field, positive = CCW from the top.
    sign_it: Sign of toroidal current, positive = CCW from the top.
    xnu_factor: Multiplier for the trapped/passing boundary collision terms, not
      the same as collisionality_multiplier.
    debye_factor: Multiplier for the normalized Debye length.
    etg_factor: Exponent for the ETG saturation rule.
    find_width: Toggle mode width scan for maximum growth rate search, uses
      width argument if False.
    use_mhd_rule: If True, ignore pressure gradient contribution to curvature
      drift. Recommended to set False for high beta.
    use_bpar: If True, include compressional magnetic fluctuations,
      :math:`\delta B_{\par}`.
    use_bper: If true, include transverse magnetic fluctuations, :math:`\delta
      B_{\perp}`.
    use_inboard_detrapped: If True, set trapped fraction to zero if eigenmode is
      inward ballooning.
    use_ave_ion_grid: If True, use average ion gyroradius as opposed to main ion
      gyroradius
    alpha_e: Multiplier for ExB velocity shear for spectral shift model.
    alpha_zf: Zonal flow mixing coefficient. If -1.0, toggles switch that avoids
      picking lowest ky as maximum gamma/ky for intensity spectrum shape in
      quasilinear saturation rules.
    alpha_quench: 0.0 = use spectral shift model, 1.0 = use quench rule.
    n_xgrid: Number of nodes in Gauss-Hermite quadrature. Recommended to use 4 *
      n_basis_max
    n_basis_min: Minimum number of parallel basis functions (Hermite
      polynomials) used to find mode width.
    n_basis_max: Maximum number of parallel basis functions (Hermite
      polynomials) used to find mode width. Recommended 4 for SAT0 and  SAT1, 6
      for SAT2 and SAT3.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'tglf'.
    tglf_exec_path: Path to the TGLF executable.
    output_directory: Path to output directory for temp files.
    n_processes: Set number of parallel TGLF calculations to run.
    n_cores_per_process: Number of cores to use for each parallel TGLF
      calculation.
    verbose: Whether to enable verbose logging for TGLF subprocess runs.
    use_rotation: Toggles the use of rotation shear model.
    rotation_multiplier: Multiplier for the input rotation shear.
    DV_effective: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
    collisionality_multiplier: Collisionality multiplier.
    tglf_settings: Dictionary of TGLF namelist parameters.
    use_legacy_torax_defaults: If True, use legacy TORAX defaults for TGLF
      parameters. Otherwise, use the defaults distributed with TGLF. Note that
      in a future release, this option will be removed and the defaults will be
      those distributed with TGLF.
  """

  model_name: Annotated[Literal['tglf'], torax_pydantic.JAX_STATIC] = 'tglf'
  tglf_exec_path: Annotated[str, torax_pydantic.JAX_STATIC] = '~/tglf'
  output_directory: Annotated[str, torax_pydantic.JAX_STATIC] = (
      '/tmp/torax_tglf_runs'
  )
  n_processes: pydantic.PositiveInt = 8
  n_cores_per_process: pydantic.PositiveInt = 2
  verbose: bool = False
  use_rotation: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  rotation_multiplier: pydantic.NonNegativeFloat = 1.0
  DV_effective: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  An_min: pydantic.PositiveFloat = 0.05
  collisionality_multiplier: float = 1.0
  tglf_settings: Annotated[
      dict[str, TGLFSettingsValueTypes], torax_pydantic.JAX_STATIC
  ] = pydantic.Field(default_factory=dict)
  use_legacy_torax_defaults: bool = True

  # TODO(b/434175938): remove support for parsing via the legacy config kwargs
  # and remove use_legacy_torax_defaults.
  @pydantic.model_validator(mode='before')
  @classmethod
  def _validate_tglf_settings(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Parses tglf_settings combining defaults and old config kwargs."""
    if data.get('model_name', '') != 'tglf':
      return data

    if data.get('use_legacy_torax_defaults', True):
      logging.warning(
          'use_legacy_torax_defaults=True is deprecated. This flag uses'
          ' TORAX-specific defaults for TGLF settings. In future, the defaults'
          ' will be those distributed with TGLF and this flag will be removed.'
      )
      tglf_settings = dict(tglf_defaults.LEGACY_TORAX_TGLF_DEFAULTS)
    else:
      tglf_settings = dict(tglf_defaults.TGLF_DEFAULTS)

    # Merge settings from the legacy config parameters.
    legacy_config_used = False
    for old_key, new_key in _OLD_CONFIG_MAPPING.items():
      if old_key in data:
        legacy_config_used = True
        val = data.pop(old_key)
        if isinstance(val, bool):
          val = '.true.' if val else '.false.'
        tglf_settings[new_key] = val
    if legacy_config_used:
      logging.warning(
          'Parsing TGLF settings from the legacy config kwargs is deprecated. '
          'Please use the tglf_settings dictionary instead.'
      )

    # Merge settings from the user-provided tglf_settings dictionary.
    user_tglf_settings = data.pop('tglf_settings', {})
    for k, v in user_tglf_settings.items():
      if isinstance(v, bool):
        v = '.true.' if v else '.false.'
      tglf_settings[k] = v

    data['tglf_settings'] = tglf_settings
    return data

  def build_transport_model(self) -> TGLFTransportModel:
    return TGLFTransportModel(
        tglf_exec_path=self.tglf_exec_path,
        output_directory=self.output_directory,
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        n_processes=self.n_processes,
        n_cores_per_process=self.n_cores_per_process,
        verbose=self.verbose,
        use_rotation=self.use_rotation,
        rotation_multiplier=self.rotation_multiplier,
        DV_effective=self.DV_effective,
        collisionality_multiplier=self.collisionality_multiplier,
        An_min=self.An_min,
        tglf_settings=self.tglf_settings,
        **base_kwargs,
    )
