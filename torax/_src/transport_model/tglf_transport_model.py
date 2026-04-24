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
from typing import Any, Literal, Mapping, Sequence
import uuid

from absl import logging
import chex
import jax
import numpy as np
import pydantic
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model

# Internal import.

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for the TGLF transport model."""

  n_processes: int
  n_cores_per_process: int
  verbose: bool
  kygrid_model: int
  ky: float
  n_ky: int
  n_modes: int
  geometry_flag: int
  sat_rule: int
  xnu_model: int
  n_width: float
  width_min: float
  width: float
  filter: float
  theta_trapped: float
  w_dia_trapped: float
  sign_bt: float
  sign_it: float
  xnu_factor: float
  debye_factor: float
  etg_factor: float
  find_width: bool
  use_mhd_rule: bool
  use_bpar: bool
  use_bper: bool
  use_inboard_detrapped: bool
  use_ave_ion_grid: bool
  alpha_e: float
  alpha_zf: float
  alpha_quench: float
  n_xgrid: int
  n_basis_min: int
  n_basis_max: int


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

    def callback(tglf_inputs, transport_runtime_params, geo, core_profiles):
      tglf_plan = _extract_tglf_plan(
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )
      plan_output_directory = self._run_tglf(
          tglf_plan=tglf_plan,
          n_processes=transport_runtime_params.n_processes,
          n_cores_per_process=transport_runtime_params.n_cores_per_process,
          verbose=transport_runtime_params.verbose,
      )
      core_transport = self._extract_run_data(
          tglf_plan=tglf_plan,
          plan_output_directory=plan_output_directory,
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )
      return core_transport

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
      tglf_plan: Sequence[dict[str, Any]],
      n_processes: int,
      n_cores_per_process: int,
      verbose: bool = True,
  ) -> str:
    """Runs TGLF using command line tools. Loose coupling with TORAX.

    Args:
      tglf_plan: List of TGLF input dictionaries.
      n_processes: Number of processes to run in parallel.
      n_cores_per_process: Number of cores to use for each TGLF process.
      verbose: If True, print the output of each TGLF process.

    Returns:
      Path to the directory containing the TGLF output files.
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

    run_directories = []
    for run in tglf_plan:
      # Create a directory for each individual TGLF run.
      run_directory = os.path.join(plan_directory, run['label'])
      if not os.path.exists(run_directory):
        os.makedirs(run_directory)

      # Write TGLF input file for this run.
      assert isinstance(run['inputs'], dict)
      fstr = '\n'.join([f'{k}={v}' for k, v in run['inputs'].items()])
      with open(run_directory + '/input.tglf', 'w+') as f:
        f.write(fstr)
      run_directories.append(run_directory)

    def _run_tglf_single(run_directory: str) -> str | None:
      """Execute a single TGLF run."""
      result = subprocess.run(
          # Run TGLF in the given working directory.
          [
              str(self.tglf_exec_path),
              '-n',
              str(n_cores_per_process),
              '-e',
              run_directory,
          ],
          capture_output=verbose,
          text=verbose,
          stdout=None if verbose else subprocess.DEVNULL,
          stderr=None if verbose else subprocess.DEVNULL,
          # Run from the plan directory to avoid issues with relative paths.
          cwd=plan_directory,
          check=True,  # Raise error if the command fails.
      )

      if verbose:
        subprocess_output = result.stdout
        if result.stderr:
          subprocess_output += result.stderr

        return subprocess_output
      return None

    with pool.ThreadPool(processes=n_processes) as thread_pool:
      subprocess_outputs = thread_pool.map(_run_tglf_single, run_directories)
    if verbose:
      for subprocess_output in subprocess_outputs:
        logging.info(subprocess_output)

    return plan_directory

  def _extract_run_data(
      self,
      tglf_plan: Sequence[Mapping[str, Any]],
      plan_output_directory: str,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_model.TurbulentTransport:
    """Constructs TORAX core transport object from TGLF output.

    Args:
      tglf_plan: List of TGLF input dictionaries.
      plan_output_directory: Directory containing a subdirectory for each TGLF
        run.
      tglf_inputs: Precomputed physics data.
      transport: Runtime parameters for the transport model.
      geo: TORAX geometry object.
      core_profiles: TORAX CoreProfiles object.

    Returns:
      TORAX core transport object.
    """
    qe = np.zeros((len(tglf_plan),))
    qi = np.zeros((len(tglf_plan),))
    ge = np.zeros((len(tglf_plan),))
    for i, run in enumerate(tglf_plan):
      # np.fromfile is more efficient than np.loadtxt for reading large files
      # with a consistent format.
      gbfluxes = np.fromfile(
          os.path.join(plan_output_directory, run['label'], 'out.tglf.gbflux'),
          sep=' ',
      )
      nspecies = len(gbfluxes) // 4
      # TGLF species 1 is electrons
      qe[i] = float(gbfluxes[1 * nspecies + 0])
      qi[i] = float(sum(gbfluxes[1 * nspecies + 1 : 1 * nspecies + nspecies]))
      ge[i] = float(gbfluxes[0 * nspecies + 0])

    return self._make_core_transport(
        electron_heat_flux_GB=qe,
        ion_heat_flux_GB=qi,
        electron_particle_flux_GB=ge,
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


def _extract_tglf_plan(
    tglf_inputs: tglf_based_transport_model.TGLFInputs,
    transport: RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> Sequence[Mapping[str, Any]]:
  """Converts TORAX parameters to TGLF input dictionary.

  Args:
      tglf_inputs: Precomputed physics data.
      transport: Runtime parameters for the qualikiz transport model.
      geo: TORAX geometry object.
      core_profiles: TORAX CoreProfiles object, containing time-evolvable
        quantities like q

  Returns:
      A list of dictionaries containing TGLF input namelists, one for each
      radial grid cell.
  """

  species_template = {
      'ZS': None,
      'MASS': None,
      'RLNS': None,
      'RLTS': None,
      'TAUS': None,
      'AS': None,
      'VPAR': 0.0,
      'VPAR_SHEAR': 0.0,
  }
  tglf_input_template = {
      # Control
      'UNITS': 'CGYRO',
      'NS': 3,
      'USE_TRANSPORT_MODEL': '.true.',
      'GEOMETRY_FLAG': transport.geometry_flag,
      'USE_BPER': '.true.' if transport.use_bper else '.false.',
      'USE_BPAR': '.true.' if transport.use_bpar else '.false.',
      'USE_BISECTION': '.true.',
      'USE_MHD_RULE': '.true.' if transport.use_mhd_rule else '.false.',
      'USE_INBOARD_DETRAPPED': (
          '.true.' if transport.use_inboard_detrapped else '.false.'
      ),
      'USE_AVE_ION_GRID': '.true.' if transport.use_ave_ion_grid else '.false.',
      'SAT_RULE': transport.sat_rule,
      'KYGRID_MODEL': transport.kygrid_model,
      'XNU_MODEL': transport.xnu_model,
      'VPAR_MODEL': 0,
      'SIGN_BT': transport.sign_bt,
      'SIGN_IT': transport.sign_it,
      'KY': transport.ky,
      'NEW_EIKONAL': '.true.',
      'VEXB': 0.0,
      'VEXB_SHEAR': 0.0,
      'BETAE': 0.0,
      'XNUE': 0.0,
      'ZEFF': 1.0,
      'DEBYE': 0.0,
      'IFLUX': '.true.',
      'IBRANCH': -1,
      'NMODES': transport.n_modes,
      'NBASIS_MIN': transport.n_basis_min,
      'NBASIS_MAX': transport.n_basis_max,
      'NXGRID': transport.n_xgrid,
      'NKY': transport.n_ky,
      'ADIABATIC_ELEC': '.false.',
      # Multipliers
      'ALPHA_P': 1.0,
      'ALPHA_MACH': 0.0,
      'ALPHA_E': transport.alpha_e,
      'ALPHA_QUENCH': transport.alpha_quench,
      'ALPHA_ZF': transport.alpha_zf,
      'XNU_FACTOR': transport.xnu_factor,
      'DEBYE_FACTOR': transport.debye_factor,
      'ETG_FACTOR': transport.etg_factor,
      'B_MODEL_SA': 1,
      'FT_MODEL_SA': 1,
      # Gaussian mode width search
      'WRITE_WAVEFUNCTION_FLAG': 0,
      'WIDTH_MIN': transport.width_min,
      'WIDTH': transport.width,
      'NWIDTH': transport.n_width,
      'FIND_WIDTH': '.true.' if transport.find_width else '.false.',
      # Miller shape parameters
      'RMIN_LOC': 0.5,
      'RMAJ_LOC': 3.0,
      'ZMAJ_LOC': 0.0,
      'Q_LOC': 2.0,
      'Q_PRIME_LOC': 16.0,
      'P_PRIME_LOC': 0.0,
      'DRMINDX_LOC': 1.0,
      'DRMAJDX_LOC': 0.0,
      'DZMAJDX_LOC': 0.0,
      'KAPPA_LOC': 1.0,
      'S_KAPPA_LOC': 0.0,
      'DELTA_LOC': 0.0,
      'S_DELTA_LOC': 0.0,
      'ZETA_LOC': 0.0,
      'S_ZETA_LOC': 0.0,
      'KX0_LOC': 0.0,
      # Expert options
      'THETA_TRAPPED': transport.theta_trapped,
      'PARK': 1.0,
      'GHAT': 1.0,
      'GCHAT': 1.0,
      'WD_ZERO': 0.1,
      'LINSKER_FACTOR': 0.0,
      'GRADB_FACTOR': 0.0,
      'FILTER': transport.filter,
      'DAMP_PSI': 0.0,
      'DAMP_SIG': 0.0,
      'WDIA_TRAPPED': transport.w_dia_trapped,
  }

  for species_number in range(1, 4):
    tglf_input_template.update({
        f'{key}_{species_number}': value
        for key, value in species_template.items()
    })

  tglf_plan = []
  zi0 = np.array(core_profiles.Z_i_face)
  ai0 = np.array(core_profiles.A_i)
  zi1 = np.array(core_profiles.Z_impurity_face)
  ai1 = np.array(core_profiles.A_impurity_face)
  for i, _ in enumerate(np.array(geo.rho_face_norm)):
    # Shallow copy is ok, as we are only modifying top-level fields.
    tglf_runpars = tglf_input_template.copy()
    tglf_runpars['BETAE'] = float(tglf_inputs.BETAE[i])
    tglf_runpars['XNUE'] = float(tglf_inputs.XNUE[i])
    tglf_runpars['ZEFF'] = float(tglf_inputs.ZEFF[i])
    tglf_runpars['DEBYE'] = float(tglf_inputs.DEBYE[i])
    tglf_runpars['RMIN_LOC'] = float(tglf_inputs.RMIN_LOC[i])
    tglf_runpars['RMAJ_LOC'] = float(tglf_inputs.RMAJ_LOC[i])
    tglf_runpars['Q_LOC'] = float(tglf_inputs.Q_LOC[i])
    tglf_runpars['Q_PRIME_LOC'] = float(tglf_inputs.Q_PRIME_LOC[i])
    tglf_runpars['P_PRIME_LOC'] = float(tglf_inputs.P_PRIME_LOC[i])
    tglf_runpars['DRMAJDX_LOC'] = float(tglf_inputs.DRMAJDX_LOC[i])
    tglf_runpars['KAPPA_LOC'] = float(tglf_inputs.KAPPA_LOC[i])
    tglf_runpars['S_KAPPA_LOC'] = float(tglf_inputs.S_KAPPA_LOC[i])
    tglf_runpars['DELTA_LOC'] = float(tglf_inputs.DELTA_LOC[i])
    tglf_runpars['S_DELTA_LOC'] = float(tglf_inputs.S_DELTA_LOC[i])
    tglf_runpars['ZS_1'] = -1.0
    tglf_runpars['MASS_1'] = float(
        constants.CONSTANTS.m_e
        / (constants.CONSTANTS.m_amu * constants.ION_PROPERTIES_DICT['D'].A)
    )
    tglf_runpars['RLNS_1'] = float(tglf_inputs.RLNS_1[i])
    tglf_runpars['RLTS_1'] = float(tglf_inputs.RLTS_1[i])
    tglf_runpars['TAUS_1'] = 1.0
    tglf_runpars['AS_1'] = 1.0
    tglf_runpars['ZS_2'] = float(zi0[i])
    tglf_runpars['MASS_2'] = float(ai0 / constants.ION_PROPERTIES_DICT['D'].A)
    tglf_runpars['RLNS_2'] = float(tglf_inputs.RLNS_2[i])
    tglf_runpars['RLTS_2'] = float(tglf_inputs.RLTS_2[i])
    tglf_runpars['TAUS_2'] = float(tglf_inputs.TAUS_2[i])
    tglf_runpars['AS_2'] = float(tglf_inputs.AS_2[i])
    tglf_runpars['ZS_3'] = float(zi1[i])
    tglf_runpars['MASS_3'] = float(
        ai1[i] / constants.ION_PROPERTIES_DICT['D'].A
    )
    tglf_runpars['RLNS_3'] = float(tglf_inputs.RLNS_3[i])
    tglf_runpars['RLTS_3'] = float(tglf_inputs.RLTS_3[i])
    tglf_runpars['TAUS_3'] = float(tglf_inputs.TAUS_3[i])
    tglf_runpars['AS_3'] = float(tglf_inputs.AS_3[i])
    tglf_plan.append({
        'inputs': tglf_runpars,
        'label': f'tglf_run_{i:04d}',
    })

  return tglf_plan


class TGLFTransportModelConfig(pydantic_model_base.TransportBase):
  r"""Model for the TGLF transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'tglf'.
    tglf_exec_path: Path to the TGLF executable.
    n_processes: Set number of parallel TGLF calculations to run.
    n_cores_per_process: Number of cores to use for each parallel TGLF
      calculation.
    DV_effective: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
    collisionality_multiplier: Collisionality multiplier.
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

  # Mode settings
  kygrid_model: pydantic.PositiveInt = 4
  ky: pydantic.PositiveFloat = 0.3
  n_ky: pydantic.PositiveInt = 19
  n_modes: pydantic.PositiveInt = 5

  # Model settings
  geometry_flag: pydantic.PositiveInt = 1
  sat_rule: pydantic.PositiveInt = 3
  xnu_model: pydantic.PositiveInt = 3
  n_width: pydantic.PositiveInt = 21
  width_min: pydantic.FiniteFloat = -0.3
  width: pydantic.PositiveFloat = 1.65
  filter: pydantic.FiniteFloat = 2.0
  theta_trapped: pydantic.PositiveFloat = 0.7
  w_dia_trapped: pydantic.PositiveFloat = 1.0
  sign_bt: pydantic.FiniteFloat = 1.0
  sign_it: pydantic.FiniteFloat = 1.0
  xnu_factor: pydantic.PositiveFloat = 1.0
  debye_factor: pydantic.PositiveFloat = 1.0
  etg_factor: pydantic.FiniteFloat = 1.25

  # Flags
  find_width: bool = True
  use_mhd_rule: bool = False
  use_bpar: bool = True
  use_bper: bool = False
  use_inboard_detrapped: bool = False
  use_ave_ion_grid: bool = False

  # Multipliers
  alpha_e: pydantic.FiniteFloat = 1.0
  alpha_zf: pydantic.FiniteFloat = 1.0
  alpha_quench: pydantic.FiniteFloat = 0.0

  # Numerical grid settings
  n_xgrid: pydantic.PositiveInt = 16
  n_basis_min: pydantic.PositiveInt = 2
  n_basis_max: pydantic.PositiveInt = 6

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
        kygrid_model=self.kygrid_model,
        ky=self.ky,
        n_ky=self.n_ky,
        n_modes=self.n_modes,
        geometry_flag=self.geometry_flag,
        sat_rule=self.sat_rule,
        xnu_model=self.xnu_model,
        n_width=self.n_width,
        width_min=self.width_min,
        width=self.width,
        filter=self.filter,
        theta_trapped=self.theta_trapped,
        w_dia_trapped=self.w_dia_trapped,
        sign_bt=self.sign_bt,
        sign_it=self.sign_it,
        xnu_factor=self.xnu_factor,
        debye_factor=self.debye_factor,
        etg_factor=self.etg_factor,
        find_width=self.find_width,
        use_mhd_rule=self.use_mhd_rule,
        use_bpar=self.use_bpar,
        use_bper=self.use_bper,
        use_inboard_detrapped=self.use_inboard_detrapped,
        use_ave_ion_grid=self.use_ave_ion_grid,
        alpha_e=self.alpha_e,
        alpha_zf=self.alpha_zf,
        alpha_quench=self.alpha_quench,
        n_xgrid=self.n_xgrid,
        n_basis_min=self.n_basis_min,
        n_basis_max=self.n_basis_max,
        **base_kwargs,
    )
