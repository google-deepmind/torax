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

"""A transport model that calls TGLF.

Used for generating ground truth for surrogate model evaluations.
"""

import dataclasses
import datetime
import os
import copy
import subprocess
import multiprocessing
import tempfile
from typing import Annotated
from typing import Literal
from typing import Sequence, Mapping, Any

import chex
import jax
import numpy as np
import pydantic
from torax._src import jax_utils
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  n_processes: int


_DEFAULT_TGLFRUN_NAME_PREFIX = 'torax_tglf_runs'


def _get_tglf_exec_path() -> str:
  #default_tglf_exec_path = '~/gacode/tglf/bin/tglf'
  default_tglf_exec_path = 'tglf'
  return os.environ.get('TORAX_TGLF_EXEC_PATH', default_tglf_exec_path)


class TGLFTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Calculates turbulent transport coefficients with TGLF."""

  def __init__(self):
    self._tglfrun_parentdir = tempfile.TemporaryDirectory()
    self._tglfrun_name = (
        _DEFAULT_TGLFRUN_NAME_PREFIX
        + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    self._runpath = os.path.join(self._tglfrun_parentdir.name, self._tglfrun_name)
    self._frozen = True

  def _call_implementation(
      self,
      transport_runtime_params: runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model.TurbulentTransport:
    """Calculates several transport coefficients simultaneously.

    Args:
      transport_runtime_params: Input runtime parameters for this
        transport model.
      runtime_params: Input runtime parameters for all components
        of the simulation at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: transport coefficients
    """
    del pedestal_model_output, runtime_params  # Unused.

    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )

    def callback(tglf_inputs, transport_runtime_params, geo, core_profiles):
      # Keep mapping to numpy arrays
      (tglf_inputs, transport_runtime_params, geo, core_profiles) = (
          jax.tree.map(
              np.asarray,
              (tglf_inputs, transport_runtime_params, geo, core_profiles),
          )
      )
      tglf_plan = _extract_tglf_plan(
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )
      self._run_tglf(tglf_plan, transport_runtime_params.n_processes)
      core_transport = self._extract_run_data(
          tglf_plan=tglf_plan,
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )
      return core_transport

    face_array_shape_dtype = jax.ShapeDtypeStruct(
        shape=(geo.torax_mesh.nx+1,), dtype=jax_utils.get_dtype()
    )
    result_shape_dtypes = transport_model.TurbulentTransport(
        chi_face_ion=face_array_shape_dtype,
        chi_face_el=face_array_shape_dtype,
        d_face_el=face_array_shape_dtype,
        v_face_el=face_array_shape_dtype,
    )
    # Even though tglf has side-effects (writing and reading from disk) we
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
      tglf_plan: Sequence[Mapping[str, Any]],
      n_processes: int,
      verbose: bool = True,
  ) -> None:
    """Runs QuaLiKiz using command line tools. Loose coupling with TORAX."""
    execution_path = _get_tglf_exec_path()

    os.makedirs(self._runpath, exist_ok=True)
    for run in tglf_plan:
        run.update({'execution_path': execution_path, 'run_path': self._runpath, 'verbose': verbose})
        path = os.path.join(self._runpath, run['location'])
        os.makedirs(path, exist_ok=True)
        fstr = "\n".join(["{}={}".format(k, v) for k, v in run['inputs'].items()])
        with open(path+'/input.tglf','w+') as f:
            f.write(fstr)

    ctx = multiprocessing.get_context('forkserver')
    queue = ctx.Queue()
    with ctx.Manager() as manager:
        queue = manager.Queue()
        with ctx.Pool(processes=n_processes) as pool:
            arguments = [(run, queue) for run in tglf_plan]
            _ = pool.starmap(_run_tglf_single, arguments)
        if verbose:
            for i in range(len(arguments)):
                print(queue.get())


  def _extract_run_data(
      self,
      tglf_plan: Sequence[Mapping[str, Any]],
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_model.TurbulentTransport:
    """Extracts QuaLiKiz run data from runpath."""

    def read_gbflux(path):
        # Read the out.tglf.gbflux file
        with open(os.path.join(path, 'out.tglf.gbflux'), 'r') as f:
            lines = f.readlines()
        species = []
        if lines:
            _fluxes = {
                'Gamma': 0.0,
                'Q': 0.0,
                'Pi': 0.0,
                'S': 0.0,
            }
            # Convert the line of values into a list of floats
            fluxes_list = [float(value) for value in lines[0].split()]
            nspecies = len(fluxes_list) // len(list(_fluxes.keys()))
            for i_species in range(nspecies):
                fluxes = copy.deepcopy(_fluxes)
                for i_flux, key_flux in enumerate(_fluxes.keys()):
                    fluxes[key_flux] = ((fluxes_list[i_flux*nspecies:(i_flux+1)*nspecies])[:nspecies])[i_species]
                species.append(copy.deepcopy(fluxes))
        return species

    qe = np.zeros((len(tglf_plan), ))
    qi = np.zeros((len(tglf_plan), ))
    ge = np.zeros((len(tglf_plan), ))
    for i, run in enumerate(tglf_plan):
        gbfluxes = np.loadtxt(os.path.join(self._runpath, run['location'], 'out.tglf.gbflux'))
        nspecies = len(gbfluxes) // 4
        #gbfluxes = read_gbflux(os.path.join(self._runpath, run['location']))
        qe[i] = float(gbfluxes[1*nspecies+0])  # Defined TGLF species 1 as electrons
        qi[i] = float(sum(gbfluxes[1*nspecies+1:1*nspecies+nspecies]))
        ge[i] = float(gbfluxes[0*nspecies+0])

    return self._make_core_transport(
        electron_heat_flux_GB=qe,
        ion_heat_flux_GB=qi,
        electron_particle_flux_GB=ge,
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        #gradient_reference_length=geo.R_major,
        #gyrobohm_flux_reference_length=geo.a_minor,
    )

  def __hash__(self) -> int:
    return hash(('TGLFTransportModel' + self._runpath))

  def __eq__(self, other) -> bool:
    return (
        isinstance(other, TGLFTransportModel)
        and self._runpath == other._runpath
    )


def _extract_tglf_plan(
    tglf_inputs: tglf_based_transport_model.TGLFInputs,
    transport: RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> Sequence[Mapping[str, Any]]:

  # pylint: disable=invalid-name
  species_template = {
      'ZS': None,
      'MASS': None,
      'RLNS': None,
      'RLTS': None,
      'TAUS': None,
      'AS': None,
      'VPAR': 0.0,
      'VPAR_SHEAR': 0.0,
      #'VNS_SHEAR': 0.0,
      #'VTS_SHEAR': 0.0,
  }
  tglf_input_template = {
      # control
      'UNITS': 'CGYRO',
      'NS': 3,
      'USE_TRANSPORT_MODEL': '.true.',
      'GEOMETRY_FLAG': 1,
      'USE_BPER': '.true.',
      'USE_BPAR': '.false.',
      'USE_BISECTION': '.true.',
      'USE_MHD_RULE': '.false.',
      'USE_INBOARD_DETRAPPED': '.false.',
      'SAT_RULE': 3,
      'KYGRID_MODEL': 4,
      'XNU_MODEL': 3,
      'VPAR_MODEL': 0,
      #'VPAR_SHEAR_MODEL': 0,
      'SIGN_BT': 1.0,
      'SIGN_IT': 1.0,
      'KY': 0.3,
      'NEW_EIKONAL': '.true.',
      'VEXB': 0.0,
      'VEXB_SHEAR': 0.0,
      'BETAE': 0.0,
      'XNUE': 0.0,
      'ZEFF': 1.0,
      'DEBYE': 0.0,
      'IFLUX': '.true.',
      'IBRANCH': -1,
      'NMODES': 5,
      'NBASIS_MAX': 6,  # Default is 4
      'NBASIS_MIN': 2,
      'NXGRID': 16,
      'NKY': 19,
      'ADIABATIC_ELEC': '.false.',
      'ALPHA_P': 1.0,
      'ALPHA_MACH': 0.0,
      'ALPHA_E': 1.0,
      'ALPHA_QUENCH': 0.0,
      'ALPHA_ZF': 1.0,
      'XNU_FACTOR': 1.0,
      'DEBYE_FACTOR': 1.0,
      'ETG_FACTOR': 1.25,
      'B_MODEL_SA': 1,
      'FT_MODEL_SA': 1,
      # gaussian
      'WRITE_WAVEFUNCTION_FLAG': 0,
      'WIDTH': 1.65,
      'WIDTH_MIN': 0.3,
      'NWIDTH': 21,
      'FIND_WIDTH': '.true.',
      # miller
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
      # expert
      'THETA_TRAPPED': 0.7,
      'PARK': 1.0,
      'GHAT': 1.0,
      'GCHAT': 1.0,
      'WD_ZERO': 0.1,
      'LINSKER_FACTOR': 0.0,
      'GRADB_FACTOR': 0.0,
      'FILTER': 2.0,
      'DAMP_PSI': 0.0,
      'DAMP_SIG': 0.0,
      #'NN_MAX_ERROR': -1.0,
      'WDIA_TRAPPED': 1.0,
  }

  for s in range(1, 4):
      tglf_input_template.update({k+f'_{s:d}': v for k, v in species_template.items()})

  tglf_plan = []
  zi0 = np.array(core_profiles.Z_i_face)
  ai0 = np.array(core_profiles.A_i)
  zi1 = np.array(core_profiles.Z_impurity_face)
  ai1 = np.array(core_profiles.A_impurity_face)
  for i, rho in enumerate(np.array(geo.rho_face_norm)):
      tglf_runpars = copy.deepcopy(tglf_input_template)
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
      tglf_runpars['MASS_1'] = 0.00027428995
      tglf_runpars['RLNS_1'] = float(tglf_inputs.RLNS_1[i])
      tglf_runpars['RLTS_1'] = float(tglf_inputs.RLTS_1[i])
      tglf_runpars['TAUS_1'] = 1.0
      tglf_runpars['AS_1'] = 1.0
      tglf_runpars['ZS_2'] = float(zi0[i])
      tglf_runpars['MASS_2'] = float(ai0 / constants.ION_PROPERTIES_DICT["D"].A)
      tglf_runpars['RLNS_2'] = float(tglf_inputs.RLNS_2[i])
      tglf_runpars['RLTS_2'] = float(tglf_inputs.RLTS_2[i])
      tglf_runpars['TAUS_2'] = float(tglf_inputs.TAUS_2[i])
      tglf_runpars['AS_2'] = float(tglf_inputs.AS_2[i])
      tglf_runpars['ZS_3'] = float(zi1[i])
      tglf_runpars['MASS_3'] = float(ai1[i] / constants.ION_PROPERTIES_DICT["D"].A)
      tglf_runpars['RLNS_3'] = float(tglf_inputs.RLNS_3[i])
      tglf_runpars['RLTS_3'] = float(tglf_inputs.RLTS_3[i])
      tglf_runpars['TAUS_3'] = float(tglf_inputs.TAUS_3[i])
      tglf_runpars['AS_3'] = float(tglf_inputs.AS_3[i])
      tglf_plan.append({
          'inputs': copy.deepcopy(tglf_runpars),
          'location': f'tglf_run_{i:04d}',
      })
  # pylint: enable=invalid-name

  return tglf_plan


def _run_tglf_single(run, queue):
  command = [
      run['execution_path'],
      '-n',
      str(2),
      '-e',
      str(run['location'])
  ]
  process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      cwd=run['run_path'],
  )

  if run['verbose']:
    # Get output and error messages
    stdout, stderr = process.communicate()

    # Print the output
    ostr = stdout.decode()

    # Print any error messages
    if stderr:
      ostr += stderr.decode()

    queue.put(ostr)


# pylint: disable=invalid-name
class TGLFTransportModelConfig(pydantic_model_base.TransportBase):
  """Model for the TGLF transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'tglf'.
    n_processes: Set number of cores used QuaLiKiz calculations.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """

  model_name: Annotated[Literal['tglf'], torax_pydantic.JAX_STATIC] = (
      'tglf'
  )
  n_processes: pydantic.PositiveInt = 8
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  def build_transport_model(self) -> TGLFTransportModel:
    return TGLFTransportModel()

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        n_processes=self.n_processes,
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        **base_kwargs,
    )
