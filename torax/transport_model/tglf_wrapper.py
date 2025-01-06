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

"""A wrapper around tglf.

The wrapper calls tglf itself. Must be run with
TORAX_COMPILATION_ENABLED=False. Used for generating ground truth for QLKNN11D
evaluation. Kept as an internal model.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import os
import subprocess
import tempfile
from functools import partial
from typing import Dict, List, Union
from dataclasses import fields
from multiprocessing import Pool
import pandas as pd

import chex
import numpy as np
from quasilinear_utils import  QuasilinearTransportModel
#tglf_tools.tglf_io import inputfiles as tglf_inputtools
#from tglf_tools.tglf_io import tglfrun as tglf_runtools
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.transport_model import tglf_based_transport_model
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model
from torax.transport import tglf_tools



# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """
  numprocs: int = 2
  NBASIS_MAX: int = 4
  NBASIS_MIN: int = 2
  USE_TRANSPORT_MODEL: bool = True
  NS: int = 2 
  NXGRID: int = 16
  GEOMETRY_FLAG: int = 1
  USE_BPER: bool = True
  USE_BPAR: bool = True
  KYGRID_MODEL: int = 4
  SAT_RULE: int = 1
  USE_MHD_RULE: bool = False
  ALPHA_ZF: int = -1
  FILTER: float = 2.0

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(tglf_based_transport_model.DynamicRuntimeParams):
  numprocs: int
  NBASIS_MAX: int
  NBASIS_MIN: int
  USE_TRANSPORT_MODEL: bool
  NS: int  
  NXGRID: int
  GEOMETRY_FLAG: int
  USE_BPER: bool
  USE_BPAR: bool
  KYGRID_MODEL: int
  SAT_RULE: int
  USE_MHD_RULE: bool
  ALPHA_ZF: int
  FILTER: float

class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))

_DEFAULT_tglfrun_NAME_PREFIX = 'torax_tglf_runs'
_DEFAULT_TGLF_EXEC_PATH = '~/tglf/tglf'
_TGLF_EXEC_PATH = os.environ.get(
    'TORAX_TGLF_EXEC_PATH', _DEFAULT_TGLF_EXEC_PATH
)

class TGLFTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculates turbulent transport coefficients with tglf."""

  def __init__(
      self,
      runtime_params: RuntimeParams | None = None,
  ):
    self._runtime_params = runtime_params or RuntimeParams()
    self._tglfrun_parentdir = tempfile.TemporaryDirectory()
    self._tglfrun_name = (
        _DEFAULT_tglfrun_NAME_PREFIX
        + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    self._runpath = os.path.join(self._tglfrun_parentdir.name, self._tglfrun_name)
    self._frozen = True

  @property
  def runtime_params(self) -> RuntimeParams:
    return self._runtime_params

  @runtime_params.setter
  def runtime_params(self, runtime_params: RuntimeParams) -> None:
    self._runtime_params = runtime_params

  def _get_one_simulation_rundir(self, n: float):
    return os.path.join(self._runpath,f'sim_{n}')
  
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice, 
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Calculates several transport coefficients simultaneously.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.

    Returns:
      coeffs: transport coefficients

    Raises:
      EnvironmentError: if TORAX_COMPILATION_ENABLED is set to True.
    """

    if jax_utils.env_bool('TORAX_COMPILATION_ENABLED', True):
      raise EnvironmentError(
          'TORAX_COMPILATION_ENABLED environment variable is set to True.'
          'JAX Compilation is not supported with tglf.'
      )
    
    # TODO
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )
    transport = dynamic_runtime_params_slice.transport

    tglf_inputs = self._prepare_tglf_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        geo=geo,
        core_profiles=core_profiles,
    )
    # Generate list of dictionaries that will correspond to input.tglf
    tglf_plan = _extract_tglf_plan(
        tglf_inputs=tglf_inputs, 
        dynamic_runtime_params_slice=dynamic_runtime_params_slice
    )
    self._run_tglf(
        tglf_plan=tglf_plan, 
        numprocs=dynamic_runtime_params_slice.transport.numprocs
    )
    core_transport = self._extract_run_data( 
        tglf_inputs=tglf_inputs,
        transport=transport, 
        geo=geo, 
        core_profiles=core_profiles,
    )

    return core_transport

  def _run_tglf(
      self,
      tglf_plan: List[Dict[str,Union[int,float,bool]]],
      numprocs: int,
      verbose: bool = True,
  ) -> None:
    """Runs tglf using command line tools. Loose coupling with TORAX."""

    # Prepare parent run directory
    if not os.path.exists(self._runpath):
      os.makedirs(self._runpath)

    num_simulations = len(tglf_plan)
    for n in num_simulations:
      # Prepare local simulation directory
      this_rundir = self._get_one_simulation_rundir(n)
      if not os.path.exists(this_rundir):
        os.makedirs(this_rundir)      
      # Dump input file
      with open(os.path.join(this_rundir,'input.tglf'), 'w') as f:
        for key,value in tglf_plan[n].items():
          f.write(f'{key}={value}')
      # run TGLF
      command = [
        'tglf',
        '-n',
        str(numprocs),
        '-e',
        '.'
      ]      
      process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,        
        cwd=this_rundir
      )
      if verbose:
        # Get output and error messages
        stdout, stderr = process.communicate()

      # Print the output
      print(stdout.decode())

      # Print any error messages
      if stderr:
        print(stderr.decode())

  def _extract_run_data(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Extracts tglf run data from runpath."""

    qi = []
    qe = []
    pfe = []
    for run_dir in  list(filter(os.path.isdir, os.listdir(self._runpath))):
      df = pd.read_fwf(os.path.join(run_dir, 'out.tglf.run'), skiprows=5, index_col=0)
      pfe.append(df.loc['elec','Gam/Gam_GB'])
      qe.append(df.loc['elec','Q/Q_GB'])
      qi.append(df.loc['ion1','Q/Q_GB'])
    return self._make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        quasilinear_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


def _extract_tglf_plan(
    tglf_inputs: tglf_based_transport_model.TGLFInputs,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
  )->List[Dict[str,Union[int,float,bool]]]:
  """Converts TORAX parameters to tglf input file input.tglf. Currently only supports electron and main ion.

  Args:
      tglf_inputs: Precomputed physics data.
      dynamic_runtime_params_slice: Runtime params at time t.
      geo: TORAX geometry object.
      core_profiles: TORAX CoreProfiles object, containing time-evolvable
        quantities like q

  Returns:
      A list containing dictionaries of input configs for TGLF.
  """

  def _get_q_prime_loc(params):
    return params['SHAT']*(params['Q_LOC']/params['RMIN_LOC'])**2 # --- q_prime_loc = (q / r) ** 2  * shat  = q / r * dq/dr

  def _get_p_prime_loc(params):
      # The below is a manipulation of https://gafusion.github.io/doc/tglf/tglf_list.html#p-prime-loc
      # --- p_prime_loc = -q/r * betae/(8*pi) * \sum_k [n_k/n_e * T_k/T_e * (a/Ln_k + a/LT_k)]
      first_term = params['Q_LOC']/params['RMIN_LOC']
      second_term = params['BETAE']/(8*np.pi)
      Sum_term = 0
      for k in [1,2]: #Electrons and main ion only
          tmp = params[f'AS_{k}']*params[f'TAUS_{k}']*(params[f'RLNS_{k}']+params[f'RLTS_{k}'])
          Sum_term += tmp
      return -first_term*second_term*Sum_term

  def add_missing_params(
      params: Dict[str,float], 
      transport: DynamicRuntimeParams)->Dict[str,float]:
    """Utility to create TGLF input file
    
    Args:
      physical_params: Physical parameters
      transport: Runtime params at time t 

    Returns:
      TGLF inputs inclusive of numerics and other parameters not included in the

    """
    numerics_params = {
      'NBASIS_MAX': transport.NBASIS_MAX,
      'NBASIS_MIN': transport.NBASIS_MIN,
      'USE_TRANSPORT_MODEL': transport.USE_TRANSPORT_MODEL,
      'NS': transport.NS,
      'NXGRID': transport.NXGRID,
      'GEOMETRY_FLAG': transport.GEOMETRY_FLAG,
      'USE_BPER': transport.USE_BPER,
      'USE_BPAR': transport.USE_BPAR,
      'KYGRID_MODEL': transport.KYGRID_MODEL,
      'SAT_RULE': transport.SAT_RULE,
      'USE_MHD_RULE': transport.USE_MHD_RULE,
      'ALPHA_ZF': transport.ALPHA_ZF,
      'FILTER': transport.FILTER
    }
    params['P_PRIME_LOC'] = _get_p_prime_loc(params)
    params['Q_PRIME_LOC'] = _get_q_prime_loc(params)
    params.update(numerics_params)
    return params
    
  assert isinstance(
      dynamic_runtime_params_slice.transport, DynamicRuntimeParams
  )
  transport: DynamicRuntimeParams = dynamic_runtime_params_slice.transport
  prepare_input_dict = partial(add_missing_params, transport=transport)
  zipped_arrays = zip(
    np.array(tglf_inputs.Rmin),
    np.array(tglf_inputs.dRmaj),
    np.array(tglf_inputs.q),
    np.array(tglf_inputs.Ate),
    np.array(tglf_inputs.Ati),
    np.array(tglf_inputs.Ane),
    np.array(tglf_inputs.Ti_over_Te),
    np.array(tglf_inputs.nu_ee),
    np.array(tglf_inputs.kappa),
    np.array(tglf_inputs.kappa_shear),
    np.array(tglf_inputs.delta),
    np.array(tglf_inputs.delta_shear),
    np.array(tglf_inputs.beta_e),
  )
  tglf_plan = [
    prepare_input_dict(
      {
      'RMIN_LOC': rmin, 
      'DRMAJDX_LOC': dR, 
      'Q_LOC': q, 
      'RMAJ_LOC': R, 
      'RLTS_1': ate,  
      'RLTS_2': ati,
      'RLNS_1': ane,
      'RLNS_2': ane, # quasineutrality
      'TAUS_1': 1,
      'TAUS_2': tie, 
      'XNUE': nu, 
      'KAPPA_LOC': k, 
      'S_KAPPA_LOC': sk, 
      'DELTA_LOC': d, 
      'S_DELTA_LOC': sd, 
      'BETAE': b, 
      'ZEFF': z,
      'AS_1': 1,
      'AS_2': 1
     }
    )
    for rmin, dR, q, R, ate, ati, ane, tie, nu, k, sk, d, sd, b, z in zipped_arrays
  ]
  return tglf_plan


def _default_tglf_builder() -> TGLFTransportModel:
  return TGLFTransportModel()


@dataclasses.dataclass(kw_only=True)
class tglfTransportModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class tglfTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )
  model_path: str | None = None

  _builder: Callable[
      [],
      TGLFTransportModel,
  ] = _default_tglf_builder

  def __call__(
      self,
  ) -> TGLFTransportModel:
    return self._builder()
