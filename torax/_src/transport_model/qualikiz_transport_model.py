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

"""A transport model that calls QuaLiKiz.

Used for generating ground truth for surrogate model evaluations.
"""

import dataclasses
import datetime
import os
import subprocess
import tempfile
from typing import Annotated, Literal
import uuid

import chex
import jax
from jax import numpy as jnp
import numpy as np
import pydantic
from qualikiz_tools.qualikiz_io import inputfiles as qualikiz_inputtools
from qualikiz_tools.qualikiz_io import qualikizrun as qualikiz_runtools
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(qualikiz_based_transport_model.RuntimeParams):
  n_max_runs: int
  n_processes: int


_DEFAULT_QLKRUN_NAME_PREFIX = 'torax_qualikiz_runs'


def _get_qlk_exec_path() -> str:
  default_qlk_exec_path = '~/qualikiz/QuaLiKiz'
  return os.environ.get('TORAX_QLK_EXEC_PATH', default_qlk_exec_path)


class QualikizTransportModel(
    qualikiz_based_transport_model.QualikizBasedTransportModel
):
  """Calculates turbulent transport coefficients with QuaLiKiz."""

  def __init__(self):
    self._qlkrun_parentdir = tempfile.TemporaryDirectory()
    # Include UUID to prevent collisions when multiple simulations start
    # simultaneously (e.g., in SLURM distributed systems)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:8]
    # Check for SLURM job ID if available (common in distributed computing)
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
      unique_suffix = f'job_{slurm_job_id}_uuid_{short_uuid}'
    else:
      unique_suffix = f'uuid_{short_uuid}'
    self._qlkrun_name = (
        _DEFAULT_QLKRUN_NAME_PREFIX + '_' + timestamp + '_' + unique_suffix
    )
    self._runpath = os.path.join(self._qlkrun_parentdir.name, self._qlkrun_name)

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

    qualikiz_inputs = self._prepare_qualikiz_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    def callback(
        qualikiz_inputs,
        transport_runtime_params,
        geo,
        core_profiles,
        two_point_mask,
    ):
      # Qualikiz expects numpy arrays, but the callback passes jax.Array.
      (
          qualikiz_inputs,
          transport_runtime_params,
          geo,
          core_profiles,
          two_point_mask,
      ) = jax.tree.map(
          np.asarray,
          (
              qualikiz_inputs,
              transport_runtime_params,
              geo,
              core_profiles,
              two_point_mask,
          ),
      )
      # Generate nested ordered dict that will correspond to the input
      # QuaLiKiz json file
      qualikiz_plan = _extract_qualikiz_plan(
          qualikiz_inputs=qualikiz_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )
      self._run_qualikiz(qualikiz_plan, transport_runtime_params.n_processes)
      core_transport = self._extract_run_data(
          qualikiz_inputs=qualikiz_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          two_point_mask=two_point_mask,
      )
      return core_transport

    result_shape_dtypes = transport_coeffs.TransportCoeffs.zeros(geo)
    # Even though qualikiz has side-effects (writing and reading from disk) we
    # still use a pure_callback here as:
    # 1. Nothing outside of this method depends on the side-effect.
    # 2. We don't mind if results are cached or recomputed.
    # 3. DCE will not happen here as we make use of the `core_transport` result.
    # This is based on the current implementation of pure_callback and JAX
    # may change the implementation making this not appropriate down the line.
    core_transport = jax.pure_callback(
        callback,
        result_shape_dtypes,
        qualikiz_inputs,
        transport_runtime_params,
        geo,
        core_profiles,
        two_point_mask,
    )

    return core_transport, component_state

  def _run_qualikiz(
      self,
      qualikiz_plan: qualikiz_inputtools.QuaLiKizPlan,
      n_processes: int,
      verbose: bool = True,
  ) -> None:
    """Runs QuaLiKiz using command line tools. Loose coupling with TORAX."""
    execution_path = _get_qlk_exec_path()
    run = qualikiz_runtools.QuaLiKizRun(
        parent_dir=self._qlkrun_parentdir.name,
        binaryrelpath=execution_path,
        name=self._qlkrun_name,
        qualikiz_plan=qualikiz_plan,
        verbose=verbose,
    )

    # Prepare run directory
    if not os.path.exists(self._runpath):
      run.prepare()
    else:
      qualikiz_plan.to_json(
          os.path.join(
              self._qlkrun_parentdir.name, self._qlkrun_name, 'parameters.json'
          )
      )

    # Generate QuaLiKiz input binaries
    run.generate_input()

    # Run QuaLiKiz
    command = [
        'mpirun',
        '-np',
        str(n_processes),
        execution_path,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=f'{self._qlkrun_parentdir.name}/{self._qlkrun_name}',
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
      qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    """Extracts QuaLiKiz run data from runpath."""

    # Extract QuaLiKiz outputs
    qi = jnp.asarray(np.loadtxt(self._runpath + '/output/efi_GB.dat')[:, 0])
    qe = jnp.asarray(np.loadtxt(self._runpath + '/output/efe_GB.dat'))
    pfe = jnp.asarray(np.loadtxt(self._runpath + '/output/pfe_GB.dat'))

    return self._make_transport_coeffs(
        qi=qi,
        qe=qe,
        pfe=pfe,
        qualikiz_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

  def __hash__(self) -> int:
    return hash(('QualikizTransportModel' + self._runpath))

  def __eq__(self, other) -> bool:
    return (
        isinstance(other, QualikizTransportModel)
        and self._runpath == other._runpath
    )


def _extract_qualikiz_plan(
    qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs,
    transport: RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> qualikiz_inputtools.QuaLiKizPlan:
  """Converts TORAX parameters to QuaLiKiz input JSON.

  Args:
      qualikiz_inputs: Precomputed physics data.
      transport: Runtime parameters for the qualikiz transport model.
      geo: TORAX geometry object.
      core_profiles: TORAX CoreProfiles object, containing time-evolvable
        quantities like q

  Returns:
      A qualikiz_tools.qualikiz_io.inputfiles.QuaLiKizPlan
  """
  # Generate QuaLiKizXPoint, changing object defaults

  if transport.rotation_mode == qualikiz_based_transport_model.RotationMode.OFF:
    rot_flag = 0
  elif (
      transport.rotation_mode
      == qualikiz_based_transport_model.RotationMode.FULL_RADIUS
  ):
    rot_flag = 1
  elif (
      transport.rotation_mode
      == qualikiz_based_transport_model.RotationMode.HALF_RADIUS
  ):
    rot_flag = 2
  else:
    raise ValueError(f'Unsupported rotation mode: {transport.rotation_mode}')

  # numerical parameters
  meta = qualikiz_inputtools.QuaLiKizXpoint.Meta(
      maxpts=5e6,
      numsols=2,
      separateflux=True,
      phys_meth=1,
      rhomin=0.0,
      rhomax=0.98,
      maxruns=transport.n_max_runs,
      rot_flag=rot_flag,
      timeout=30,
  )

  options = qualikiz_inputtools.QuaLiKizXpoint.Options(
      recalc_Nustar=False,
  )

  # wavenumber grid
  kthetarhos = [
      0.1,
      0.175,
      0.25,
      0.325,
      0.4,
      0.5,
      0.7,
      1.0,
      1.8,
      3.0,
      9.0,
      15.0,
      21.0,
      27.0,
      36.0,
      45.0,
  ]

  # magnetic geometry and rotation
  qlk_geometry = qualikiz_inputtools.QuaLiKizXpoint.Geometry(
      x=0.5,  # will be scan variable
      rho=0.5,  # will be scan variable
      Ro=np.array(geo.R_major),
      Rmin=np.array(geo.a_minor),
      Bo=np.array(geo.B_0),
      q=2,  # will be scan variable
      smag=1,  # will be scan variable
      alpha=0,  # will be scan variable
      Machtor=0,  # will be scan variable
      Autor=0,  # will be scan variable (computed from Machtor and GammaE)
      Machpar=0,  # will be scan variable (computed from Machtor and GammaE)
      Aupar=0,  # will be scan variable (computed from Machtor and GammaE)
      gammaE=0,  # will be scan variable
  )

  elec = qualikiz_inputtools.Electron(
      T=8,  # will be scan variable
      n=1,  # will be scan variable
      At=0,  # will be scan variable
      An=0,  # will be scan variable
      type=1,
      anis=1,
      danisdr=0,
  )

  Zi0 = core_profiles.ions.Z_i_face
  Zi1 = core_profiles.ions.Z_impurity_face

  # Calculate main ion dilution
  ni0 = (
      core_profiles.kinetic.n_i.face_value()
      / core_profiles.kinetic.n_e.face_value()
  )

  ion0 = qualikiz_inputtools.Ion(
      T=8,  # will be scan variable
      n=1,  # will be scan variable
      At=0,  # will be scan variable
      An=0,  # will be scan variable
      type=1,
      anis=1,
      danisdr=0,
      A=np.array(core_profiles.ions.A_i),
      Z=1,  # will be a scan variable
  )

  ni1 = (1 - ni0 * Zi0) / Zi1  # quasineutrality

  ion1 = qualikiz_inputtools.Ion(
      T=8,  # will be scan variable
      n=0,  # will be scan variable
      At=0,  # will be scan variable
      An=0,  # will be scan variable
      type=1,
      anis=1,
      danisdr=0,
      A=20,  # will be a scan variable
      Z=10,  # will be a scan variable
  )

  ions = qualikiz_inputtools.IonList(ion0, ion1)

  # build QuaLiKizXpoint
  xpoint_base = qualikiz_inputtools.QuaLiKizXpoint(
      kthetarhos=kthetarhos,
      electrons=elec,
      ions=ions,
      **qlk_geometry,
      **meta,
      **options,
  )

  scan_dict: dict[str, array_typing.Array] = {
      'x': np.array(qualikiz_inputs.x),
      'rho': np.array(geo.rho_face_norm),
      'q': np.array(qualikiz_inputs.q),
      'smag': np.array(qualikiz_inputs.smag),
      'alpha': np.array(qualikiz_inputs.alpha),
      'Te': np.array(core_profiles.kinetic.T_e.face_value()),
      'ne': np.array(core_profiles.kinetic.n_e.face_value()) / 1e19,
      'Ate': np.array(qualikiz_inputs.Ate),
      'Ane': np.array(qualikiz_inputs.Ane),
      'Ti0': np.array(core_profiles.kinetic.T_i.face_value()),
      'ni0': np.array(ni0),
      'Ati0': np.array(qualikiz_inputs.Ati),
      'Ani0': np.array(qualikiz_inputs.Ani0),
      'Zi0': np.array(Zi0),
      'Ti1': np.array(core_profiles.kinetic.T_i.face_value()),
      'ni1': np.array(ni1),
      'Ati1': np.array(qualikiz_inputs.Ati),
      'Ani1': np.array(qualikiz_inputs.Ani1),
      'Zi1': np.array(Zi1),
      'Ai1': np.array(core_profiles.ions.A_impurity_face),
  }
  if transport.rotation_mode != qualikiz_based_transport_model.RotationMode.OFF:
    scan_dict['gammaE'] = qualikiz_inputs.gamma_E_QLK
    scan_dict['Machtor'] = qualikiz_inputs.mach_toroidal

  qualikiz_plan = qualikiz_inputtools.QuaLiKizPlan(
      scan_dict=scan_dict, scan_type='parallel', xpoint_base=xpoint_base
  )

  return qualikiz_plan


class QualikizTransportModelConfig(
    qualikiz_based_transport_model.QualikizBasedTransportModelConfig
):
  """Model for the Qualikiz transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'qualikiz'.
    n_max_runs: Set frequency of full QuaLiKiz contour solutions.
    n_processes: Set number of cores used QuaLiKiz calculations.
    collisionality_multiplier: Collisionality multiplier.
  """

  model_name: Annotated[Literal['qualikiz'], torax_pydantic.JAX_STATIC] = (
      'qualikiz'
  )
  n_max_runs: pydantic.PositiveInt = 2
  n_processes: pydantic.PositiveInt = 8
  collisionality_multiplier: pydantic.PositiveFloat = 1.0

  def build_transport_model(self) -> QualikizTransportModel:
    return QualikizTransportModel()

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        n_max_runs=self.n_max_runs,
        n_processes=self.n_processes,
        **vars(base_params),
    )
