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
from typing import Annotated
from typing import Literal

import chex
import jax
import numpy as np
import pydantic
from qualikiz_tools.qualikiz_io import inputfiles as qualikiz_inputtools
from qualikiz_tools.qualikiz_io import qualikizrun as qualikiz_runtools
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model


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
    self._qlkrun_name = (
        _DEFAULT_QLKRUN_NAME_PREFIX
        + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    self._runpath = os.path.join(self._qlkrun_parentdir.name, self._qlkrun_name)

  def _call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
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

    qualikiz_inputs = self._prepare_qualikiz_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )

    def callback(qualikiz_inputs, transport_runtime_params, geo, core_profiles):
      # Qualikiz expects numpy arrays, but the callback passes jax.Array.
      (qualikiz_inputs, transport_runtime_params, geo, core_profiles) = (
          jax.tree.map(
              np.asarray,
              (qualikiz_inputs, transport_runtime_params, geo, core_profiles),
          )
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
    )

    return core_transport

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
  ) -> transport_model.TurbulentTransport:
    """Extracts QuaLiKiz run data from runpath."""

    # Extract QuaLiKiz outputs
    qi = np.loadtxt(self._runpath + '/output/efi_GB.dat')[:, 0]
    qe = np.loadtxt(self._runpath + '/output/efe_GB.dat')
    pfe = np.loadtxt(self._runpath + '/output/pfe_GB.dat')

    return self._make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        quasilinear_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.R_major,
        gyrobohm_flux_reference_length=geo.a_minor,
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

  # numerical parameters
  meta = qualikiz_inputtools.QuaLiKizXpoint.Meta(
      maxpts=5e6,
      numsols=2,
      separateflux=True,
      phys_meth=1,
      rhomin=0.0,
      rhomax=0.98,
      maxruns=transport.n_max_runs,
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
      Machtor=0,
      Autor=0,
      Machpar=0,
      Aupar=0,
      gammaE=0,
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

  # pylint: disable=invalid-name
  Zi0 = core_profiles.Z_i_face
  Zi1 = core_profiles.Z_impurity_face

  # Calculate main ion dilution
  ni0 = core_profiles.n_i.face_value() / core_profiles.n_e.face_value()

  ion0 = qualikiz_inputtools.Ion(
      T=8,  # will be scan variable
      n=1,  # will be scan variable
      At=0,  # will be scan variable
      An=0,  # will be scan variable
      type=1,
      anis=1,
      danisdr=0,
      A=np.array(core_profiles.A_i),
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

  scan_dict = {
      'x': np.array(qualikiz_inputs.x),
      'rho': np.array(geo.rho_face_norm),
      'q': np.array(qualikiz_inputs.q),
      'smag': np.array(qualikiz_inputs.smag),
      'alpha': np.array(qualikiz_inputs.alpha),
      'Te': np.array(core_profiles.T_e.face_value()),
      'ne': np.array(core_profiles.n_e.face_value()) / 1e19,
      'Ate': np.array(qualikiz_inputs.Ate),
      'Ane': np.array(qualikiz_inputs.Ane),
      'Ti0': np.array(core_profiles.T_i.face_value()),
      'ni0': np.array(ni0),
      'Ati0': np.array(qualikiz_inputs.Ati),
      'Ani0': np.array(qualikiz_inputs.Ani0),
      'Zi0': np.array(Zi0),
      'Ti1': np.array(core_profiles.T_i.face_value()),
      'ni1': np.array(ni1),
      'Ati1': np.array(qualikiz_inputs.Ati),
      'Ani1': np.array(qualikiz_inputs.Ani1),
      'Zi1': np.array(Zi1),
      'Ai1': np.array(core_profiles.A_impurity_face),
  }
  # pylint: enable=invalid-name

  qualikiz_plan = qualikiz_inputtools.QuaLiKizPlan(
      scan_dict=scan_dict, scan_type='parallel', xpoint_base=xpoint_base
  )

  return qualikiz_plan


# pylint: disable=invalid-name
class QualikizTransportModelConfig(pydantic_model_base.TransportBase):
  """Model for the Qualikiz transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'qualikiz'.
    n_max_runs: Set frequency of full QuaLiKiz contour solutions.
    n_processes: Set number of cores used QuaLiKiz calculations.
    collisionality_multiplier: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DV_effective: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """

  model_name: Annotated[Literal['qualikiz'], torax_pydantic.JAX_STATIC] = (
      'qualikiz'
  )
  n_max_runs: pydantic.PositiveInt = 2
  n_processes: pydantic.PositiveInt = 8
  collisionality_multiplier: pydantic.PositiveFloat = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  def build_transport_model(self) -> QualikizTransportModel:
    return QualikizTransportModel()

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        n_max_runs=self.n_max_runs,
        n_processes=self.n_processes,
        collisionality_multiplier=self.collisionality_multiplier,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        **base_kwargs,
    )
