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

"""A wrapper around QuaLiKiz.

The wrapper calls QuaLiKiz itself. Must be run with
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

import chex
import numpy as np
from qualikiz_tools.qualikiz_io import inputfiles as qualikiz_inputtools
from qualikiz_tools.qualikiz_io import qualikizrun as qualikiz_runtools
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.transport_model import qualikiz_utils
from torax.transport_model import quasilinear_utils
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # QuaLiKiz model configuration
  # No collision correction needed for QuaLiKiz
  coll_mult: float = 1.0
  # effective D / effective V approach for particle transport
  DVeff: bool = False
  # minimum |R/Lne| below which effective V is used instead of effective D
  An_min: float = 0.05
  # ensure that smag - alpha > -0.2 always, to compensate for no slab modes
  avoid_big_negative_s: bool = True
  # reduce magnetic shear by 0.5*alpha
  smag_alpha_correction: bool = False
  # if q < 1, modify input q and smag as if q~1 as if there are sawteeth
  q_sawtooth_proxy: bool = True
  # set frequency of full QuaLiKiz contour solutions
  maxruns: int = 2
  # set number of cores used QuaLiKiz calculations
  numprocs: int = 8

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(qualikiz_utils.QualikizDynamicRuntimeParams):
  maxruns: int
  numprocs: int


_DEFAULT_QLKRUN_NAME_PREFIX = 'torax_qualikiz_runs'
_DEFAULT_QLK_EXEC_PATH = '~/qualikiz/QuaLiKiz'
_QLK_EXEC_PATH = os.environ.get(
    'TORAX_QLK_EXEC_PATH', _DEFAULT_QLK_EXEC_PATH
)


class QualikizTransportModel(transport_model.TransportModel):
  """Calculates turbulent transport coefficients with QuaLiKiz."""

  def __init__(
      self,
      runtime_params: RuntimeParams | None = None,
  ):
    self._runtime_params = runtime_params or RuntimeParams()
    self._qlkrun_parentdir = tempfile.TemporaryDirectory()
    self._qlkrun_name = (
        _DEFAULT_QLKRUN_NAME_PREFIX
        + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    self._runpath = os.path.join(self._qlkrun_parentdir.name, self._qlkrun_name)
    self._frozen = True

  @property
  def runtime_params(self) -> RuntimeParams:
    return self._runtime_params

  @runtime_params.setter
  def runtime_params(self, runtime_params: RuntimeParams) -> None:
    self._runtime_params = runtime_params

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
          'JAX Compilation is not supported with QuaLiKiz.'
      )
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )
    transport = dynamic_runtime_params_slice.transport

    qualikiz_inputs = qualikiz_utils.prepare_qualikiz_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        nref=dynamic_runtime_params_slice.numerics.nref,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    # Generate nested ordered dict that will correspond to the input
    # QuaLiKiz json file
    qualikiz_plan = _extract_qualikiz_plan(
        qualikiz_inputs=qualikiz_inputs,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    self._run_qualikiz(
        qualikiz_plan, dynamic_runtime_params_slice.transport.numprocs
    )
    core_transport = self._extract_run_data(
        qualikiz_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )

    return core_transport

  def _run_qualikiz(
      self,
      qualikiz_plan: qualikiz_inputtools.QuaLiKizPlan,
      numprocs: int,
      verbose: bool = True,
  ) -> None:
    """Runs QuaLiKiz using command line tools. Loose coupling with TORAX."""

    run = qualikiz_runtools.QuaLiKizRun(
        parent_dir=self._qlkrun_parentdir.name,
        binaryrelpath=_QLK_EXEC_PATH,
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
        str(numprocs),
        _QLK_EXEC_PATH,
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
      qualikiz_inputs: qualikiz_utils.QualikizInputs,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Extracts QuaLiKiz run data from runpath."""

    # Extract QuaLiKiz outputs
    qi = np.loadtxt(self._runpath + '/output/efi_GB.dat')[:, 0]
    qe = np.loadtxt(self._runpath + '/output/efe_GB.dat')
    pfe = np.loadtxt(self._runpath + '/output/pfe_GB.dat')

    return quasilinear_utils.make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        quasilinear_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


def _extract_qualikiz_plan(
    qualikiz_inputs: qualikiz_utils.QualikizInputs,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
):
  """Converts TORAX parameters to QuaLiKiz input JSON.

  Args:
      qualikiz_inputs: Precomputed physics data.
      dynamic_runtime_params_slice: Runtime params at time t.
      geo: TORAX geometry object.
      core_profiles: TORAX CoreProfiles object, containing time-evolvable
        quantities like q

  Returns:
      A qualikiz_tools.qualikiz_io.inputfiles.QuaLiKizPlan
  """
  # Generate QuaLiKizXPoint, changing object defaults

  # Needed to avoid pytype complaints
  assert isinstance(
      dynamic_runtime_params_slice.transport, DynamicRuntimeParams
  )
  transport: DynamicRuntimeParams = dynamic_runtime_params_slice.transport

  # numerical parameters
  meta = qualikiz_inputtools.QuaLiKizXpoint.Meta(
      maxpts=5e6,
      numsols=2,
      separateflux=True,
      phys_meth=1,
      rhomin=0.0,
      rhomax=0.98,
      maxruns=transport.maxruns,
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
      Ro=np.array(geo.Rmaj),
      Rmin=np.array(geo.Rmin),
      Bo=np.array(geo.B0),
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
  Zi0 = core_profiles.Zi
  Zi1 = core_profiles.Zimp

  # Calculate main ion dilution
  ni0 = core_profiles.ni.face_value() / core_profiles.ne.face_value()

  ion0 = qualikiz_inputtools.Ion(
      T=8,  # will be scan variable
      n=1,  # will be scan variable
      At=0,  # will be scan variable
      An=0,  # will be scan variable
      type=1,
      anis=1,
      danisdr=0,
      A=np.array(dynamic_runtime_params_slice.plasma_composition.Ai),
      Z=np.array(Zi0),
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
      A=np.array(Zi1 * 2),
      Z=np.array(Zi1),
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
      'Te': np.array(core_profiles.temp_el.face_value()),
      'ne': (
          np.array(
              core_profiles.ne.face_value()
              * dynamic_runtime_params_slice.numerics.nref
          )
          / 1e19
      ),
      'Ate': np.array(qualikiz_inputs.Ate),
      'Ane': np.array(qualikiz_inputs.Ane),
      'Ti0': np.array(core_profiles.temp_ion.face_value()),
      'ni0': np.array(ni0),
      'Ati0': np.array(qualikiz_inputs.Ati),
      'Ani0': np.array(qualikiz_inputs.Ani0),
      'Ti1': np.array(core_profiles.temp_ion.face_value()),
      'ni1': np.array(ni1),
      'Ati1': np.array(qualikiz_inputs.Ati),
      'Ani1': np.array(qualikiz_inputs.Ani1),
  }
  # pylint: enable=invalid-name

  qualikiz_plan = qualikiz_inputtools.QuaLiKizPlan(
      scan_dict=scan_dict, scan_type='parallel', xpoint_base=xpoint_base
  )

  return qualikiz_plan


def _default_qualikiz_builder() -> QualikizTransportModel:
  return QualikizTransportModel()


@dataclasses.dataclass(kw_only=True)
class QualikizTransportModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class QualikizTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )
  model_path: str | None = None

  _builder: Callable[
      [],
      QualikizTransportModel,
  ] = _default_qualikiz_builder

  def __call__(
      self,
  ) -> QualikizTransportModel:
    return self._builder()
