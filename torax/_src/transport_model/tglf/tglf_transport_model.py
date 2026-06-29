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
from typing import Annotated, Any, Literal, Mapping, Sequence, TypeAlias

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
from torax._src.transport_model.tglf import defaults as tglf_defaults
from torax._src.transport_model.tglf import tglf2py

from absl import app

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
  tglf_exec_path: str = dataclasses.field(
      default='~/tglf', metadata={'static': True}
  )
  output_directory: str = dataclasses.field(
      default='/tmp/torax_tglf_runs', metadata={'static': True}
  )


def _run_single_tglf(
    i: int, run_inputs: dict[str, Any], base_defaults: dict[str, Any]
) -> tuple[int, float, float, float]:
  """Runs a single tglf2py evaluation in a worker process.

  This function must be at the top level of the module to be pickleable, which
  is required for passing to the executor.

  Args:
    i: The index of the run.
    run_inputs: The inputs to the run.
    base_defaults: The base defaults for the run.

  Returns:
    A tuple of the index, electron heat flux, ion heat flux, and total ion
    particle flux.
  """
  combined_inputs = dict(base_defaults)
  combined_inputs.update(run_inputs)

  py_inputs = {}
  for k, v in combined_inputs.items():
    if k.endswith('_7') or k.startswith('SHAPE_'):
      continue
    if isinstance(v, str):
      v_low = v.lower()
      if v_low in [
          '.true.',
          '.false.',
          'true',
          'false',
          't',
          'f',
          'yes',
          'no',
      ]:
        v = v_low in ['.true.', 'true', '1', 't', 'yes', 'y']
      else:
        try:
          v = int(v)
        except ValueError:
          try:
            v = float(v)
          except ValueError:
            pass
    py_inputs[k] = v
  pe, _, qe, qi = tglf2py.run_tglf(**py_inputs)
  return i, float(pe), float(qe), float(np.sum(qi))


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class TGLFTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
  """Calculates turbulent transport coefficients with tglf2py in-memory."""

  # Hash by id as should be unique per instance
  executor: futures.Executor = dataclasses.field(metadata={'hash_by_id': True})

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
      tglf_plan = self._extract_tglf_plan(
          tglf_inputs=tglf_inputs,
          transport=transport_runtime_params,
          geo=geo,
          core_profiles=core_profiles,
      )

      base_defaults = dict(tglf_defaults.TGLF_DEFAULTS)

      qe_py = np.zeros((len(tglf_plan),))
      qi_py = np.zeros((len(tglf_plan),))
      ge_py = np.zeros((len(tglf_plan),))

      future_list = [
          self.executor.submit(
              _run_single_tglf, i, run['inputs'], base_defaults
          )
          for i, run in enumerate(tglf_plan)
      ]
      for future in futures.as_completed(future_list):
        i, pe_val, qe_val, qi_val = future.result()
        ge_py[i] = pe_val
        qe_py[i] = qe_val
        qi_py[i] = qi_val

      core_transport = self._make_core_transport(
          electron_heat_flux_GB=qe_py,
          ion_heat_flux_GB=qi_py,
          electron_particle_flux_GB=ge_py,
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

    core_transport = jax.pure_callback(
        callback,
        result_shape_dtypes,
        tglf_inputs,
        transport_runtime_params,
        geo,
        core_profiles,
    )

    return core_transport

  def _extract_tglf_plan(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> Sequence[Mapping[str, Any]]:
    """Converts TORAX parameters to TGLF input dictionary."""
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
    tglf_input_template = dict(transport.tglf_settings)

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

    for deprecated_param in ['tglf_exec_path', 'output_directory', 'verbose']:
      if deprecated_param in data:
        logging.warning(
            "Config option '%s' is deprecated and has no effect.",
            deprecated_param,
        )

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
        executor=futures.ProcessPoolExecutor(
            max_workers=self.n_processes,
            mp_context=g3_multiprocessing.get_context(
                g3_multiprocessing.ABSL_SPAWN
            ),
        )
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        n_processes=self.n_processes,
        n_cores_per_process=self.n_cores_per_process,
        verbose=self.verbose,
        tglf_exec_path=self.tglf_exec_path,
        output_directory=self.output_directory,
        use_rotation=self.use_rotation,
        rotation_multiplier=self.rotation_multiplier,
        DV_effective=self.DV_effective,
        collisionality_multiplier=self.collisionality_multiplier,
        An_min=self.An_min,
        tglf_settings=self.tglf_settings,
        **base_kwargs,
    )
