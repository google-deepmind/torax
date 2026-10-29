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

"""Pydantic config for Torax."""

import copy
import logging
from typing import Any, Mapping, Self

import numpy as np
import pydantic
from torax._src import models
from torax._src import version
from torax._src.config import numerics as numerics_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import runtime_params as core_profile_runtime_params_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.edge import config as edge_config
from torax._src.edge.extended_lengyel import config as extended_lengyel_config
from torax._src.edge.extended_lengyel import extended_lengyel_enums
from torax._src.fvm import enums
from torax._src.geometry import config as geometry_config
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import config as ibc_config
from torax._src.mhd import config as mhd_config
from torax._src.neoclassical import config as neoclassical_config
from torax._src.pedestal_model import config as pedestal_config
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params
from torax._src.solver import config as solver_config
from torax._src.sources import config as sources_config
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.time_step_calculator import config as time_step_calculator_config
from torax._src.torax_pydantic import file_restart as file_restart_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import config as transport_config


class ToraxConfig(torax_pydantic.ConfigBase):
  """Base config class for Torax.

  Attributes:
    profile_conditions: Config for the profile conditions.
    numerics: Config for the numerics.
    plasma_composition: Config for the plasma composition.
    geometry: Config for the geometry.
    pedestal: Config for the pedestal model. If an empty dictionary is passed
      in, the pedestal model will be set to `no_pedestal`.
    sources: Config for the sources.
    neoclassical: Config for the neoclassical models.
    solver: Config for the solver. If an empty dictionary is passed in, the
      solver model will be set to `linear`.
    transport: Config for the transport model. If an empty dictionary is passed
      in, the transport model will be set to `constant`.
    mhd: Optional config for mhd models. If None, no MHD models are used.
    time_step_calculator: Optional config for the time step calculator. If not
      provided the default chi time step calculator is used.
    restart: Optional config for file restart. If None, no file restart is
      performed.
  """

  profile_conditions: profile_conditions_lib.ProfileConditions
  numerics: numerics_lib.NumericsConfig
  plasma_composition: plasma_composition_lib.PlasmaComposition
  geometry: geometry_config.GeometryConfig
  sources: sources_config.SourcesConfig
  neoclassical: neoclassical_config.NeoclassicalConfig = pydantic.Field()
  solver: solver_config.SolverConfig = pydantic.Field()
  transport: transport_config.TransportConfig = pydantic.Field()
  pedestal: pedestal_config.PedestalConfig = pydantic.Field()
  mhd: mhd_config.MHDConfig = mhd_config.MHDConfig()
  edge: edge_config.EdgeConfig | None = None
  time_step_calculator: time_step_calculator_config.TimeStepCalculatorConfig = (
      time_step_calculator_config.ChiTimeStepCalculatorConfig()
  )
  restart: file_restart_config.FileRestartConfig | None = pydantic.Field(
      default=None
  )

  def build_models(self):
    edge_model = self.edge.build_edge_model() if self.edge else None
    return models.Models(
        pedestal_model=self.pedestal.build_pedestal_model(),
        source_models=self.sources.build_models(),
        transport_model=self.transport.build_transport_model(),
        neoclassical_model=self.neoclassical.build_model(),
        mhd_models=self.mhd.build_mhd_models(),
        edge_model=edge_model,
        time_step_calculator=self.time_step_calculator.build_time_step_calculator(),
        internal_boundary_condition_model=(
            self.profile_conditions.internal_boundary_conditions.build_model()
        ),
    )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if 'neoclassical' not in configurable_data:
      configurable_data['neoclassical'] = {'model_name': 'analytical'}
    elif (
        isinstance(configurable_data['neoclassical'], dict)
        and 'model_name' not in configurable_data['neoclassical']
    ):
      configurable_data['neoclassical']['model_name'] = 'analytical'
    if (
        isinstance(configurable_data['transport'], dict)
        and 'core_transport_models' not in configurable_data['transport']
        and 'pedestal_transport_models' not in configurable_data['transport']
    ):
      configurable_data['transport']['core_transport_models'] = {
          'prescribed': {'model_name': 'prescribed'}
      }
    if (
        isinstance(configurable_data['solver'], dict)
        and 'solver_type' not in configurable_data['solver']
    ):
      configurable_data['solver']['solver_type'] = 'linear'
    if 'time_step_calculator' not in configurable_data:
      configurable_data['time_step_calculator'] = {'calculator_type': 'chi'}
    elif (
        isinstance(configurable_data['time_step_calculator'], dict)
        and 'calculator_type' not in configurable_data['time_step_calculator']
    ):
      configurable_data['time_step_calculator']['calculator_type'] = 'chi'
    return configurable_data

  def _get_extended_lengyel_configs(
      self,
  ) -> list[extended_lengyel_config.ExtendedLengyelConfig]:
    if isinstance(
        self.edge,
        extended_lengyel_config.ExtendedLengyelConfig,
    ):
      return [self.edge]
    if isinstance(self.edge, edge_config.CombinedEdgeConfig):
      return [
          sub
          for sub in self.edge.sub_models.values()
          if isinstance(sub, extended_lengyel_config.ExtendedLengyelConfig)
      ]
    return []

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    core_transport_models = self.transport.core_transport_models.values()
    pedestal_transport_models = (
        self.transport.pedestal_transport_models.values()
    )
    using_nonlinear_transport_model = any(
        model.model_name in ['qualikiz', 'qlknn', 'CGM']
        for model in list(core_transport_models) + list(
            pedestal_transport_models
        )
    )
    using_linear_solver = isinstance(
        self.solver, solver_config.LinearThetaMethodConfig
    )

    # pylint: disable=g-long-ternary
    # pylint: disable=attribute-error
    initial_guess_mode_is_linear = (
        False
        if using_linear_solver
        else self.solver.initial_guess_mode == enums.InitialGuessMode.LINEAR  # pyrefly: ignore[missing-attribute]
    )

    if (
        using_nonlinear_transport_model
        and (using_linear_solver or initial_guess_mode_is_linear)
        and not self.solver.use_pereverzev
    ):
      logging.warning("""
          use_pereverzev=False in a configuration where setting
          use_pereverzev=True is recommended.

          A nonlinear transport model is used. However, a linear solver is also
          being used, either directly, or to provide an initial guess for a
          nonlinear solver.

          With this configuration, it is strongly recommended to set
          use_pereverzev=True to avoid numerical instability in the solver.
          """)
    return self

  @pydantic.model_validator(mode='after')
  def _check_psidot_and_evolve_current(self) -> Self:
    """Warns if psidot is provided but evolve_current is True."""
    if (
        self.profile_conditions.psidot is not None
        and self.numerics.evolve_current
    ):
      logging.warning("""
          profile_conditions.psidot input is ignored as numerics.evolve_current
          is True.

          Prescribed psidot is only applied when current diffusion is off.
          """)
    return self

  @pydantic.model_validator(mode='after')
  def _validate_initial_psi_source(self) -> Self:
    """Validates `profile_conditions.initial_psi_source` compatibility."""
    initial_psi_source = self.profile_conditions.initial_psi_source
    is_restarting = self.restart is not None and self.restart.do_restart
    if (
        initial_psi_source
        == core_profile_runtime_params_lib.InitialPsiSource.PROFILE_CONDITIONS
        and self.profile_conditions.psi is None
        and not is_restarting
    ):
      raise ValueError(
          'profile_conditions.psi must be provided when'
          " profile_conditions.initial_psi_source is 'profile_conditions'"
          ' (unless restarting from a file).'
      )
    if (
        initial_psi_source
        == core_profile_runtime_params_lib.InitialPsiSource.GEOMETRY
        and self.geometry.geometry_type == geometry.GeometryType.CIRCULAR
    ):
      raise ValueError(
          "profile_conditions.initial_psi_source='geometry' is not supported"
          ' for CircularGeometry.'
      )
    if (
        self.profile_conditions.psi is not None
        and initial_psi_source
        != core_profile_runtime_params_lib.InitialPsiSource.PROFILE_CONDITIONS
        and self.numerics.evolve_current
    ):
      logging.warning(
          'profile_conditions.psi is provided but will be ignored because'
          ' profile_conditions.initial_psi_source is %r and'
          ' numerics.evolve_current is True.',
          initial_psi_source.value,
      )
    return self

  @pydantic.model_validator(mode='after')
  def _check_pedestal_with_non_uniform_grid(self) -> Self:
    """Warns if a pedestal and non-uniform grid are used."""
    if self.pedestal.structure_model.model_name != 'no_pedestal':
      face_centers = self.geometry.get_face_centers()
      is_uniform = np.all(
          np.isclose(np.diff(face_centers), np.diff(face_centers)[0])
      )
      if not is_uniform:
        logging.warning("""
            Config has both a pedestal model and a non-uniform grid. Currently,
            the numerics of the non-uniform grid can cause pedestal gradients to
            leak into the core, leading to incorrect transport near the pedestal
            top. This can be mitigated by using a uniform grid around the
            pedestal top.
            """)
    return self

  @pydantic.model_validator(mode='after')
  def _validate_I_p_source(self) -> Self:
    """Validates `profile_conditions.I_p_source` compatibility."""
    i_p_source = self.profile_conditions.I_p_source
    if (
        i_p_source == core_profile_runtime_params_lib.IpSource.GEOMETRY
        and self.geometry.geometry_type == geometry.GeometryType.CIRCULAR
    ):
      raise ValueError(
          "profile_conditions.I_p_source='geometry' is not supported for"
          ' CircularGeometry.'
      )
    if (
        self.profile_conditions.I_p is not None
        and i_p_source
        != core_profile_runtime_params_lib.IpSource.PROFILE_CONDITIONS
    ):
      logging.warning(
          'profile_conditions.I_p is provided but will be ignored because'
          ' profile_conditions.I_p_source is %r.',
          i_p_source.value,
      )
    return self

  @pydantic.model_validator(mode='after')
  def _validate_pedestal_mode_and_internal_boundary_conditions(
      self,
  ) -> Self:
    """Validates that internal boundary conditions are not used with ADAPTIVE_TRANSPORT."""
    ibc = self.profile_conditions.internal_boundary_conditions
    if (
        self.pedestal.mode == pedestal_runtime_params.Mode.ADAPTIVE_TRANSPORT
        and not isinstance(ibc, ibc_config.NoIBCConfig)
    ):
      raise ValueError(
          'Internal boundary conditions cannot be configured when pedestal'
          ' mode is ADAPTIVE_TRANSPORT. In ADAPTIVE_TRANSPORT mode, transport'
          ' suppression governs the edge continuously across the entire grid,'
          ' which is incompatible with internal boundary conditions.'
      )
    return self

  @pydantic.model_validator(mode='after')
  def _check_edge_with_circular_geometry(self) -> Self:
    """Validates that edge models are not used with CircularGeometry."""
    if (
        self.edge is not None
        and self.geometry.geometry_type == geometry.GeometryType.CIRCULAR
    ):
      raise ValueError(
          'Edge models are not supported for use with CircularGeometry.'
      )
    return self

  @pydantic.model_validator(mode='after')
  def _validate_edge_and_impurity_mode(
      self,
  ) -> Self:
    """Ensures edge models use n_e_ratios impurity mode."""
    if (
        self.edge is not None
        and self.plasma_composition.impurity.impurity_mode != 'n_e_ratios'
    ):
      raise ValueError(
          'Edge models require'
          " plasma_composition.impurity_mode to be 'n_e_ratios'."
          f" Got '{self.plasma_composition.impurity.impurity_mode}'."
      )
    return self

  @pydantic.model_validator(mode='after')
  def _validate_edge_diverted_status(self) -> Self:
    """Validates diverted status configuration in edge model.

    Ensures that `diverted` is handled correctly based on geometry type:
    - For MEQ geometry: `diverted` must NOT be set (it's provided by MEQ).
    - For non-MEQ geometry: `diverted` MUST be set if edge model is used.
    """
    for el_config in self._get_extended_lengyel_configs():
      is_meq = self.geometry.geometry_type == geometry.GeometryType.MEQ
      diverted = el_config.diverted is not None

      if is_meq and diverted:
        raise ValueError(
            'Extended Lengyel edge model configuration error: `diverted`'
            ' must NOT be set when using MEQ geometry. MEQ geometry files'
            ' inherently provide diverted status information.'
        )

      if not is_meq and not diverted:
        raise ValueError(
            'Extended Lengyel edge model configuration error: `diverted`'
            ' MUST be set when using non-MEQ geometry (e.g. CHEASE, EQDSK,'
            ' IMAS). These geometry sources do not reliably provide diverted'
            ' status, so it must be explicitly configured in the edge model.'
        )

    return self

  @pydantic.model_validator(mode='after')
  def _validate_edge_core_impurity_consistency(self) -> Self:
    """Validates consistency between plasma composition and edge impurities."""
    # TODO(b/557540194): Generalize this validation to other edge models once
    # `EdgeModelConfig` exposes an instance-level hook to declare its
    # configured impurity species.
    for el_config in self._get_extended_lengyel_configs():
      core_species = set(self.plasma_composition.impurity.species.keys())
      edge_fixed = set(el_config.fixed_impurity_concentrations.keys())

      if (
          el_config.computation_mode
          == extended_lengyel_enums.ComputationMode.INVERSE
      ):
        if el_config.seed_impurity_weights is not None:
          edge_seed = set(el_config.seed_impurity_weights.keys())
        else:
          edge_seed = set()

        # Intersection check (should be empty)
        if not edge_fixed.isdisjoint(edge_seed):
          raise ValueError(
              'Edge fixed and seeded impurities must be disjoint. Overlap:'
              f' {edge_fixed.intersection(edge_seed)}'
          )

        edge_species = edge_fixed.union(edge_seed)
      else:  # FORWARD mode
        edge_species = edge_fixed

      if core_species != edge_species:
        raise ValueError(
            'Mismatch between core plasma composition impurities and edge'
            f' impurities. Core: {core_species}, Edge: {edge_species}.'
            f' Difference: {core_species.symmetric_difference(edge_species)}.'
            ' Likely reason: edge.fixed_impurity_concentrations and/or'
            ' edge.seed_impurity_weights do not match plasma_composition.'
            'impurity.species.'
        )

    return self

  @pydantic.model_validator(mode='after')
  def _validate_nonzero_n_e_ratios_at_lcfs(self) -> Self:
    """Validates that n_e_ratio profiles are non-zero at the LCFS.

    When the extended Lengyel edge model is active, core impurity profiles
    are rescaled by dividing by the profile value at the LCFS (rho_norm=1).
    If this value is zero, the rescaled profile will also be zero regardless
    of what the edge model computes, silently breaking edge-core coupling.

    This validator checks the raw TimeVaryingArray data for each impurity
    species, and raises an error if any time slice has a zero value at
    rho_norm=1.
    """
    # TODO(b/557540194): Generalize this validation to all edge models that
    # update impurities (`supported_boundary_conditions.n_impurities=True`).
    # Because `supported_boundary_conditions` is a ClassVar, doing so requires
    # an instance-level hook on `EdgeModelConfig` to declare which impurity
    # species it populates in `impurity_right_bc`.
    el_configs = self._get_extended_lengyel_configs()
    if not el_configs:
      return self
    impurity = self.plasma_composition.impurity
    if not isinstance(impurity, electron_density_ratios.ElectronDensityRatios):
      return self

    for el_config in el_configs:
      if (
          el_config.computation_mode
          == extended_lengyel_enums.ComputationMode.INVERSE
      ):
        seeded_species = (
            set(el_config.seed_impurity_weights.keys())
            if el_config.seed_impurity_weights
            else set()
        )
      else:
        seeded_species = set()

      # Fixed impurities with EDGE as source of truth.
      if (
          el_config.impurity_sot
          == extended_lengyel_enums.FixedImpuritySourceOfTruth.EDGE
      ):
        fixed_edge_species = set(el_config.fixed_impurity_concentrations.keys())
      else:
        fixed_edge_species = set()

      species_to_check = seeded_species | fixed_edge_species

      # update_n_impurities is a TimeVaryingScalarStep (step-interpolated).
      # We need to check its value at the same times as the species profile.
      update_n_impurities_time = el_config.update_n_impurities.time
      update_n_impurities_value = el_config.update_n_impurities.value

      for species_name, species_profile in impurity.species.items():
        if species_name not in species_to_check:
          continue
        if species_profile is None:
          continue
        for t, (_, values) in species_profile.value.items():
          # Sufficient to check the last rho_norm value, due to constant
          # extrapolation to the LCFS if rho_norm=1 is not directly included.
          if values[-1] <= 0.0:
            # Step-interpolate update_n_impurities at time t. Only raise if
            # the edge model would actually rescale impurities at this time.
            # searchsorted(side='right') - 1 finds the index of the last
            # update_n_impurities time point <= t, i.e. the step value active
            # at time t.
            idx = max(
                0,
                int(np.searchsorted(update_n_impurities_time, t, side='right'))
                - 1,
            )
            if not update_n_impurities_value[idx]:
              continue
            raise ValueError(
                f"Impurity species '{species_name}' has a zero or negative"
                f' n_e_ratio at rho_norm=1.0 (the LCFS) at time t={t}.'
                ' When the extended Lengyel edge model is active, it rescales'
                ' core impurity profiles by dividing by the LCFS value. A'
                ' zero LCFS value means the rescaled profile will remain'
                ' zero regardless of the edge model output, silently'
                ' breaking edge-core coupling. Please set a small positive'
                ' value at rho_norm=1.0 (the actual value will be overwritten'
                ' by the edge model).'
            )
    return self

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these ancestral models will be
    re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """

    old_mesh = self.geometry.build_provider.torax_mesh
    self._update_fields(x)
    new_mesh = self.geometry.build_provider.torax_mesh

    if old_mesh != new_mesh:
      # The grid has changed, e.g. due to a new n_rho.
      # Clear the cached properties of all submodels and update the grid.
      for model in self.submodels:
        model.clear_cached_properties()
      torax_pydantic.set_grid(self, new_mesh, mode='force')
    else:
      # Update the grid on any new models which are added and have not had their
      # grid set yet.
      torax_pydantic.set_grid(self, new_mesh, mode='relaxed')

  @pydantic.model_validator(mode='after')
  def _set_grid(self) -> Self:
    # Interpolated `TimeVaryingArray` objects require a mesh, only available
    # once the geometry provider is built. This could be done in the before
    # validator, but is harder than setting it after construction.
    mesh = self.geometry.build_provider.torax_mesh
    # Note that the grid could already be set, eg. if the config is serialized
    # and deserialized. In this case, we do not want to overwrite it nor fail
    # when trying to set it, which is why mode='relaxed'.
    torax_pydantic.set_grid(self, mesh, mode='relaxed')
    return self

  # This is primarily used for serialization, so the importer can check which
  # version of Torax was used to generate the serialized config.
  @pydantic.computed_field
  @property
  def torax_version(self) -> str:
    return version.TORAX_VERSION

  @pydantic.model_validator(mode='after')
  def _validate_icrh_minority_species_presence(self) -> Self:
    """Validates that ICRH minority species is present in plasma composition."""
    if (
        self.sources.icrh is not None
        and self.sources.icrh.mode
        == sources_runtime_params_lib.Mode.MODEL_BASED
    ):
      minority = self.sources.icrh.minority_species
      minority_present = (
          minority in self.plasma_composition.get_main_ion_names()
          or minority in self.plasma_composition.get_impurity_names()
      )
      if not minority_present:
        raise ValueError(
            f'The ICRH source requires "{minority}" to be present in the '
            'plasma composition (either as main ion or impurity).'
        )
    return self

  @pydantic.model_validator(mode='before')
  @classmethod
  def _remove_version_field(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if 'torax_version' in data:
        data = {k: v for k, v in data.items() if k != 'torax_version'}
    return data
