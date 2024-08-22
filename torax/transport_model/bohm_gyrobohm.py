"""The BohmGyroBohmModel class."""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
from jax import numpy as jnp

from torax import constants as constants_module
from torax import geometry, state
from torax.config import config_args, runtime_params_slice
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
    """Extends the base runtime params with additional params for this model.

    See base class runtime_params.RuntimeParams docstring for more info.
    """

    chi_e_bohm_coeff: runtime_params_lib.TimeInterpolated = 8e-5
    chi_e_gyrobohm_coeff: runtime_params_lib.TimeInterpolated = 5e-6
    chi_i_bohm_coeff: runtime_params_lib.TimeInterpolated = 8e-5
    chi_i_gyrobohm_coeff: runtime_params_lib.TimeInterpolated = 5e-6
    d_face_c1: runtime_params_lib.TimeInterpolated = 1.0
    d_face_c2: runtime_params_lib.TimeInterpolated = 0.3

    def make_provider(
        self, torax_mesh: geometry.Grid1D | None = None
    ) -> RuntimeParamsProvider:
        return RuntimeParamsProvider(
            runtime_params_config=self,
            apply_inner_patch=config_args.get_interpolated_var_single_axis(
                self.apply_inner_patch
            ),
            De_inner=config_args.get_interpolated_var_single_axis(self.De_inner),
            Ve_inner=config_args.get_interpolated_var_single_axis(self.Ve_inner),
            chii_inner=config_args.get_interpolated_var_single_axis(self.chii_inner),
            chie_inner=config_args.get_interpolated_var_single_axis(self.chie_inner),
            rho_inner=config_args.get_interpolated_var_single_axis(self.rho_inner),
            apply_outer_patch=config_args.get_interpolated_var_single_axis(
                self.apply_outer_patch
            ),
            De_outer=config_args.get_interpolated_var_single_axis(self.De_outer),
            Ve_outer=config_args.get_interpolated_var_single_axis(self.Ve_outer),
            chii_outer=config_args.get_interpolated_var_single_axis(self.chii_outer),
            chie_outer=config_args.get_interpolated_var_single_axis(self.chie_outer),
            rho_outer=config_args.get_interpolated_var_single_axis(self.rho_outer),
            chi_e_bohm_coeff=config_args.get_interpolated_var_single_axis(
                self.chi_e_bohm_coeff
            ),
            chi_e_gyrobohm_coeff=config_args.get_interpolated_var_single_axis(
                self.chi_e_gyrobohm_coeff
            ),
            chi_i_bohm_coeff=config_args.get_interpolated_var_single_axis(
                self.chi_i_bohm_coeff
            ),
            chi_i_gyrobohm_coeff=config_args.get_interpolated_var_single_axis(
                self.chi_i_gyrobohm_coeff
            ),
            d_face_c1=config_args.get_interpolated_var_single_axis(self.d_face_c1),
            d_face_c2=config_args.get_interpolated_var_single_axis(self.d_face_c2),
        )


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    """Provides a RuntimeParams to use during time t of the sim."""

    runtime_params_config: RuntimeParams
    chi_e_bohm_coeff: runtime_params_lib.InterpolatedVarSingleAxis
    chi_e_gyrobohm_coeff: runtime_params_lib.InterpolatedVarSingleAxis
    chi_i_bohm_coeff: runtime_params_lib.InterpolatedVarSingleAxis
    chi_i_gyrobohm_coeff: runtime_params_lib.InterpolatedVarSingleAxis
    d_face_c1: runtime_params_lib.InterpolatedVarSingleAxis
    d_face_c2: runtime_params_lib.InterpolatedVarSingleAxis

    def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
        return DynamicRuntimeParams(
            chimin=self.runtime_params_config.chimin,
            chimax=self.runtime_params_config.chimax,
            Demin=self.runtime_params_config.Demin,
            Demax=self.runtime_params_config.Demax,
            Vemin=self.runtime_params_config.Vemin,
            Vemax=self.runtime_params_config.Vemax,
            apply_inner_patch=bool(self.apply_inner_patch.get_value(t)),
            De_inner=float(self.De_inner.get_value(t)),
            Ve_inner=float(self.Ve_inner.get_value(t)),
            chii_inner=float(self.chii_inner.get_value(t)),
            chie_inner=float(self.chie_inner.get_value(t)),
            rho_inner=float(self.rho_inner.get_value(t)),
            apply_outer_patch=bool(self.apply_outer_patch.get_value(t)),
            De_outer=float(self.De_outer.get_value(t)),
            Ve_outer=float(self.Ve_outer.get_value(t)),
            chii_outer=float(self.chii_outer.get_value(t)),
            chie_outer=float(self.chie_outer.get_value(t)),
            rho_outer=float(self.rho_outer.get_value(t)),
            smoothing_sigma=self.runtime_params_config.smoothing_sigma,
            smooth_everywhere=self.runtime_params_config.smooth_everywhere,
            chi_e_bohm_coeff=self.chi_e_bohm_coeff.get_value(t),
            chi_e_gyrobohm_coeff=self.chi_e_gyrobohm_coeff.get_value(t),
            chi_i_bohm_coeff=self.chi_i_bohm_coeff.get_value(t),
            chi_i_gyrobohm_coeff=self.chi_i_gyrobohm_coeff.get_value(t),
            d_face_c1=self.d_face_c1.get_value(t),
            d_face_c2=self.d_face_c2.get_value(t),
        )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
    """Dynamic runtime params for the BgB transport model."""

    chi_e_bohm_coeff: float
    chi_e_gyrobohm_coeff: float
    chi_i_bohm_coeff: float
    chi_i_gyrobohm_coeff: float
    d_face_c1: float
    d_face_c2: float

    def sanity_check(self):
        runtime_params_lib.DynamicRuntimeParams.sanity_check(self)

    def __post_init__(self):
        self.sanity_check()


class BohmGyroBohmModel(transport_model.TransportModel):
    """Calculates various coefficients related to particle transport according to the Bohm + gyro-Bohm Model."""

    def __init__(
        self,
    ):
        super().__init__()
        self._frozen = True

    def _call_implementation(
        self,
        dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> state.CoreTransport:
        r"""Calculates transport coefficients using the Bohm + gyro-Bohm Model.

        We use the implementation from [1], Section 3.3.

        Heat diffusivities
        ==================

        The heat diffusivities for electrons and ions are given by:

        .. math::
          \chi_e = \alpha_{e, \text{B}} \chi_{e, \text{B}} + \alpha_{e, \text{gB}} \chi_{e, \text{gB}}

        .. math::
          \chi_i = \alpha_{i, \text{B}} \chi_{i, \text{B}} + \alpha_{i, \text{gB}} \chi_{i, \text{gB}}

        where :math:`\alpha_{s, \text{B}}` and :math:`\alpha_{s, \text{gB}}` are the coefficients for the Bohm and gyro-Bohm contribution for species :math:`s` respectively.
        These are given by:

        .. math::
          \chi_{e, \text{B}}
            = 0.5 \chi_{i, \text{B}}
            = \frac{a_\text{min} q^2}{e B_\text{ax} n_e}
              \sqrt{
                \frac{\pi B_\text{geo}}{\Psi_\text{tor, sep}}
              }
              \left|
                \frac{\partial p_e}{\partial \rho_{\text{tor}}}
              \right|

        .. math::
          \chi_{e, \text{gB}}
            = 2 \chi_{i, \text{gB}}
            =  \frac{\sqrt{T_e}}{B_\text{ax}^2}
              \sqrt{
                \frac{\pi B_\text{geo}}{\Psi_\text{tor, sep}}
              }
              \left|
                \frac{\partial T_e}{\partial \rho_{\text{tor}}}
              \right|

        where :math:`a_\text{min}` is the minor radius, :math:`q` is the safety factor, :math:`e` is the elementary charge, :math:`B_\text{ax}` is the toroidal magnetic field at the magnetic axis, :math:`n_e` is the electron density, :math:`B_\text{geo}` is the geometric toroidal magnetic field, :math:`\Psi_\text{tor, sep}` is the toroidal flux at the separatrix, :math:`p_e` is the electron pressure, and :math:`T_e` is the electron temperature.

        Electron diffusivity
        ====================

        .. math::
          D_e = \eta \frac{\chi_e \chi_i}{\chi_e + \chi_i}

        where :math:`\eta` is a weighting factor given by:

        .. math::

          \eta = c_1 + (c_2 - c_1) \rho_{\text{tor}}

        where :math:`c_1` and :math:`c_2` are constants.


        Electron convectivity
        =====================

        .. math::
          v_e = \frac{1}{2} \frac{D_e A^2}{V \frac{dV}{d\rho}}

        where :math:`A` and :math:`V` are the area and volume of the flux surface respectively.

        References:
        ===========

        [1]: https://doi.org/10.1088/1741-4326/ad6ea2

        Args:
          dynamic_runtime_params_slice: Input runtime parameters that can change
            without triggering a JAX recompilation.
          geo: Geometry of the torus.
          core_profiles: Core plasma profiles.

        Returns:
          coeffs: The transport coefficients
        """
        # Many variables throughout this function are capitalized based on physics
        # notational conventions rather than on Google Python style
        # pylint: disable=invalid-name
        assert isinstance(dynamic_runtime_params_slice.transport, DynamicRuntimeParams)

        # Bohm term of heat transport
        chi_e_B = (
            geo.rmid_face
            * core_profiles.q_face**2
            / (constants_module.CONSTANTS.qe * geo.B0 * core_profiles.ne.face_value())
            * (
                core_profiles.ne.face_grad() * core_profiles.temp_el.face_value()
                + core_profiles.temp_el.face_grad() * core_profiles.ne.face_value()
            )
            / geo.rho_b
        )

        # Gyrobohm term of heat transport
        chi_e_gB = (
            jnp.sqrt(dynamic_runtime_params_slice.plasma_composition.Ai / 2)
            * jnp.sqrt(core_profiles.temp_el.face_value() * 1e3)
            / geo.B0**2
            * core_profiles.temp_el.face_grad()
            / geo.rho_b
        )

        chi_i_B = 2 * chi_e_B
        chi_i_gB = 0.5 * chi_e_gB

        # Total heat transport
        chi_i = (
            dynamic_runtime_params_slice.transport.chi_i_bohm_coeff * chi_i_B
            + dynamic_runtime_params_slice.transport.chi_i_gyrobohm_coeff * chi_i_gB
        )
        chi_e = (
            dynamic_runtime_params_slice.transport.chi_e_bohm_coeff * chi_e_B
            + dynamic_runtime_params_slice.transport.chi_e_gyrobohm_coeff * chi_e_gB
        )

        # Electron diffusivity
        weighting = (
            dynamic_runtime_params_slice.transport.d_face_c1
            + (
                dynamic_runtime_params_slice.transport.d_face_c2
                - dynamic_runtime_params_slice.transport.d_face_c1
            )
            * geo.rho_face_norm
        )
        d_face_el = weighting * chi_e * chi_i / (chi_e + chi_i)

        # Pinch velocity
        v_face_el = (
            0.5 * d_face_el * geo.area_face**2 / (geo.volume_face * geo.vpr_face)
        )

        return state.CoreTransport(
            chi_face_ion=chi_i,
            chi_face_el=chi_e,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def __hash__(self):
        # All BohmGyroBohmModels are equivalent and can hash the same
        return hash("BohmGyroBohmModel")

    def __eq__(self, other):
        return isinstance(other, BohmGyroBohmModel)


def _default_bgb_builder() -> BohmGyroBohmModel:
    return BohmGyroBohmModel()


@dataclasses.dataclass(kw_only=True)
class BohmGyroBohmModelBuilder(transport_model.TransportModelBuilder):
    """Builds a class BohmGyroBohmModel."""

    runtime_params: RuntimeParams = dataclasses.field(default_factory=RuntimeParams)

    builder: Callable[
        [],
        BohmGyroBohmModel,
    ] = _default_bgb_builder

    def __call__(
        self,
    ) -> BohmGyroBohmModel:
        return self.builder()
