"""The BohmGyroBohmModel class."""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
from jax import numpy as jnp
from torax import constants as constants_module
from torax import geometry
from torax import jax_utils
from torax import state
from torax import versioning
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """
  chi_e_bohm_coeff: float
  chi_e_gyrobohm_coeff: float
  chi_i_bohm_coeff: float
  chi_i_gyrobohm_coeff: float
  d_face_c1: float
  d_face_c2: float

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
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
    # TODO: Any sanity checks required?

  def __post_init__(self):
    self.sanity_check()


class BohmGyroBohmModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

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

    Particle diffusivities
    ======================

    The electron diffusivity is given by:

    .. math::
      D_e = \zeta(\rho_{\text{tor}}) \frac{\chi_e \chi_i}{\chi_e + \chi_i}

    where :math:`\zeta(\rho_{\text{tor}}) = A_1 + (A_2 - A_1) \rho_{\text{tor}}` is a linear weighting function, with manually set coefficients :math:`A_1` and :math:`A_2`.

    Pinch velocities
    ================

    The electron pinch velocity is given by:

    .. math::
      v_{\text{in},e} = 0.5 \frac{D_e S_{text{flux}}^2}{V} \left( \frac{\text{d}V}{\text{d}\rho_\text{tor}})^{-1} \right) \sqrt{\frac{\Psi_{\text{tor, sep}}}{\pi B_{\text{geo}}}}

    Glossary of terms
    =================

    - :math:`\chi_e`: Total electron heat transport
    - :math:`\chi_i`: Total ion heat transport
    - :math:`D_e`: Electron particle diffusivity
    - :math:`v_{\text{in},e}`: Electron pinch velocity
    - :math:`\chi_{e, \text{B}}`: Bohm term of electron heat transport
    - :math:`\chi_{e, \text{gB}}`: Gyrobohm term of electron heat transport
    - :math:`\chi_{i, \text{B}}`: Bohm term of ion heat transport
    - :math:`\chi_{i, \text{gB}}`: Gyrobohm term of ion heat transport
    - :math:`\alpha_{e, \text{B}}`: Coefficient of Bohm term of electron heat transport
    - :math:`\alpha_{e, \text{gB}}`: Coefficient of gyro-Bohm term of electron heat transport
    - :math:`\alpha_{i, \text{B}}`: Coefficient of Bohm term of ion heat transport
    - :math:`\alpha_{i, \text{gB}}`: Coefficient of gyro-Bohm term of ion heat transport
    - :math:`a_\text{min}`: *?*
    - :math:`q`: Safety factor
    - :math:`e`: *?*
    - :math:`B_\text{ax}`: Magnetic field at the magnetic axis
    - :math:`n_e`: Electron density
    - :math:`B_\text{geo}`: ??
    - :math:`\Psi_\text{tor, sep}`: ??
    - :math:`p_e`: Electron pressure, :math:`p_e = n_e T_e` *?*
    - :math:`\rho_{\text{tor}}`: ??
    - :math:`T_e`: Electron temperature
    - :math:`A_1`: Value of :math:`\zeta` at :math:`\rho_{\text{tor}} = 0`
    - :math:`A_2`: Value of :math:`\zeta` at :math:`\rho_{\text{tor}} = 1`
    - :math:`S_{text{flux}}`: Flux surface area
    - :math:`V`: Flux surface volume

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

    assert isinstance(
            dynamic_runtime_params_slice.transport, DynamicRuntimeParams
        )

    # Collect useful variables
    constants = constants_module.CONSTANTS
    Te = core_profiles.temp_el.face_value()
    grad_Te = core_profiles.temp_el.face_grad()
    ne = core_profiles.ne.face_value()
    grad_ne = core_profiles.ne.face_grad()
    grad_pe = grad_ne * Te + ne * grad_Te            # Electron pressure (by chain rule)

    # Bohm term of heat transport
    chi_e_B = (
      geo.Rmaj * core_profiles.q_face ** 2
      / (constants.qe * geo.B0 * ne)
      * grad_pe
    )

    # Gyrobohm term of heat transport
    chi_e_gB = (
      jnp.sqrt(Te)
      / geo.B0 ** 2
      * grad_Te
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
    weighting = dynamic_runtime_params_slice.transport.d_face_c1 + (
        dynamic_runtime_params_slice.transport.d_face_c2
        - dynamic_runtime_params_slice.transport.d_face_c1
    ) * geo.rho_face_norm
    d_face_el = weighting * chi_e * chi_i / (chi_e + chi_i)

    # Pinch velocity
    v_face_el = (
      0.5 * d_face_el * geo.area_face ** 2
      / (geo.volume_face * geo.vpr_face)
    )

    return state.CoreTransport(
        chi_face_ion=chi_i,
        chi_face_el=chi_e,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self):
    # All BohmGyroBohmModels are equivalent and can hash the same
    return hash(('BohmGyroBohmModel', versioning.torax_hash))

  def __eq__(self, other):
    return isinstance(other, BohmGyroBohmModel)


def _default_bgb_builder() -> BohmGyroBohmModel:
  return BohmGyroBohmModel()


@dataclasses.dataclass(kw_only=True)
class BohmGyroBohmModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class BohmGyroBohmModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  builder: Callable[
      [],
      BohmGyroBohmModel,
  ] = _default_bgb_builder

  def __call__(
      self,
  ) -> BohmGyroBohmModel:
    return self.builder()
