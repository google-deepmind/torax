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
  # Coefficient of Bohm term of χ_e
  chi_e_bohm_coeff: float = 8e-5
  # Coefficient of gyro-Bohm term of χ_e
  chi_e_gyrobohm_coeff: float = 7e-2
  # Coefficient of Bohm term of χ_i
  chi_i_bohm_coeff: float = 1.6e-4
  # Coefficient of gyro-Bohm term of χ_i
  chi_i_gyrobohm_coeff: float = 1.75e-2
  # Constant neoclassical transport term
  neoclassical_const: float = 1e-3
  # Electron particle diffusion = chi_d_coeff * χ_e
  chi_d_coeff: float = 0.2
  # Electron particle convection = chi_v_coeff * χ_e
  chi_v_coeff: float = -1.0

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
  neoclassical_const: float
  chi_d_coeff: float
  chi_v_coeff: float

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
    """Calculates transport coefficients using the Bohm + gyro-Bohm Model.

    TODO: Fill in equation
    The model is given by [1]:

      chi_e = chi_e_bohm_coeff * ( ... ) + chi_e_gyrobohm_coeff * ( ... ) + neoclassical_const
      chi_i = chi_i_bohm_coeff * ( ... ) + chi_i_gyrobohm_coeff * ( ... ) + neoclassical_const
      d_e = chi_d_coeff * chi_e
      v_e = chi_v_coeff * chi_e

    [1]: Erba et al. (1998), "Validation of a new mixed Bohm/gyro-Bohm model for
         electron and ion heat transport against the ITER, Tore Supra and START
         database discharges", Nuclear Fusion 38 1013.

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
    # Define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5
    constants = constants_module.CONSTANTS
    Te = core_profiles.temp_el.face_value()
    grad_Te = core_profiles.temp_el.face_grad(rmid)
    ne = core_profiles.ne.face_value()
    grad_ne = core_profiles.ne.face_grad(rmid)
    a = geo.Rmaj / geo.Rmin

    # Bohm term of heat transport
    # TODO: check, particularly grad_pe/pe definition, and whether this is correct B
    chi_B = (
      Te * a * core_profiles.q_face ** 2
      / (constants.qe * geo.B0)
      * (grad_Te / Te + grad_ne / ne)
    )

    # Gyrobohm term of heat transport
    # TODO: check, particularly units conversion, and whether this is correct B
    chi_gB = (
        (dynamic_runtime_params_slice.plasma_composition.Ai * constants.mp
        * Te * constants.keV2J) ** 0.5
        * grad_Te * constants.keV2J
        / (constants.qe * geo.B0) ** 2
    )

    # Total heat transport
    chi_i = (
      dynamic_runtime_params_slice.transport.chi_i_bohm_coeff * chi_B
      + dynamic_runtime_params_slice.transport.chi_i_gyrobohm_coeff * chi_gB
      + dynamic_runtime_params_slice.transport.neoclassical_const
    )
    chi_e = (
      dynamic_runtime_params_slice.transport.chi_e_bohm_coeff * chi_B
      + dynamic_runtime_params_slice.transport.chi_e_gyrobohm_coeff * chi_gB
      + dynamic_runtime_params_slice.transport.neoclassical_const
    )

    # Set electron diffusivity from electron heat transport
    d_face_el = dynamic_runtime_params_slice.transport.chi_d_coeff * chi_e
    # Set electron convectivity from electron heat transport
    v_face_el = dynamic_runtime_params_slice.transport.chi_v_coeff * chi_e

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
