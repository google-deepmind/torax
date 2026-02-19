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
"""EPEDNN-mit pedestal model.

This model is only valid for the SPARC parameter space, as specified in
https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using this model.
"""

import dataclasses
import functools
import pathlib
from typing import Any, Final, TypeAlias
from epednn_mit.models.sparc import jax_model as epednn_mit_jax_model
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.physics import formulas
from typing_extensions import override

EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]

_INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Ip": (1.6, 14.3),
    "Bt": (7.2, 12.2),
    "R": (1.85, 1.85),
    "a": (0.57, 0.57),
    "kappa": (1.53, 2.29),
    "delta": (0.39, 0.59),
    "neped": (2.84, 90.235),
    "betan": (0.8, 1.6),
    "zeff": (1.3, 2.5),
}


def _check_input_bounds(
    epednn_mit_inputs: jax.Array,
) -> None:
  """Checks that the EPEDNN-mit inputs are within the bounds."""
  for i, (key, (lower, upper)) in enumerate(_INPUT_BOUNDS.items()):
    if not (lower <= epednn_mit_inputs[i] <= upper):
      raise ValueError(
          f"EPEDNN-mit input {key} is out of bounds of the training"
          f" distribution. Value is {epednn_mit_inputs[i]}, but"
          f" bounds are [{lower}, {upper}]."
      )


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params_lib.RuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_ped: array_typing.FloatScalar
  T_i_T_e_ratio: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(
    set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
):
  """Pedestal model using EPEDNN-mit to predict pedestal pressure and width."""

  def _prepare_epednn_mit_inputs(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> jax.Array:
    """Prepares the inputs for EPEDNN-mit."""
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    _, _, beta_N = formulas.calculate_betas(core_profiles, geo)

    # TODO(b/323504363): We really want the Z_eff at the pedestal top;
    # however, the location of the pedestal top is an *output* of the model.
    # Currently, we instead compute a density-weighted volume average of Z_eff
    # over the entire domain.
    Z_eff_average = math_utils.volume_integration(
        core_profiles.Z_eff * core_profiles.n_e.value, geo
    ) / math_utils.volume_integration(core_profiles.n_e.value, geo)

    inputs = jnp.array([
        core_profiles.Ip_profile_face[-1] * 1e-6,  # [MA]
        geo.B_0,  # [T]
        geo.R_major,  # [m]
        geo.a_minor,  # [m]
        geo.elongation_face[-1],  # []
        geo.delta_face[-1],  # []
        runtime_params.pedestal.n_e_ped * 1e-19,  # [10^19 m^-3]
        beta_N,  # [%]
        Z_eff_average,  # [C]
    ])
    _check_input_bounds(inputs)
    return inputs

  @functools.cached_property
  def _get_model(
      self,
  ) -> tuple[
      EPEDNNmitStats,
      EPEDNNmitParams,
      epednn_mit_jax_model.EPEDNNmitEnsemble,
  ]:
    """Returns the EPEDNN-mit model and parameters."""
    model_dir = pathlib.Path(epednn_mit_jax_model.__file__).parent
    model_weights = sorted(model_dir.glob("epednn_mit_sparc_*.pkl"))
    stats, params = epednn_mit_jax_model.load_ensemble_params_from_pickle(
        model_weights
    )
    model = epednn_mit_jax_model.EPEDNNmitEnsemble()
    return stats, params, model

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    # Get P_ped and rho_norm_ped_top from EPEDNN-mit.
    stats, params, model = self._get_model()
    epednn_mit_inputs = self._prepare_epednn_mit_inputs(
        runtime_params, geo, core_profiles
    )
    P_ped_kPa, pedestal_width_psi_norm = model.apply(
        params, epednn_mit_inputs, **stats
    )

    # Convert pedestal width to rho_norm
    psi_norm = (core_profiles.psi.value - core_profiles.psi.value[0]) / (
        core_profiles.psi.value[-1] - core_profiles.psi.value[0]
    )
    psi_norm_ped_top = 1.0 - pedestal_width_psi_norm
    rho_norm_ped_top = jnp.interp(psi_norm_ped_top, psi_norm, geo.rho_norm)

    # Convert P_ped from kPa to Pa.
    P_ped = P_ped_kPa * 1e3

    # Use the set_pped_tpedratio_nped model to calculate the pedestal profiles.
    super_runtime_params = set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=runtime_params.pedestal.set_pedestal,
        P_ped=P_ped,
        n_e_ped=runtime_params.pedestal.n_e_ped,
        T_i_T_e_ratio=runtime_params.pedestal.T_i_T_e_ratio,
        rho_norm_ped_top=rho_norm_ped_top,
        n_e_ped_is_fGW=runtime_params.pedestal.n_e_ped_is_fGW,
    )
    modified_runtime_params = dataclasses.replace(
        runtime_params, pedestal=super_runtime_params
    )
    return super()._call_implementation(
        modified_runtime_params, geo, core_profiles
    )
