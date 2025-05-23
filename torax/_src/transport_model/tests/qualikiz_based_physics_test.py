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

"""Tests for physics calculations in QualikizBasedTransportModel."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from torax._src import constants as constants_module
from torax._src import state as state_module
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_module
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import quasilinear_transport_model

# pylint: disable=invalid-name

# Define a minimal concrete implementation of QualikizBasedTransportModel for testing
class _TestQualikizModel(qualikiz_based_transport_model.QualikizBasedTransportModel):
  """Minimal concrete QLK-based model for testing _prepare_qualikiz_inputs."""

  def predict(
      self,
      inputs: qualikiz_based_transport_model.QualikizInputs,
  ) -> quasilinear_transport_model.ModelOutput:
    # This method is not used when testing _prepare_qualikiz_inputs directly.
    raise NotImplementedError('Predict not implemented for this test model.')


class QualikizBasedPhysicsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.constants = constants_module.CONSTANTS
    self.n_cells = 4  # Number of grid cells
    self.rho_norm = jnp.linspace(0, 1, self.n_cells + 1)[1:-1] # Should be cell centers
    self.drho_norm = 1.0 / self.n_cells
    # Corrected rho_norm for cell centers:
    self.rho_norm = jnp.linspace(
        self.drho_norm / 2, 1.0 - self.drho_norm / 2, self.n_cells
    )
    self.rho_face_norm = jnp.linspace(0, 1, self.n_cells + 1)

    # Mock Geometry
    self.geo = mock.create_autospec(geometry_module.Geometry, instance=True)
    self.geo.R_major = 1.7  # meters
    self.geo.a_minor = 0.5  # meters
    self.geo.B_0 = 2.0  # Tesla
    self.geo.rho_norm = self.rho_norm
    self.geo.rho_face_norm = self.rho_face_norm
    self.geo.drho_norm = self.drho_norm
    # dr/drho_norm_face, assume constant for simplicity = a_minor
    self.geo.dr_drho_norm_face = jnp.full_like(self.rho_face_norm, self.geo.a_minor)
    # For rmid and rmid_face
    # Let R_out = R_major + r, R_in = R_major - r
    # r = rho_norm * a_minor
    self.geo.R_out = self.geo.R_major + self.rho_norm * self.geo.a_minor
    self.geo.R_in = self.geo.R_major - self.rho_norm * self.geo.a_minor
    self.geo.R_out_face = self.geo.R_major + self.rho_face_norm * self.geo.a_minor
    self.geo.R_in_face = self.geo.R_major - self.rho_face_norm * self.geo.a_minor
    # For psi_calculations.calc_s_rmid, psi needs to be a CellVariable
    # Let's make a simple psi for s_rmid, e.g. psi ~ rho^2
    psi_values = (self.rho_norm**2)
    self.psi_cv = cell_variable.CellVariable(
        value=psi_values,
        dr=self.drho_norm,
        left_face_grad_constraint=0.0,
        right_face_constraint=1.0, # dummy
        name='psi_test',
    )
    # calc_s_rmid also needs geo.F_hires, geo.g2g3_over_rhon_hires, geo.rho_hires_norm
    # This makes mocking geo difficult. Let's use a simpler approach for smag if possible
    # or accept that smag might be inaccurate in this test if psi setup is too simple.
    # For now, we are not testing smag itself, but its effect on other QLK inputs.
    # The _prepare_qualikiz_inputs calls calc_s_rmid which needs a full geo.
    # For now, let's mock calc_s_rmid.

    # Mock transport parameters
    self.transport_params = qualikiz_based_transport_model.DynamicRuntimeParams(
        collisionality_multiplier=1.0,
        avoid_big_negative_s=False,
        smag_alpha_correction=False, # True might be harder to test without full alpha calc
        q_sawtooth_proxy=False,
        # from base class quasilinear_transport_model.DynamicRuntimeParams
        include_ITG=True,
        include_TEM=True,
        include_ETG=True,
        saturation_rule=quasilinear_transport_model.SaturationRule.FLUX_MATCHING_MAXWELL_SAT,
        scale_TE_ITG=False,
        scale_TE_TEM=False,
        gamma_ITG_multiplier=1.0,
        gamma_TEM_multiplier=1.0,
        gamma_ETG_multiplier=1.0,
        chi_ITG_multiplier=1.0,
        chi_TEM_multiplier=1.0,
        chi_ETG_multiplier=1.0,
        D_ITG_multiplier=1.0,
        D_TEM_multiplier=1.0,
        D_ETG_multiplier=1.0,
        include_pfe_TE=False,
        pfe_multiplier=1.0,
        fixed_efold_length_TEM=None,
        fixed_efold_length_ITG=None,
        fixed_efold_length_ETG=None,
    )
    self.density_reference = 1e20  # m^-3
    self.Z_eff_face = jnp.full_like(self.rho_face_norm, 1.5)

    self.model = _TestQualikizModel()

  def _create_core_profiles(
      self,
      omega_tor_coeffs,  # e.g. [val_at_axis, val_at_edge] for linear profile
      Ti_coeffs,
      ni_coeffs,
      Zi_val = 1.0, # constant Zi
      q_val = 2.0, # constant q
  ) -> state_module.CoreProfiles:
    """Helper to create CoreProfiles with simple linear profiles."""

    def linear_profile(coeffs, x):
      return coeffs[0] * (1 - x) + coeffs[1] * x

    omega_tor_arr = linear_profile(jnp.array(omega_tor_coeffs), self.rho_norm)
    Ti_arr = linear_profile(jnp.array(Ti_coeffs), self.rho_norm) # keV
    ni_arr = linear_profile(jnp.array(ni_coeffs), self.rho_norm) # normalized to density_reference

    omega_tor_cv = cell_variable.CellVariable(
        value=omega_tor_arr, dr=self.drho_norm, name='omega_tor',
        left_face_grad_constraint= (omega_tor_coeffs[1] - omega_tor_coeffs[0]) / (1.0/self.geo.a_minor) if self.geo.a_minor!=0 else 0.0, # approx
        right_face_constraint=omega_tor_coeffs[1] * 0.9 # made up, just to have one
    )
    # For simplicity in hand calculation, let's use zero grad on left, fixed value on right for BCs
    # The actual BCs affect face_value() and face_grad() at boundaries.
    # For internal points, central differences are used.
    # Let's make BCs simple:
    def _make_cv(val_arr, name, bc_val_right, grad_left=0.0):
        # val_arr is on cell centers.
        # Estimate grad for left BC from first two points if not zero.
        # However, the implementation uses left_face_grad_constraint.
        return cell_variable.CellVariable(
            value=val_arr, dr=self.drho_norm, name=name,
            left_face_grad_constraint=grad_left, # grad wrt rho_norm
            right_face_constraint=bc_val_right
        )

    omega_tor_cv = _make_cv(omega_tor_arr, 'omega_tor', omega_tor_coeffs[1])
    Ti_cv = _make_cv(Ti_arr, 'Ti', Ti_coeffs[1])
    # ni_abs = ni_arr * self.density_reference. For n_i CellVariable, value is normalized.
    ni_cv = _make_cv(ni_arr, 'ni', ni_coeffs[1])

    # Dummy impurity profile (zero for simplicity in Pi calculation)
    n_impurity_arr = jnp.zeros_like(self.rho_norm)
    n_impurity_cv = _make_cv(n_impurity_arr, 'n_impurity', 0.0)

    # Electron profiles (needed for some internal calculations, e.g. nu_star, alpha)
    # Let Te = Ti, ne = ni for simplicity.
    Te_cv = _make_cv(Ti_arr, 'Te', Ti_coeffs[1]) # Assuming Te = Ti
    ne_cv = _make_cv(ni_arr, 'ne', ni_coeffs[1]) # Assuming ne = ni for Z_eff=1 if Zi=1 and n_imp=0

    # Z_i, A_i, Z_impurity, A_impurity
    Z_i_arr = jnp.full_like(self.rho_norm, Zi_val)
    # Z_i_face from cell values, simple average for now, or from CellVariable.face_value() if complex.
    # _get_charge_states in getters.py uses T_e to calculate Z_i if ion_mixture is provided.
    # For this test, we directly provide Z_i values.
    Z_i_face_arr = jnp.full_like(self.rho_face_norm, Zi_val)

    # Mock q_face and s_face (not directly used in ator, Er, omegaExB, but needed by QLKInputs)
    q_face_arr = jnp.full_like(self.rho_face_norm, q_val)
    s_face_arr = jnp.zeros_like(self.rho_face_norm) # e.g. zero shear for simplicity

    # Other CoreProfile fields (can be zero/default if not affecting the tested physics)
    # psidot, sigma, j_total, Ip_profile_face can be zeros.
    # A_i, A_impurity
    # v_loop_lcfs
    zero_cv = _make_cv(jnp.zeros_like(self.rho_norm), 'zeros', 0.0)

    return state_module.CoreProfiles(
        T_i=Ti_cv, T_e=Te_cv, # Assuming T_e = T_i for simplicity
        psi=self.psi_cv, # Defined in setUp
        psidot=zero_cv,
        n_e=ne_cv, # Assuming n_e approx n_i for this test if Z_eff is low.
        n_i=ni_cv,
        n_impurity=n_impurity_cv,
        q_face=q_face_arr,
        s_face=s_face_arr, # calc_s_rmid will be mocked. This is standard s_face.
        density_reference=jnp.asarray(self.density_reference),
        v_loop_lcfs=0.0,
        Z_i=Z_i_arr, Z_i_face=Z_i_face_arr, A_i=2.0, # Deuterium
        Z_impurity=jnp.zeros_like(self.rho_norm), # Assuming no impurity for Z_eff related calcs if n_imp=0
        Z_impurity_face=jnp.zeros_like(self.rho_face_norm), A_impurity=12.0, # Carbon
        sigma=jnp.zeros_like(self.rho_norm), sigma_face=jnp.zeros_like(self.rho_face_norm),
        j_total=jnp.zeros_like(self.rho_norm), j_total_face=jnp.zeros_like(self.rho_face_norm),
        Ip_profile_face=jnp.zeros_like(self.rho_face_norm),
        omega_tor=omega_tor_cv,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='base_case',
          omega_tor_coeffs=[100.0, 10.0],  # rad/s, linear profile from 100 at r=0 to 10 at r=a
          Ti_coeffs=[2.0, 0.5],  # keV, linear
          ni_coeffs=[1.0, 0.2],  # x1e20 m^-3 (normalized), linear
          Zi_val=1.0, q_val=2.0,
          # Expected values to be calculated manually based on these inputs
      ),
      dict(
          testcase_name='zero_rotation',
          omega_tor_coeffs=[0.0, 0.0],
          Ti_coeffs=[2.0, 0.5], ni_coeffs=[1.0, 0.2],
          Zi_val=1.0, q_val=2.0,
      ),
      dict(
          testcase_name='solid_body_rotation', # ator should be zero
          omega_tor_coeffs=[50.0, 50.0], # Constant omega_tor
          Ti_coeffs=[2.0, 0.5], ni_coeffs=[1.0, 0.2],
          Zi_val=1.0, q_val=2.0,
      ),
      dict(
          testcase_name='no_pressure_gradient', # dPi/dr = 0
          omega_tor_coeffs=[100.0, 10.0],
          Ti_coeffs=[1.0, 1.0], # Flat Ti
          ni_coeffs=[0.5, 0.5], # Flat ni
          Zi_val=1.0, q_val=2.0,
      ),
  )
  @mock.patch('torax._src.physics.psi_calculations.calc_s_rmid') # Mock to avoid full geo needs for s_rmid
  def test_ator_Er_exbshear_calculations(
      self, mock_calc_s_rmid,
      omega_tor_coeffs, Ti_coeffs, ni_coeffs, Zi_val, q_val,
  ):
    mock_calc_s_rmid.return_value = jnp.zeros_like(self.rho_face_norm) # smag = 0 for these tests

    core_profiles = self._create_core_profiles(
        omega_tor_coeffs, Ti_coeffs, ni_coeffs, Zi_val, q_val
    )

    # Call the method to get QualikizInputs
    qualikiz_inputs_out = self.model._prepare_qualikiz_inputs(
        Z_eff_face=self.Z_eff_face, # Defined in setUp
        density_reference=self.density_reference,
        transport=self.transport_params, # Defined in setUp
        geo=self.geo,
        core_profiles=core_profiles,
    )

    # --- Calculate expected ator ---
    # ator = (R_major / omega_tor_face) * (domega_tor / dr_face)
    # domega_tor / dr_face = omega_tor_cv.face_grad() / geo.dr_drho_norm_face
    # omega_tor_cv.face_grad() is domega_tor/drho_norm on faces.
    # For linear profile omega(rho) = c0 * (1-rho) + c1 * rho = c0 + (c1-c0)*rho
    # domega/drho = c1-c0. This is constant.
    # So omega_tor_cv.face_grad() should be approx (omega_tor_coeffs[1] - omega_tor_coeffs[0])
    # except at boundaries due to BCs.
    # from_one_profile in implementation is more robust.
    # Let's use the same logic as in implementation for expected values for consistency.

    rmid_cell = (self.geo.R_out - self.geo.R_in) * 0.5
    omega_tor_norm_log_grad_expected = quasilinear_transport_model.NormalizedLogarithmicGradients.from_one_profile(
        profile=core_profiles.omega_tor,
        radial_coordinate=rmid_cell,
        reference_length=self.geo.R_major,
    )
    expected_ator = omega_tor_norm_log_grad_expected.lref_over_lx_face

    # Handle case where omega_tor is zero for ator (should be zero)
    if np.allclose(omega_tor_coeffs[0], 0.0) and np.allclose(omega_tor_coeffs[1], 0.0):
        expected_ator = jnp.zeros_like(expected_ator)
    # Handle solid body rotation for ator (should be zero)
    if np.allclose(omega_tor_coeffs[0], omega_tor_coeffs[1]):
        expected_ator = jnp.zeros_like(expected_ator)


    # --- Calculate expected Er and ExB shear rate ---
    # Follow the steps in _prepare_qualikiz_inputs

    # Ion pressure gradient (dPi/drho_norm)
    n_i_face = core_profiles.n_i.face_value()
    n_impurity_face = core_profiles.n_impurity.face_value()
    T_i_face = core_profiles.T_i.face_value()
    dn_i_drhon = core_profiles.n_i.face_grad()
    dn_impurity_drhon = core_profiles.n_impurity.face_grad()
    dT_i_drhon = core_profiles.T_i.face_grad()

    expected_dp_i_drhon_face = (
        ( (n_i_face + n_impurity_face) * dT_i_drhon + (dn_i_drhon + dn_impurity_drhon) * T_i_face )
        * self.density_reference * self.constants.keV2J
    )

    # Radial Electric Field (Er)
    dr_drho_norm_val = self.geo.dr_drho_norm_face
    expected_dp_i_dr_face = expected_dp_i_drhon_face / (dr_drho_norm_val + self.constants.eps)

    rmid_face = (self.geo.R_out_face - self.geo.R_in_face) * 0.5
    q_face_safe = jnp.where(jnp.abs(core_profiles.q_face) < self.constants.eps, self.constants.eps, core_profiles.q_face)
    q_face_safe = jnp.where(jnp.abs(q_face_safe) < self.constants.eps, self.constants.eps, q_face_safe)

    expected_B_theta_face = self.geo.B_0 * rmid_face / (q_face_safe * self.geo.R_major + self.constants.eps)

    omega_tor_face_val = core_profiles.omega_tor.face_value()
    Vphi_Btheta_term_expected = (omega_tor_face_val * self.geo.R_major) * expected_B_theta_face

    n_i_abs_face_expected = n_i_face * self.density_reference
    safe_denom_Er_expected = core_profiles.Z_i_face * n_i_abs_face_expected * self.constants.elementary_charge + self.constants.eps

    expected_Er_face = (1 / safe_denom_Er_expected) * expected_dp_i_dr_face - Vphi_Btheta_term_expected

    # ExB Shearing Rate (omega_ExB)
    X_denom_expected = rmid_face * expected_B_theta_face + self.constants.eps
    X_expected = expected_Er_face / X_denom_expected

    dX_drho_norm_expected = jnp.gradient(X_expected, self.geo.rho_face_norm)
    dX_dr_expected = dX_drho_norm_expected / (dr_drho_norm_val + self.constants.eps)

    omega_ExB_prefactor_expected = (self.geo.R_major * expected_B_theta_face) / (self.geo.B_0 + self.constants.eps)
    expected_exb_shear_rate = jnp.abs(omega_ExB_prefactor_expected * dX_dr_expected)

    # Assertions
    # Using a slightly larger tolerance due to numerical differentiation and potential mock inaccuracies.
    rtol = 1e-4; atol = 1e-5

    np.testing.assert_allclose(
        qualikiz_inputs_out.ator, expected_ator, rtol=rtol, atol=atol,
        err_msg=f"Testcase {self.testcase_name}: ator mismatch."
    )
    np.testing.assert_allclose(
        qualikiz_inputs_out.exb_shear_rate, expected_exb_shear_rate, rtol=rtol, atol=atol,
        err_msg=f"Testcase {self.testcase_name}: exb_shear_rate mismatch."
    )


if __name__ == '__main__':
  absltest.main()
