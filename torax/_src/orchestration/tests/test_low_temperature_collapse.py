"""Test for low temperature collapse detection."""

import jax.numpy as jnp
from absl.testing import absltest
from torax._src import state
from torax._src.config import numerics as numerics_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing


class LowTemperatureCollapseTest(absltest.TestCase):
    """Tests for low temperature collapse error detection."""

    def test_low_temperature_triggers_error(self):
        """Test that temperatures below threshold trigger LOW_TEMPERATURE_COLLAPSE error."""
        
        # Create a simple circular geometry
        geo = circular_geometry.build_circular_geometry(
            n_rho=25,
            elongation_LCFS=1.0,
            R_major=6.2,
            a_minor=2.0,
            B_0=5.3,
            hires_factor=4,
        )
        
        # Create numerics config with minimum temperature threshold
        numerics = numerics_lib.Numerics(
            minimum_temperature_eV=50.0,  # 50 eV threshold
        )
        
        # Build minimal core profiles with very low temperature
        # 0.01 eV = 1e-5 keV (well below 50 eV threshold)
        low_temp_keV = 1e-5
        
        # Get the shape for cell values
        n_cells = geo.rho.shape[0] - 1
        
        # Create cell variables with zero gradient boundary condition
        T_e = cell_variable.CellVariable(
            value=jnp.full(n_cells, low_temp_keV),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        T_i = cell_variable.CellVariable(
            value=jnp.full(n_cells, low_temp_keV),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        # Create dummy n_e, n_i, n_impurity with reasonable values
        n_e = cell_variable.CellVariable(
            value=jnp.full(n_cells, 1e20),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        n_i = cell_variable.CellVariable(
            value=jnp.full(n_cells, 1e20),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        n_impurity = cell_variable.CellVariable(
            value=jnp.zeros(n_cells),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        psi = cell_variable.CellVariable(
            value=jnp.zeros(n_cells),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        psidot = cell_variable.CellVariable(
            value=jnp.zeros(n_cells),
            dr=geo.drho_norm,
            right_face_grad_constraint=jnp.array(0.0),
            right_face_constraint=None,
        )
        
        # Build CoreProfiles
        core_profiles = state.CoreProfiles(
            T_i=T_i,
            T_e=T_e,
            psi=psi,
            psidot=psidot,
            n_e=n_e,
            n_i=n_i,
            n_impurity=n_impurity,
            impurity_fractions={},
            q_face=jnp.ones(geo.rho_face.shape),
            s_face=jnp.ones(geo.rho_face.shape),
            v_loop_lcfs=jnp.array(0.0),
            Z_i=jnp.ones(n_cells),
            Z_i_face=jnp.ones(geo.rho_face.shape),
            A_i=jnp.array(2.0),
            Z_impurity=jnp.ones(n_cells),
            Z_impurity_face=jnp.ones(geo.rho_face.shape),
            A_impurity=jnp.ones(n_cells),
            A_impurity_face=jnp.ones(geo.rho_face.shape),
            Z_eff=jnp.ones(n_cells),
            Z_eff_face=jnp.ones(geo.rho_face.shape),
            sigma=jnp.ones(n_cells),
            sigma_face=jnp.ones(geo.rho_face.shape),
            j_total=jnp.zeros(n_cells),
            j_total_face=jnp.zeros(geo.rho_face.shape),
            Ip_profile_face=jnp.zeros(geo.rho_face.shape),
        )
        
        # Create dummy solver numeric outputs
        solver_numeric_outputs = state.SolverNumericOutputs(
            outer_solver_iterations=jnp.array(0),
            solver_error_state=jnp.array(0),
            inner_solver_iterations=jnp.array(0),
            sawtooth_crash=jnp.array(False),
        )
        
        # Build ToraxSimState
        sim_state_obj = sim_state.ToraxSimState(
            t=jnp.array(0.0),
            dt=jnp.array(1e-4),
            core_profiles=core_profiles,
            core_sources=None,
            core_transport=state.CoreTransport.zeros(geo),
            geometry=geo,
            solver_numeric_outputs=solver_numeric_outputs,
            edge_outputs=None,
        )
        
        # Create a minimal PostProcessedOutputs with None values
        # This bypasses the 80-argument requirement
        dummy_pp = post_processing.PostProcessedOutputs(
            **{field: None for field in post_processing.PostProcessedOutputs.__dataclass_fields__}
        )
        
        # Check for errors
        error = step_function.check_for_errors(numerics, sim_state_obj, dummy_pp)
        
        # Assert that LOW_TEMPERATURE_COLLAPSE error is detected
        self.assertEqual(error, state.SimError.LOW_TEMPERATURE_COLLAPSE)


if __name__ == '__main__':
    absltest.main()