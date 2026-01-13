
import dataclasses
import jax
import jax.numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params as runtime_params_lib
from torax.pedestal_model import pedestal_model
from torax.sim import sim

@dataclasses.dataclass(frozen=True)
class FixedProfilePedestalModel(pedestal_model.PedestalModel):
    def _call_implementation(
        self,
        runtime_params: runtime_params_lib.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> pedestal_model.PedestalModelOutput:
        # Create a mask for rho < 0.15
        mask = geo.rho_norm < 0.15

        # Create a fixed profile (e.g., constant value)
        T_i_profile = jnp.ones_like(geo.rho_norm) * 5.0  # 5 keV

        return pedestal_model.PedestalModelOutput(
            rho_norm_ped_top=0.9,
            rho_norm_ped_top_idx=jnp.argmin(jnp.abs(geo.rho_norm - 0.9)),
            T_i_ped=5.0,
            T_e_ped=5.0,
            n_e_ped=0.7e20,
            T_i_profile=T_i_profile,
            profile_mask=mask,
        )

def test_fixed_profile():
    # Setup simulation with custom pedestal model
    # This is a simplified setup, might need more boilerplate depending on how sim is initialized
    # For now, just checking if we can instantiate it and if the logic in calc_coeffs would seemingly work
    # But since we can't run jax, this test might fail to run.

    print("Verification script created. If JAX is available, this would run a sim.")

if __name__ == "__main__":
    test_fixed_profile()
