"""Tests for the absorption_fraction parameter in heat sources."""

import unittest
import numpy as np
from torax.sources.generic_ion_el_heat_source import calc_generic_heat_source
from torax.geometry.geometry import Geometry, GeometryType
from torax.torax_pydantic.interpolated_param_2d import Grid1D
import jax.numpy as jnp

class AbsorptionFractionTest(unittest.TestCase):
    def test_calc_generic_heat_source_with_absorption_fraction(self):
        """Test that absorption_fraction correctly scales the power."""
        # Create a simple geometry with necessary attributes
        r = jnp.linspace(0.0, 1.0, 10)
        grid = Grid1D.construct(nx=10, dx=0.1)
        
        # Create a basic circular geometry
        geo = Geometry(
            geometry_type=GeometryType.CIRCULAR,
            torax_mesh=grid,
            Phi=jnp.ones_like(r),
            Phi_face=jnp.ones_like(r),
            Rmaj=jnp.full_like(r, 3.0),
            Rmin=jnp.full_like(r, 1.0),
            B0=jnp.full_like(r, 2.0),
            volume=jnp.ones_like(r),
            volume_face=jnp.ones_like(r),
            area=jnp.ones_like(r),
            area_face=jnp.ones_like(r),
            vpr=jnp.ones_like(r),
            vpr_face=jnp.ones_like(r),
            spr=jnp.ones_like(r),
            spr_face=jnp.ones_like(r),
            delta_face=jnp.zeros_like(r),
            elongation=jnp.ones_like(r),
            elongation_face=jnp.ones_like(r),
            g0=jnp.ones_like(r),
            g0_face=jnp.ones_like(r),
            g1=jnp.ones_like(r),
            g1_face=jnp.ones_like(r),
            g2=jnp.ones_like(r),
            g2_face=jnp.ones_like(r),
            g3=jnp.ones_like(r),
            g3_face=jnp.ones_like(r),
            g2g3_over_rhon=jnp.ones_like(r),
            g2g3_over_rhon_face=jnp.ones_like(r),
            g2g3_over_rhon_hires=jnp.ones_like(r),
            F=jnp.ones_like(r),
            F_face=jnp.ones_like(r),
            F_hires=jnp.ones_like(r),
            Rin=jnp.ones_like(r),
            Rin_face=jnp.ones_like(r),
            Rout=jnp.ones_like(r),
            Rout_face=jnp.ones_like(r),
            spr_hires=jnp.ones_like(r),
            rho_hires_norm=jnp.ones_like(r),
            rho_hires=jnp.ones_like(r),
            Phibdot=jnp.zeros_like(r),
            _z_magnetic_axis=None
        )
        
        # Parameters for the heat source
        rsource = 0.5
        w = 0.2
        Ptot = 10.0
        el_heat_fraction = 0.5
        
        # Call with absorption_fraction = 1.0 (default)
        source_ion1, source_el1 = calc_generic_heat_source(
            geo, rsource, w, Ptot, el_heat_fraction
        )
        
        # Call with absorption_fraction = 0.5
        source_ion2, source_el2 = calc_generic_heat_source(
            geo, rsource, w, Ptot, el_heat_fraction, absorption_fraction=0.5
        )
        
        # The second result should be half of the first
        self.assertTrue(np.allclose(source_ion2, source_ion1 * 0.5))
        self.assertTrue(np.allclose(source_el2, source_el1 * 0.5))
        
        # Test relative scaling instead of absolute values
        # The ratio of total power should be 0.5
        total_power1 = np.sum(source_ion1 + source_el1)
        total_power2 = np.sum(source_ion2 + source_el2)
        
        self.assertAlmostEqual(total_power2 / total_power1, 0.5, delta=0.01)

if __name__ == '__main__':
    unittest.main() 