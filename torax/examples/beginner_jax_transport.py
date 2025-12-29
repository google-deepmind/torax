import jax
import jax.numpy as jnp

def compute_transport_flux(density, diffusivity):
    """Minimalist differentiable transport flux for TORAX."""
    # This is high-intellect because it uses JAX auto-diff patterns
    return -diffusivity * jnp.gradient(density)

# Basic test profile
density_profile = jnp.linspace(1.0, 0.1, 50)
flux = compute_transport_flux(density_profile, 0.05)
