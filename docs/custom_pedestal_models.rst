.. _custom-pedestal-models:

Custom Pedestal Models
######################

TORAX allows you to define custom pedestal scaling laws without modifying the source code.
This is useful for machine-specific models like those used for STEP.

Quick Start
===========

Follow these four steps to create and use a custom pedestal model:

1. Define Your JAX Pedestal Model
----------------------------------

Create a class that inherits from ``PedestalModel``:

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    from torax import Geometry
    from torax import CoreProfiles
    from torax import pedestal

    @dataclasses.dataclass(frozen=True)
    class MyPedestalModel(pedestal.PedestalModel):
        """My custom pedestal model with EPED-like scaling."""

        def _call_implementation(
            self,
            runtime_params: pedestal.RuntimeParams,
            geo: Geometry,
            core_profiles: CoreProfiles,
        ) -> pedestal.PedestalModelOutput:
            # Extract plasma parameters
            Ip_MA = runtime_params.profile_conditions.Ip / 1e6
            B0 = geo.B0

            # Your custom scaling laws
            T_e_ped = 0.5 * (Ip_MA ** 0.2) * (B0 ** 0.8)
            T_i_ped = 1.2 * T_e_ped
            n_e_ped = 0.7e20
            rho_norm_ped_top = 0.91

            # Find mesh index
            rho_norm_ped_top_idx = jnp.argmin(
                jnp.abs(geo.rho_norm - rho_norm_ped_top)
            )

            return pedestal.PedestalModelOutput(
                rho_norm_ped_top=rho_norm_ped_top,
                rho_norm_ped_top_idx=rho_norm_ped_top_idx,
                T_i_ped=T_i_ped,
                T_e_ped=T_e_ped,
                n_e_ped=n_e_ped,
            )

2. Define Your Pydantic Configuration
--------------------------------------

.. code-block:: python

    from typing import Annotated, Literal
    from torax import JAX_STATIC

    class MyPedestal(pedestal.BasePedestal):
        """Configuration for my custom pedestal model."""

        model_name: Annotated[
            Literal['my_pedestal'],
            JAX_STATIC
        ] = 'my_pedestal'

        def build_pedestal_model(self) -> MyPedestalModel:
            return MyPedestalModel()

        def build_runtime_params(self, t) -> pedestal.RuntimeParams:
            return pedestal.RuntimeParams(
                set_pedestal=self.set_pedestal.get_value(t),
            )

3. Register Your Model
----------------------

.. code-block:: python

    pedestal.register_pedestal_model(MyPedestal)

4. Use in Configuration
-----------------------

Now use it in your simulation config. Note: this is a minimal example showing
only the pedestal configuration. See the full example for a complete runnable config.

.. code-block:: python

    CONFIG = {
        # ... other config sections ...
        'pedestal': {
            'model_name': 'my_pedestal',
            'set_pedestal': True,
        },
    }

Example
=======

See ``torax/examples/custom_pedestal_example.py`` for a complete working example
with EPED-like scaling that can be run directly.

Key Points
==========

* ``PedestalModel`` already inherits from ``StaticDataclass`` - don't inherit twice
* Use public API (``from torax import ...``) not ``_src``
* Models must be JAX-compatible (use ``jax.numpy``)
* Choose a unique ``model_name``
* Register before using in configuration
