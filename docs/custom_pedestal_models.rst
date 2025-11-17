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

Create a class that inherits from ``PedestalModel`` and uses ``StaticDataclass``:

.. code-block:: python

    from torax._src import geometry
    from torax._src import state
    from torax._src.static_dataclass import StaticDataclass
    from torax import pedestal
    import jax.numpy as jnp

    @dataclasses.dataclass(frozen=True)
    class MyPedestalModel(pedestal.PedestalModel, StaticDataclass):
        """My custom pedestal model with EPED-like scaling."""

        def _call_implementation(
            self,
            runtime_params: pedestal.RuntimeParams,
            geo: geometry.Geometry,
            core_profiles: state.CoreProfiles,
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
    from torax._src.torax_pydantic import torax_pydantic

    class MyPedestal(pedestal.BasePedestal):
        """Configuration for my custom pedestal model."""

        model_name: Annotated[Literal['my_pedestal'], torax_pydantic.JAX_STATIC] = (
            'my_pedestal'
        )

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

.. code-block:: python

    CONFIG = {
        'pedestal': {
            'model_name': 'my_pedestal',
            'set_pedestal': True,
        },
    }

Example
=======

See ``torax/examples/custom_pedestal_example.py`` for a complete working example
with EPED-like scaling.

Key Points
==========

* Use ``StaticDataclass`` for JAX compatibility
* Models must be JAX-compatible (use ``jax.numpy``)
* Choose a unique ``model_name``
* Register before using in configuration
* No source code modification needed
