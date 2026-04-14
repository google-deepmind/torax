.. _model-integration:

How to integrate new models
###########################

TORAX has a modular design which supports easy coupling of new physics models
such as sources, transport models, pedestal models, etc.

TORAX provides a public API for registering custom transport, pedestal, and
source models.
Once registered, custom models can be configured via TORAX config files or
dictionaries just like the built-in models.

.. contents:: On this page
   :local:
   :depth: 2


Registering a custom transport model
=====================================

To integrate a custom transport model, you need to:

1. Define a transport model class that implements the physics.
2. Define a pydantic config class for your model.
3. Register the config class with TORAX.

Step 1: Implement the transport model
--------------------------------------

Create a frozen dataclass that inherits from ``torax.transport.TransportModel``
and implements the ``call_implementation`` method. This method receives the
current simulation state and must return a ``torax.transport.TurbulentTransport``
object containing the computed transport coefficients on the face grid.

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import torax
    from torax import transport

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyTransportModel(transport.TransportModel):
      """Custom transport model."""

      def call_implementation(
          self,
          transport_runtime_params: transport.RuntimeParams,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
          pedestal_model_outputs: torax.PedestalModelOutput,
      ) -> transport.TurbulentTransport:
        # Implement your transport model here.
        # Must return a TurbulentTransport with at least the four required
        # fields: chi_face_ion, chi_face_el, d_face_el, v_face_el.
        chi_i = jnp.ones_like(geo.rho_face_norm) * 2.0
        chi_e = jnp.ones_like(geo.rho_face_norm) * 1.5
        d_e = jnp.ones_like(geo.rho_face_norm) * 0.5
        v_e = jnp.zeros_like(geo.rho_face_norm)

        return transport.TurbulentTransport(
            chi_face_ion=chi_i,
            chi_face_el=chi_e,
            d_face_el=d_e,
            v_face_el=v_e,
        )


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.transport.TransportBase`` and implements the ``build_transport_model``
method. The config class must have a ``model_name`` field with a unique
``Literal`` type that identifies your model.

.. code-block:: python

    from typing import Annotated, Literal

    class MyTransportConfig(transport.TransportBase):
      """Pydantic config for MyTransportModel."""

      model_name: Annotated[
          Literal['my_transport'], torax.JAX_STATIC
      ] = 'my_transport'

      def build_transport_model(self) -> MyTransportModel:
        return MyTransportModel()


Step 3: Register the model
---------------------------

Call ``torax.transport.register_transport_model`` with your pydantic config
class. This must be done at module level, before any TORAX config is built.

.. code-block:: python

    transport.register_transport_model(MyTransportConfig)


Using the registered model
---------------------------

Once registered, the model can be used in a TORAX config by setting the
``transport.model_name`` field to the model name you defined:

.. code-block:: python

    config = {
        ...
        'transport': {
            'model_name': 'my_transport',
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


Registering a custom pedestal model
=====================================

To integrate a custom pedestal model, you need to:

1. Define a pedestal model class that implements the physics.
2. Define a pydantic config class for your model.
3. Register the config class with TORAX.

Step 1: Implement the pedestal model
--------------------------------------

Create a frozen dataclass that inherits from ``torax.pedestal.PedestalModel``
and implements the ``_call_implementation`` method. This method must return a
``torax.pedestal.PedestalModelOutput`` with the pedestal properties.

The ``PedestalModel`` base class requires ``formation_model`` and
``saturation_model`` fields (used in ``ADAPTIVE_TRANSPORT`` mode). These should
be passed through from the pydantic config's ``build_pedestal_model`` method.

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import torax
    from torax import pedestal

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyPedestalModel(pedestal.PedestalModel):
      """Custom pedestal model."""

      def _call_implementation(
          self,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
      ) -> pedestal.PedestalModelOutput:
        # Implement your pedestal model here.
        return pedestal.PedestalModelOutput(
            rho_norm_ped_top=jnp.array(0.9),
            rho_norm_ped_top_idx=jnp.abs(geo.rho_norm - 0.9).argmin(),
            T_i_ped=jnp.array(5.0),
            T_e_ped=jnp.array(5.0),
            n_e_ped=jnp.array(0.7e20),
        )


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.pedestal.BasePedestal`` and implements the ``build_pedestal_model``
method. The config class must have a ``model_name`` field with a unique
``Literal`` type.

You may also override ``build_runtime_params`` if your model requires
additional runtime parameters beyond the base pedestal parameters.

.. code-block:: python

    from typing import Annotated, Literal

    class MyPedestalConfig(pedestal.BasePedestal):
      """Pydantic config for MyPedestalModel."""

      model_name: Annotated[
          Literal['my_pedestal'], torax.JAX_STATIC
      ] = 'my_pedestal'

      def build_pedestal_model(self) -> MyPedestalModel:
        return MyPedestalModel(
            formation_model=self.formation_model.build_formation_model(),
            saturation_model=self.saturation_model.build_saturation_model(),
        )

      def build_runtime_params(
          self, t,
      ) -> pedestal.RuntimeParams:
        return pedestal.RuntimeParams(
            set_pedestal=self.set_pedestal.get_value(t),
            mode=self.mode,
            formation=self.formation_model.build_runtime_params(t),
            saturation=self.saturation_model.build_runtime_params(t),
            chi_max=self.chi_max.get_value(t),
            D_e_max=self.D_e_max.get_value(t),
            V_e_max=self.V_e_max.get_value(t),
            V_e_min=self.V_e_min.get_value(t),
            pedestal_top_smoothing_width=(
                self.pedestal_top_smoothing_width.get_value(t)
            ),
        )


Step 3: Register the model
---------------------------

Call ``torax.pedestal.register_pedestal_model`` with your pydantic config
class. This must be done at module level, before any TORAX config is built.

.. code-block:: python

    pedestal.register_pedestal_model(MyPedestalConfig)


Using the registered model
---------------------------

Once registered, the model can be used in a TORAX config:

.. code-block:: python

    config = {
        ...
        'pedestal': {
            'model_name': 'my_pedestal',
            'set_pedestal': True,
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


Registering a custom source model
==================================

To integrate a custom source model (e.g. a new heat source, particle source,
or current source), you need to:

1. Define a model function that computes the source profile.
2. Define a pydantic config class for your model.
3. Register the config class with TORAX against a specific source name.

Unlike transport and pedestal models, source models are registered against a
specific *source name* (e.g. ``'gas_puff'``, ``'fusion'``, ``'generic_heat'``,
etc.). This allows multiple source model implementations to exist for the same
physical source. The two special sources ``'qei'`` (ion-electron heat exchange)
and ``'j_bootstrap'`` (bootstrap current) do not support custom registration.

Step 1: Implement the model function
--------------------------------------

Define a function that matches the ``torax.sources.SourceProfileFunction``
protocol. This function receives the simulation state and must return a tuple
of source profile arrays (one per affected core profile). The order of the
profiles in the tuple must match the order of the affected core profiles for the
source being registered against (e.g. for ``generic_heat``, the tuple must be
(ion heat, electron heat)).

.. code-block:: python

    import jax.numpy as jnp
    import torax
    from torax import sources

    def my_heat_source(
        runtime_params: torax.RuntimeParams,
        geo: torax.Geometry,
        source_name: str,
        core_profiles: torax.CoreProfiles,
        calculated_source_profiles: sources.SourceProfiles | None,
        unused_conductivity,
    ) -> tuple[jnp.ndarray, ...]:
      """Custom heat source model."""
      # Return a tuple with one element per affected core profile.
      # For a source affecting TEMP_ION and TEMP_EL, return two profiles.
      ion_heat = jnp.ones_like(geo.rho_norm) * 1e6
      el_heat = jnp.ones_like(geo.rho_norm) * 0.5e6
      return (ion_heat, el_heat)


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.sources.SourceModelBase`` and implements three required methods:

- ``model_func`` (property): returns the model function.
- ``build_source``: returns the ``Source`` instance.
- ``build_runtime_params``: returns source-specific ``RuntimeParams``.

The config class must have a ``model_name`` field with a unique ``Literal``
type that identifies your model. This name must be different from the default
model name for the source you are registering against.

.. code-block:: python

    import dataclasses
    from typing import Literal

    import chex
    import jax
    from torax import sources
    from torax._src.sources import generic_ion_el_heat_source as heat_source_lib

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class MyRuntimeParams(sources.RuntimeParams):
      """Custom runtime params with an extra parameter."""
      scaling_factor: float

    class MyHeatSourceConfig(sources.SourceModelBase):
      """Pydantic config for my custom heat source."""

      model_name: Literal['my_heat_model'] = 'my_heat_model'
      scaling_factor: float = 1.0

      @property
      def model_func(self) -> sources.SourceProfileFunction:
        return my_heat_source

      def build_source(self) -> sources.Source:
        return heat_source_lib.GenericIonElectronHeatSource(
            model_func=self.model_func
        )

      def build_runtime_params(
          self, t: chex.Numeric,
      ) -> MyRuntimeParams:
        return MyRuntimeParams(
            scaling_factor=self.scaling_factor,
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]
            ),
            mode=self.mode,
            is_explicit=self.is_explicit,
        )


Step 3: Register the model
---------------------------

Call ``torax.sources.register_source_model_config`` with your pydantic config
class and the name of the source to register against. This must be done at
module level, before any TORAX config is built.

.. code-block:: python

    sources.register_source_model_config(MyHeatSourceConfig, 'generic_heat')

The ``source_name`` must be one of the fields in the ``Sources`` pydantic model:
``bremsstrahlung``, ``cyclotron_radiation``, ``ecrh``, ``fusion``, ``gas_puff``,
``generic_current``, ``generic_heat``, ``generic_particle``, ``icrh``,
``impurity_radiation``, ``ohmic``, or ``pellet``.

If you want to register a custom implementation for a source that isn't in this
list (for example "nbi"), please reach out to the TORAX team and we will help.


Using the registered model
---------------------------

Once registered, the model can be used in a TORAX config by setting the
``model_name`` field within the corresponding source:

.. code-block:: python

    config = {
        ...
        'sources': {
            'generic_heat': {
                'model_name': 'my_heat_model',
                'scaling_factor': 2.0,
            },
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


.. toctree::
   :maxdepth: 1
   :caption: Model Integration Topics

   interfacing_with_surrogates
