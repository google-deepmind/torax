.. _model-integration:

How to integrate new models
###########################

TORAX has a modular design which supports easy coupling of new physics models
such as sources, transport models, pedestal models, and neoclassical models.

TORAX provides a public API for registering custom transport, pedestal,
neoclassical, source, and edge models.
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

Create a frozen dataclass that inherits from
``torax.transport.ComponentTransportModel`` and implements the
``call_implementation`` method. This method receives the current simulation
state and must return a ``torax.transport.TransportCoeffs`` object
containing the computed transport coefficients on the face grid.

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import torax
    from torax import transport

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyTransportModel(transport.ComponentTransportModel):
      """Custom transport model."""

      def call_implementation(
          self,
          transport_runtime_params: transport.ComponentRuntimeParams,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
          two_point_mask: jax.Array,
      ) -> transport.TransportCoeffs:
        # Implement your transport model here.
        # Must return a TransportCoeffs with at least the four required
        # fields: chi_face_ion, chi_face_el, d_face_el, v_face_el.
        chi_i = jnp.ones_like(geo.rho_face_norm) * 2.0
        chi_e = jnp.ones_like(geo.rho_face_norm) * 1.5
        d_e = jnp.ones_like(geo.rho_face_norm) * 0.5
        v_e = jnp.zeros_like(geo.rho_face_norm)

        return transport.TransportCoeffs(
            chi_face_ion=chi_i,
            chi_face_el=chi_e,
            d_face_el=d_e,
            v_face_el=v_e,
        )


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.transport.ComponentTransportConfigBase`` and implements the
``build_transport_model`` method. The config class must have a ``model_name``
field with a unique ``Literal`` type that identifies your model.

.. code-block:: python

    from typing import Annotated, Literal

    class MyTransportConfig(transport.ComponentTransportConfigBase):
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

Once registered, the model can be used in a TORAX config under
``core_transport_models`` or ``pedestal_transport_models`` within the
``transport`` configuration:

.. code-block:: python

    config = {
        ...
        'transport': {
            'core_transport_models': {
                'my_custom_model': {
                    'model_name': 'my_transport',
                },
            },
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


Registering a custom pedestal structure model
=============================================

The TORAX ``PedestalModel`` is composed of three sub-models: a
``PedestalStructureModel`` (which computes the H-mode target pedestal-top width,
temperatures, and density), a ``FormationModel``, and a ``SaturationModel``.
To integrate a custom pedestal structure model (for example, an external neural
network surrogate such as
`epednn-mit <https://github.com/aaronkho/epednn_mit>`_), you need to:

1. Define a pedestal structure model class that implements the physics.
2. Define a pydantic config class for your model.
3. Register the config class with TORAX.

Step 1: Implement the pedestal structure model
----------------------------------------------

Create a frozen dataclass that inherits from
``torax.pedestal.PedestalStructureModel`` and implements the ``__call__``
method. This method must return a ``torax.pedestal.PedestalStructure`` with the
H-mode target pedestal-top properties.

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import torax
    from torax import pedestal

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyPedestalStructureModel(pedestal.PedestalStructureModel):
      """Custom pedestal structure model."""

      def __call__(
          self,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
          pedestal_transition_state: pedestal.PedestalTransitionState,
      ) -> pedestal.PedestalStructure:
        # Implement your pedestal structure model here.
        return pedestal.PedestalStructure(
            rho_norm_ped_top=jnp.array(0.9),
            T_i_ped=jnp.array(5.0),
            T_e_ped=jnp.array(5.0),
            n_e_ped=jnp.array(0.7e20),
        )


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.pedestal.BaseStructure`` and implements the ``build_structure_model``
method. The config class must have a ``model_name`` field with a unique
``Literal`` type.

You may also override ``build_runtime_params`` if your model requires
additional runtime parameters. Your runtime parameters class should inherit
from ``torax.pedestal.StructureRuntimeParams`` and will be available at
``runtime_params.pedestal.structure``.

Continuing from Step 1 (additional imports only):

.. code-block:: python

    from typing import Annotated, Literal

    import jax
    from torax._src import array_typing
    from torax._src.torax_pydantic import torax_pydantic

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class MyStructureRuntimeParams(pedestal.StructureRuntimeParams):
      """Runtime parameters for MyPedestalStructureModel."""

      T_ped: array_typing.FloatScalar

    class MyStructureConfig(pedestal.BaseStructure):
      """Pydantic config for MyPedestalStructureModel."""

      model_name: Annotated[
          Literal['my_pedestal'], torax.JAX_STATIC
      ] = 'my_pedestal'
      T_ped: torax_pydantic.TimeVaryingScalar = (
          torax_pydantic.ValidatedDefault(5.0)
      )

      def build_structure_model(self) -> MyPedestalStructureModel:
        return MyPedestalStructureModel()

      def build_runtime_params(self, t) -> MyStructureRuntimeParams:
        return MyStructureRuntimeParams(
            **vars(super().build_runtime_params(t)),
            T_ped=self.T_ped.get_value(t),
        )


Step 3: Register the model
---------------------------

Call ``torax.pedestal.register_pedestal_structure_model`` with your pydantic
config class. This must be done at module level, before any TORAX config is
built.

.. code-block:: python

    pedestal.register_pedestal_structure_model(MyStructureConfig)


Using the registered model
---------------------------

Once registered, the model can be used inside ``structure_model`` in a TORAX
config:

.. code-block:: python

    config = {
        ...
        'pedestal': {
            'structure_model': {
                'model_name': 'my_pedestal',
            },
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


Registering a custom source model
==================================

To integrate a custom source model (e.g. a new heat source, particle source,
or current source), you need to:

1. Define a custom ``Source`` subclass implementing ``_get_model_value``.
2. Define a pydantic config class for your model that builds your source in
   ``build_source()``.
3. Register the config class with TORAX.

Unlike transport and pedestal models, source models are registered against a
specific *source name* (e.g. ``'gas_puff'``, ``'fusion'``, ``'generic_heat'``,
etc.). This allows multiple source model implementations to exist for the same
physical source. The special source ``'ei_exchange'`` (ion-electron heat
exchange) does not support custom registration.

Step 1: Implement the custom Source subclass
--------------------------------------------

Subclass ``torax.sources.Source`` (or an existing source class) and implement
``_get_model_value``.

When subclassing ``torax.sources.Source`` directly, you must define two class
attributes:

* ``SOURCE_ID``: The target source name (e.g. ``'generic_heat'``), matching the
  corresponding field in the ``Sources`` config.
* ``AFFECTED_CORE_PROFILES``: A tuple of ``torax.sources.AffectedCoreProfile``
  values (``PSI``, ``NE``, ``TEMP_ION``, ``TEMP_EL``, or ``FAST_IONS``)
  specifying which core profile equations the source contributes to. (If
  ``FAST_IONS`` is included, you must also implement ``zero_fast_ions``.)

If you instead subclass an existing source class (e.g.
``GenericIonElectronHeatSource``), ``SOURCE_ID`` and ``AFFECTED_CORE_PROFILES``
are inherited automatically.

``_get_model_value`` receives the simulation state and must return a tuple of
source profile arrays on the cell grid (one per entry in
``AFFECTED_CORE_PROFILES``, in the same order).

.. code-block:: python

    import dataclasses
    from typing import ClassVar
    import jax.numpy as jnp
    import torax
    from torax import sources

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class MyHeatSource(sources.Source):
      """Custom heat source model."""

      SOURCE_ID: ClassVar[str] = 'generic_heat'
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[sources.AffectedCoreProfile, ...]
      ] = (
          sources.AffectedCoreProfile.TEMP_ION,
          sources.AffectedCoreProfile.TEMP_EL,
      )

      def _get_model_value(
          self,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
          calculated_source_profiles: sources.SourceProfiles | None,
          conductivity,
      ) -> tuple[jnp.ndarray, ...]:
        # Return a tuple with one element per affected core profile.
        # For a source affecting TEMP_ION and TEMP_EL, return two profiles.
        ion_heat = jnp.ones_like(geo.rho_norm) * 1e6
        el_heat = jnp.ones_like(geo.rho_norm) * 0.5e6
        return (ion_heat, el_heat)


Step 2: Define the pydantic config
------------------------------------

Create a pydantic config class that inherits from
``torax.sources.SourceConfigBase``:

- Implement ``build_source``: returns the ``Source`` instance.
- Override ``build_runtime_params`` if your source requires custom
  ``RuntimeParams`` fields beyond the base ``RuntimeParams``.

The config class must have a ``model_name`` field with a unique ``Literal``
type that identifies your model. This name must be different from any already
registered ``model_name`` for the source slot.

.. code-block:: python

    import dataclasses
    from typing import Literal

    import chex
    import jax
    from torax import sources

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class MyRuntimeParams(sources.RuntimeParams):
      """Custom runtime params with an extra parameter."""
      scaling_factor: float

    class MyHeatSourceConfig(sources.SourceConfigBase):
      """Pydantic config for my custom heat source."""

      model_name: Literal['my_heat_model'] = 'my_heat_model'
      scaling_factor: float = 1.0

      def build_source(self) -> sources.Source:
        return MyHeatSource()

      def build_runtime_params(
          self, t: chex.Numeric,
      ) -> MyRuntimeParams:
        base_params = super().build_runtime_params(t)
        return MyRuntimeParams(
            **vars(base_params),
            scaling_factor=self.scaling_factor,
        )



Step 3: Register the model
---------------------------

Call ``torax.sources.register_source_config`` with your pydantic config class.
The target source is inferred automatically from ``build_source().SOURCE_ID``.
This must be done at module level, before any TORAX config is built.

.. code-block:: python

    sources.register_source_config(MyHeatSourceConfig)

The source's ``SOURCE_ID`` must be one of the fields in the ``SourcesConfig``
pydantic config: ``bremsstrahlung``, ``cyclotron_radiation``, ``ecrh``,
``fusion``, ``gas_puff``, ``generic_current``, ``generic_heat``,
``generic_particle``, ``icrh``, ``impurity_radiation``, ``ohmic``, or
``pellet``.

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


Registering a custom edge model
===============================

Integrating a custom edge model matches the way a custom transport model is
implemented (via ``torax.register_transport_model``). To integrate a custom
edge model, you need to:

1. Define an edge model class that computes boundary conditions for the core.
2. Define a pydantic config class for your model.
3. Register the config class with TORAX.

Step 1: Implement the edge model
--------------------------------

Create a frozen dataclass that inherits from ``torax.edge.EdgeModel`` and
implements the ``__call__`` method. The ``__call__``
method receives the current edge runtime parameters, full simulation runtime
parameters, geometry, core profiles, and source profiles, and must
return an ``torax.edge.EdgeModelOutputs`` with the boundary conditions computed
by the model (any boundary conditions not computed by the model should be left
as ``None``):

* ``T_e_right_bc``: Electron temperature boundary condition at the LCFS [keV].
* ``T_i_right_bc``: Ion temperature boundary condition at the LCFS [keV].
* ``n_e_right_bc``: Electron density boundary condition at the LCFS [m^-3].
* ``impurity_right_bc``: Mapping from impurity symbol to its right boundary
  condition ratio.

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import torax
    from torax import edge

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyEdgeModel(edge.EdgeModel):
      """Custom edge model computing temperature boundary conditions."""

      def __call__(
          self,
          edge_runtime_params: edge.RuntimeParams,
          runtime_params: torax.RuntimeParams,
          geo: torax.Geometry,
          core_profiles: torax.CoreProfiles,
          core_sources: torax.SourceProfiles,
          previous_edge_outputs: edge.EdgeModelOutputs | None = None,
      ) -> edge.EdgeModelOutputs:
        del edge_runtime_params, runtime_params, geo, core_profiles, core_sources
        del previous_edge_outputs
        return edge.EdgeModelOutputs(
            T_i_right_bc=jnp.array(0.08),  # [keV]
            T_e_right_bc=jnp.array(0.12),  # [keV]
        )


Step 2: Define the pydantic config
----------------------------------

Create a pydantic config class that inherits from
``torax.edge.EdgeModelConfig`` and implements ``build_edge_model``. Declare
``supported_boundary_conditions`` as a ``ClassVar`` of type
``torax.edge.SupportedBoundaryConditions`` to specify which boundary conditions
your model is capable of calculating (by default on
``SupportedBoundaryConditions``, all four are ``False``), and a ``model_name``
field with a unique ``Literal`` type identifying the model. Each ``update_*``
flag defaults to the corresponding value in ``supported_boundary_conditions``,
and ``EdgeModelConfig`` validates that no ``update_*`` flag is set to ``True``
for an unsupported boundary condition.

.. code-block:: python

    from typing import Annotated, ClassVar, Literal
    import torax
    from torax import edge

    class MyEdgeConfig(edge.EdgeModelConfig):
      """Pydantic config for MyEdgeModel."""

      supported_boundary_conditions: ClassVar[
          edge.SupportedBoundaryConditions
      ] = edge.SupportedBoundaryConditions(
          T_e=True,
          T_i=True,
      )
      model_name: Annotated[
          Literal['my_edge'], torax.JAX_STATIC
      ] = 'my_edge'

      def build_edge_model(self) -> MyEdgeModel:
        return MyEdgeModel()


Step 3: Register the model
--------------------------

Call ``torax.edge.register_edge_model`` with your pydantic config class.
This must be done at module level before any TORAX config is built.

.. code-block:: python

    edge.register_edge_model(MyEdgeConfig)


Using the registered model
--------------------------

Once registered, the model can be used either as a standalone edge model or as a
sub-model inside a ``'combined'`` edge model:

.. code-block:: python

    config = {
        ...
        'edge': {
            'model_name': 'my_edge',
        },
        ...
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


Combining multiple edge sub-models
----------------------------------

Any registered ``EdgeModelConfig`` can also be used inside a ``'combined'``
edge model under ``sub_models``. Each boundary condition is taken from the
sub-model whose corresponding ``update_*`` flag is enabled, and at most one
sub-model may have a given ``update_*`` flag enabled across the simulation.
Since the ``update_*`` flags default to the sub-model's
``supported_boundary_conditions``, set the flags to ``False`` on a sub-model for
any boundary conditions it should not provide:

.. code-block:: python

    config = {
        ...
        'edge': {
            'model_name': 'combined',
            'sub_models': {
                'temperatures': {
                    'model_name': 'my_edge',
                    'update_n_e': False,
                    'update_n_impurities': False,
                },
                'density': {
                    'model_name': 'my_density_edge',
                },
            },
        },
        ...
    }


Registering a custom neoclassical model
=======================================

Use ``torax.neoclassical.register_neoclassical_model`` to couple a custom
neoclassical model, for example a single solver that computes all neoclassical
quantities (bootstrap current, parallel conductivity, neoclassical transport,
and poloidal velocity) together in one evaluation.

Create a ``NeoclassicalModel`` subclass implementing ``__call__`` (returning
``NeoclassicalOutputs``), an optional ``RuntimeParams`` subclass for dynamic
parameters, and a ``NeoclassicalConfigBase`` config subclass with a unique
``model_name`` ``Literal``:

.. code-block:: python

    import dataclasses
    from typing import Annotated, Literal
    import jax
    import jax.numpy as jnp
    import torax
    from torax import neoclassical

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class MyNeoclassicalRuntimeParams(neoclassical.RuntimeParams):
      """Custom runtime params for an integrated neoclassical model."""

      sigma_scale: float

    @dataclasses.dataclass(frozen=True, eq=False)
    class MyNeoclassicalModel(neoclassical.NeoclassicalModel):
      """Custom integrated neoclassical model."""

      def __call__(
          self,
          runtime_params: torax.RuntimeParams,
          geometry: torax.Geometry,
          core_profiles: torax.CoreProfiles,
      ) -> neoclassical.NeoclassicalOutputs:
        assert isinstance(
            runtime_params.neoclassical, MyNeoclassicalRuntimeParams
        )
        scale = runtime_params.neoclassical.sigma_scale
        return neoclassical.NeoclassicalOutputs(
            bootstrap_current=neoclassical.BootstrapCurrent.zeros(geometry),
            conductivity=neoclassical.Conductivity(
                sigma=jnp.ones_like(geometry.rho_norm) * scale,
                sigma_face=jnp.ones_like(geometry.rho_face_norm) * scale,
            ),
            transport=neoclassical.NeoclassicalTransport.zeros(geometry),
            poloidal_velocity=neoclassical.PoloidalVelocity.zeros(geometry),
        )

    class MyNeoclassicalConfig(neoclassical.NeoclassicalConfigBase):
      """Pydantic config for MyNeoclassicalModel."""

      model_name: Annotated[
          Literal['my_neoclassical'], torax.JAX_STATIC
      ] = 'my_neoclassical'
      sigma_scale: float = 1.0e6

      def build_runtime_params(self) -> MyNeoclassicalRuntimeParams:
        return MyNeoclassicalRuntimeParams(sigma_scale=self.sigma_scale)

      def build_model(self) -> MyNeoclassicalModel:
        return MyNeoclassicalModel()

    neoclassical.register_neoclassical_model(MyNeoclassicalConfig)

Once registered, select your top-level model in the ``neoclassical`` config
dictionary:

.. code-block:: python

    config = {
        ...
        'neoclassical': {
            'model_name': 'my_neoclassical',
            'sigma_scale': 2.0e6,
        },
        ...
    }


Configuring time-varying parameters and physical bounds
=======================================================

When defining custom Pydantic config classes for models, TORAX provides several
type helpers in ``torax._src.torax_pydantic`` to enforce physical constraints
(such as positivity, non-negativity, intervals, or custom bounds) on
time-varying scalars and arrays:

* **Positivity (** :math:`> 0` **)**: Use ``PositiveTimeVaryingScalar`` or
  ``PositiveTimeVaryingArray`` (e.g. for diffusion coefficients, temperatures,
  or Gaussian widths).
* **Non-Negativity (** :math:`\ge 0` **)**: Use
  ``NonNegativeTimeVaryingScalar`` or ``NonNegativeTimeVaryingArray`` (e.g. for
  transport ratios).
* **Unit Interval (** :math:`[0, 1]` **)**: Use
  ``UnitIntervalTimeVaryingScalar`` (e.g. for normalized radial locations or
  heating fractions).
* **Custom Bounds**: Use
  ``scalar_bounds_validator(gt=..., ge=..., lt=..., le=...)`` or
  ``array_bounds_validator(...)`` with ``Annotated`` (e.g.
  ``Annotated[TimeVaryingScalar, scalar_bounds_validator(gt=1.0)]``).

Example usage in a custom model config:

.. code-block:: python

    from typing import Annotated, Literal
    from torax import sources
    from torax._src.torax_pydantic import torax_pydantic

    class MyCustomConfig(sources.SourceConfigBase):
      """Custom source config with bounded parameters."""

      model_name: Annotated[
          Literal['my_custom'], torax.JAX_STATIC
      ] = 'my_custom'
      width: torax_pydantic.PositiveTimeVaryingScalar = (
          torax_pydantic.ValidatedDefault(0.1)
      )
      location: torax_pydantic.UnitIntervalTimeVaryingScalar = (
          torax_pydantic.ValidatedDefault(0.5)
      )
      scale: Annotated[
          torax_pydantic.TimeVaryingScalar,
          torax_pydantic.scalar_bounds_validator(gt=1.0),
      ] = torax_pydantic.ValidatedDefault(1.5)


.. toctree::
   :maxdepth: 1
   :caption: Model Integration Topics

   interfacing_with_surrogates
