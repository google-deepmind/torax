.. _model-integration:

How to integrate new models
###########################

This page shows how to extend TORAX with new models.

Adding a source model
---------------------

TORAX sources are located in |torax.sources|_ and described in
:ref:`structure-sources`. While TORAX comes with several default sources
(see the |torax.sources|_ module or the API docs for the complete list),
users can both (a) configure the existing sources with custom models and
(b) add new source models to TORAX.

Instructions for how to do this are under construction.

Adding a transport model
------------------------

TORAX transport models are located in |torax.transport_model|_ and described
in :ref:`structure-transport-model`. TORAX comes with several transport models
to choose from (see the full list the API docs), but users may add new
transport models as well.

Defining a new transport model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All transport models in TORAX must extend the abstract base class
|TransportModel|_ and extend a corresponding |TransportModelBuilder|_.

.. code-block:: python

    class TransportModel(abc.ABC):
      ...

      @abc.abstractmethod
      def _call_implementation(
          self,
          dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
          geo: geometry.Geometry,
          core_profiles: state.CoreProfiles,
      ) -> state.CoreTransport:
        pass

    @dataclasses.dataclass(kw_only=True)
    class TransportModelBuilder(abc.ABC):
    """Factory for Stepper objects."""

      @abc.abstractmethod
      def __call__(self) -> TransportModel:
        """Builds a TransportModel instance."""

      runtime_params: runtime_params_lib.RuntimeParams = dataclasses.field(
          default_factory=runtime_params_lib.RuntimeParams
      )

As shown in the code snippet above, new transport models must implement the
``_call_implementation()`` which takes the complete set of runtime parameters
of TORAX, the current geometry of the torus, and the core profiles which are
being evolved, and it returns the transport coefficients. In the new builder,
the ``__call__()`` should return an instance of the new transport model. Users
may also optionally create a new runtime parameters dataclass holding custom
parameters to be fed into the transport model.

For example:

.. code-block:: python

    from torax.transport_model import runtime_params as transport_params

    @chex.dataclass
    class MyCustomRuntimeParams(transport_params.RuntimeParams):
      """Defines runtime inputs to the custom transport model."""

      foo: transport_params.TimeInterpolatedInput = 1.0
      bar: float = 2.0  # cannot change over the simulation run.

      def build_dynamic_params(self, t: chex.Numeric) -> MyCustomDynamicRuntimeParams:
        """Builds a set of these runtime params interpolated for a specific time t.

        Every runtime params object must implement `build_dynamic_params()`.
        """
        return MyCustomDynamicRuntimeParams(
            **config_args.get_init_kwargs(
                input_config=self,
                output_type=MyCustomDynamicRuntimeParams,
                t=t,
            )
        )


    @chex.dataclass(frozen=True)
    class MyCustomDynamicRuntimeParams(transport_params.DynamicRuntimeParams):
      """The dynamic slice of the complete runtime params, interpolated for a single time."""
      foo: float
      bar: float


    class MyCustomTransportModel(TransportModel):

      def _call_implementation(
          self,
          dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
          geo: geometry.Geometry,
          core_profiles: state.CoreProfiles,
      ) -> state.CoreTransport:
        assert isinstance(
            dynamic_runtime_params_slice.transport, MyCustomDynamicRuntimeParams
        )
        foo = dynamic_runtime_params_slice.transport.foo
        bar = dynamic_runtime_params_slice.transport.bar
        return state.CoreTransport(
            chi_face_ion=foo * jnp.ones_like(geo.rho_face),
            chi_face_el=foo * jnp.ones_like(geo.rho_face),
            d_face_el=bar * jnp.ones_like(geo.rho_face),
            v_face_el=bar * jnp.ones_like(geo.rho_face),
        )

    @dataclasses.dataclass(kw_only=True)
    class MyCustomTransportModelBuilder(TransportModelBuilder):

        runtime_params: MyCustomRuntimeParams = dataclasses.field(
            default_factory=MyCustomRuntimeParams
        )

        def __call__(self) -> MyCustomTransportModel:
          return MyCustomTransportModel()


Some important things to note:


* Every custom set of runtime params must come also with a "dynamic" set of
  params which contains the interpolated version of those parameters. The
  "dynamic" parameter set **must** be JAX-friendly (which is why we use
  ``chex.dataclass`` to define it). You can only use Python primitives and
  objects which are registered
  `PyTrees <https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree>`_.

* The ``_call_implementation()`` in a |TransportModel|_ **must** be jittable.


There is some boilerplate code here which we've kept for sake of being
explicit, but feedback on this design and configurability is welcome. Just
reach out to the TORAX team or open a discussion in GitHub.

Using a new transport model within TORAX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the new transport model is defined, users can plug it into a TORAX run via
the |torax.sim.Sim|_ object.

.. code-block:: python

    # in sim.py. Copied here for reference, no need to modify this.

    def build_sim_object(
        ...
        transport_model_builder: transport_model_lib.TransportModelBuilder,
        ...
    ) -> Sim:

    # in your TORAX configuration or run file .py

    my_custom_transport_builder = MyCustomTransportModelBuilder()
    # Configure it as needed.
    # Make foo time-dependent.
    my_custom_transport_builder.runtime_params.foo = {
        0.0: 1.0,  # value at t=0
        0.1: 2.0,  # value at t=0.1
        0.3: 3.0,  # value at t=0.3
    }
    # bar is constant.
    my_custom_transport_builder.runtime_params.bar = 4.0

    # Build the Sim object.
    sim_object = sim_lib.build_sim_object(
        ...,
        transport_model_builder=my_custom_transport_builder,
        ...
    )

    # Run TORAX.
    sim_object.run()


As of 7 June 2024, you cannot instantiate and configure a custom transport model
via the config dictionary. You may still configure the other components of your
TORAX simulation via the config dict and use other functions in
|torax.config.build_sim|_ to convert those to the objects you can pass into
``build_sim_object()``. We are working on making this easier, but reach out
if this is something you need.


.. |torax.sources| replace:: ``torax.sources``
.. _torax.sources: https://github.com/google-deepmind/torax/tree/main/torax/sources
.. |torax.transport_model| replace:: ``torax.transport_model``
.. _torax.transport_model: https://github.com/google-deepmind/torax/blob/main/torax/transport_model
.. |TransportModel| replace:: ``TransportModel``
.. _TransportModel: https://github.com/google-deepmind/torax/blob/main/torax/transport_model/transport_model.py
.. |TransportModelBuilder| replace:: ``TransportModelBuilder``
.. _TransportModelBuilder: https://github.com/google-deepmind/torax/blob/main/torax/transport_model/transport_model.py
.. |torax.sim.Sim| replace:: ``torax.sim.Sim``
.. _torax.sim.Sim: https://github.com/google-deepmind/torax/blob/main/torax/sim.py
.. |torax.config.build_sim| replace:: ``torax.config.build_sim``
.. _torax.config.build_sim: https://github.com/google-deepmind/torax/blob/main/torax/config/build_sim.py
