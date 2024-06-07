.. _structure:

TORAX code structure
####################

This page describes the structure of the TORAX repository and gives pointers for
where to find different components in the code base.

This page is just an overview. For more complete details of what's available to
call and use, see the API docs. And for more information on the logical flow of
TORAX, read through the code and reach out to our team if anything is unclear!
We strive to make the code readable and understandable with the help of this
guide, so feedback is appreciated :)

Runnable entrypoint
-------------------

|run_simulation_main.py|_ is the main runnable entrypoint
for trying out TORAX. If you are interested in getting started quickly, check
out :ref:`quickstart`.

TORAX library
-------------

The TORAX library contains modules to build and run a transport simulation. This
section describes several of the modules and how they fit together.

For reference, the standard control flow while using TORAX is:


#.
   Configure a simulation with a dictionary of Python primitives (see
   |torax/examples/|_). More on this config language below.

#.
   Use |torax.config.build_sim.build_sim_from_config()|_ to convert the config
   to a |torax.sim.Sim|_ object, a helper object containing the building blocks
   of a TORAX simulation run.

#.
   |torax.sim.Sim|_ internally calls |torax.sim.run_simulation()|_, which is the
   entrypoint for all TORAX simulations. Technically, the above steps are
   optional (though useful standards).

#.
   |torax.sim.run_simulation()|_ steps the simulation in a loop, the duration of
   each time step coming from a |TimeStepCalculator|_, and the state updates
   coming from a |Stepper|_, which internally uses |SourceModels|_ and a
   |TransportModel|_ to help evolve the state.


The rest of this section details each of these components and a few more.

sim.py
^^^^^^

|sim.py|_ holds the majority of the business logic running TORAX.
``run_simulation()`` is its main function which actually runs the TORAX
simulation. All TORAX runs hit this function.

This file also contains the |torax.sim.Sim|_ class which wraps
``run_simulation()`` for convenience. It provides a higher level API so users
can run a TORAX simulation without constructing all the inputs
``run_simulation()`` asks for.

Broadly, ``run_simulation()`` takes the following inputs:


* an initial state
* geometry
* runtime parameters
* |TimeStepCalculator|_
* |Stepper|_ (via the ``step_fn`` argument)
* |TransportModel|_ (via the ``step_fn`` argument)

The following sections cover these inputs.

Note that, while the standard flow detailed above uses the "config", and then
builds a |torax.sim.Sim|_ object, these steps are optional. Users who wish to
customize TORAX with purpose-built transport models or steppers may call
``run_simulation()`` directly.

TORAX state
^^^^^^^^^^^

The |torax.state|_ module describes the input and output states of TORAX.
|torax.state.ToraxSimState|_ is the complete simulation state, with several
attributes inspired by the IMAS schema. Each time step of a TORAX simulation
updates that state.

Geometry
^^^^^^^^

|torax.geometry.Geometry|_ describes the geometry of the torus and fields.
See the ``geometry`` module for different ways to build a ``Geometry``.

Each ``Geometry`` object represents the geometry at a single point in time.
It is on our roadmap to support time-varying geometry as well.

Config and Runtime Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within TORAX, we refer to a "config" as a Python file or dictionary configuring
the building blocks of a TORAX simulation run. A "config" may define which
time-step calculator to use, which transport model to load, and where the
geometry comes from.

We refer to "runtime parameters" as the inputs to the various objects running
within a TORAX simulation. These runtime parameters may be dynamic, meaning
potentially time-varying and will not trigger a JAX recompilation if they change.
Or they are static, meaning that they will trigger a recompilation if they
change from run to run.

The ``torax.config`` module contains files for both of these.

|torax.config.build_sim|_ builds a |torax.sim.Sim|_ object from a config
dictionary, like the ones see in the |torax/examples/|_ folder.

|torax.config.runtime_params|_ shows the runtime inputs to the TORAX simulation,
and |torax.interpolated_param|_ shows the logic of how we take user input
configs and interpolate them to a specific time-step, allowing for time-varying
runtime params.

Many of the modules below also have module-specific runtime parameters that are
also fed into the simulation. All these runtime params are packaged together and
interpolated to a specific time, referred to as a params "slice",
|torax.config.runtime_params_slice|_.

Stepper
^^^^^^^

|torax.stepper|_ contains PDE time steppers that discretize the PDE in time and
solve for the next time step with linear or nonlinear methods.

Inside the |Stepper|_ implementations in where JAX is actually used to compute
Jacobians or do optimization-based solving. See the implementations for more
details.

Most |Stepper|_\ s are built with |SourceModels|_ and a |TransportModel|_,
both described below.

The |Stepper|_ class is abstract and can be extended. Users may provide their
own implementation and feed it to |torax.sim.run_simulation()|_.

Sources
^^^^^^^

The |torax.sources|_ module contains all source models plugged into TORAX. They
are packaged together into a |SourceModels|_ object, which is a simple container
to help access all the sources while stepping through the simulation.

A TORAX ``Source`` produces heat, particle, or current deposition profiles used
to compute PDE source/sink coefficients used while solving for the next
simulation state. TORAX provides several default source model implementations,
all of which are configurable via the Python dict config, but users may also
extend ``Source`` and add their own.

More details on how to create new sources in :ref:`model-integration`.

Transport model
^^^^^^^^^^^^^^^

A TORAX |TransportModel|_ computes the heat and particle turbulent transport
coefficients. |TransportModel|_ is an abstract class, and TORAX provides several
implementations, including |QLKNN|_.

See the |torax.transport_model|_ module for all implementations. Users may
extend |TransportModel|_ to create their own implementation as well. More
details in :ref:`model-integration`.

Time step calculator
^^^^^^^^^^^^^^^^^^^^

|torax.time_step_calculator|_ contains the interface and default implementations
of |TimeStepCalculator|_, the base class which computes the duration of the next
time step in TORAX and decides when the simulation is over.

Users may use one of the provided implementations or create their own by
extending |TimeStepCalculator|_.

Spectators
^^^^^^^^^^

The ``torax.spectators`` module contains objects that can "view" the progress of
a simulation as it evolves. This is useful for tools like live-plotting.

All spectators extend the |torax.spectators.spectator.Spectator|_
abstract base class, and the spectator is called/updated within the run-loop
inside |torax.sim.run_simulation()|_.


.. |run_simulation_main.py| replace:: ``run_simulation_main.py``
.. _run_simulation_main.py: https://github.com/google-deepmind/torax/blob/main/run_simulation_main.py
.. |torax/examples/| replace:: ``torax/examples/``
.. _torax/examples/: https://github.com/google-deepmind/torax/tree/main/torax/examples
.. |torax.config.build_sim.build_sim_from_config()| replace:: ``torax.config.build_sim.build_sim_from_config()``
.. _torax.config.build_sim.build_sim_from_config(): https://github.com/google-deepmind/torax/blob/main/torax/config/build_sim.py
.. |torax.sim.Sim| replace:: ``torax.sim.Sim``
.. _torax.sim.Sim: https://github.com/google-deepmind/torax/blob/main/torax/sim.py
.. |torax.sim.run_simulation()| replace:: ``torax.sim.run_simulation()``
.. _torax.sim.run_simulation(): https://github.com/google-deepmind/torax/blob/main/torax/sim.py
.. |TimeStepCalculator| replace:: ``TimeStepCalculator``
.. _TimeStepCalculator: https://github.com/google-deepmind/torax/blob/main/torax/time_step_calculator/time_step_calculator.py
.. |Stepper| replace:: ``Stepper``
.. _Stepper: https://github.com/google-deepmind/torax/blob/main/torax/stepper/stepper.py
.. |SourceModels| replace:: ``SourceModels``
.. _SourceModels: https://github.com/google-deepmind/torax/blob/main/torax/sources/source_models.py
.. |TransportModel| replace:: ``TransportModel``
.. _TransportModel: https://github.com/google-deepmind/torax/blob/main/torax/transport_model/transport_model.py
.. |sim.py| replace:: ``sim.py``
.. _sim.py: https://github.com/google-deepmind/torax/blob/main/torax/sim.py
.. |torax.state| replace:: ``torax.state``
.. _torax.state: https://github.com/google-deepmind/torax/blob/main/torax/state.py
.. |torax.state.ToraxSimState| replace:: ``torax.state.ToraxSimState``
.. _torax.state.ToraxSimState: https://github.com/google-deepmind/torax/blob/main/torax/state.py
.. |torax.geometry.Geometry| replace:: ``torax.geometry.Geometry``
.. _torax.geometry.Geometry: https://github.com/google-deepmind/torax/blob/main/torax/geometry.py
.. |torax.config.build_sim| replace:: ``torax.config.build_sim``
.. _torax.config.build_sim: https://github.com/google-deepmind/torax/blob/main/torax/config/build_sim.py
.. |torax.config.runtime_params| replace:: ``torax.config.runtime_params``
.. _torax.config.runtime_params: https://github.com/google-deepmind/torax/blob/main/torax/config/runtime_params.py
.. |torax.interpolated_param| replace:: ``torax.interpolated_param``
.. _torax.interpolated_param: https://github.com/google-deepmind/torax/blob/main/torax/interpolated_param.py
.. |torax.config.runtime_params_slice| replace:: ``torax.config.runtime_params_slice``
.. _torax.config.runtime_params_slice: https://github.com/google-deepmind/torax/blob/main/torax/config/runtime_params_slice.py
.. |torax.stepper| replace:: ``torax.stepper``
.. _torax.stepper: https://github.com/google-deepmind/torax/tree/main/torax/stepper
.. |torax.sources| replace:: ``torax.sources``
.. _torax.sources: https://github.com/google-deepmind/torax/tree/main/torax/sources
.. |QLKNN| replace:: ``QLKNN``
.. _QLKNN: https://github.com/google-deepmind/torax/blob/main/torax/transport_model/qlknn_wrapper.py
.. |torax.transport_model| replace:: ``torax.transport_model``
.. _torax.transport_model: https://github.com/google-deepmind/torax/blob/main/torax/transport_model
.. |torax.time_step_calculator| replace:: ``torax.time_step_calculator``
.. _torax.time_step_calculator: https://github.com/google-deepmind/torax/blob/main/torax/time_step_calculator
.. |torax.spectators.spectator.Spectator| replace:: ``torax.spectators.spectator.Spectator``
.. _torax.spectators.spectator.Spectator: https://github.com/google-deepmind/torax/blob/main/torax/spectators/spectator.py
