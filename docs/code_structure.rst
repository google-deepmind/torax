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

For reference, the standard workflow while using TORAX is:


#.
   Configure a simulation with a dictionary of Python primitives, numpy arrays,
   or xarray datasets (see |torax/examples/|_). More on this config language
   in :ref:`configuration`.

#.
   Use |torax.ToraxConfig|_ to convert the config to a Pydantic model
   which is passed to the simulation. This is a class that contains all the
   information needed to run a simulation.

#.
   Use |torax.run_simulation()|_ to run the simulation. This takes a
   |ToraxConfig| and returns an ``xarray.Dataset`` containing the simulation
   state as described in :ref:`output`.

The rest of this section details each of these components and a few more.

Note that although TORAX is designed such that you can bring your own models
for transport, sources, etc, this still requires some TORAX development work for
interfacing. Future work with expose the TORAX for seamless coupling. See the
:ref:`model-interface` section for more details.

torax_pydantic
^^^^^^^^^^^^^^

Within TORAX, we refer to a "config" as a Python file or dictionary configuring
the building blocks of a TORAX simulation run. A "config" may define which
time-step calculator to use, which transport model to load, and where the
geometry comes from.

We then use the |torax.ToraxConfig| class to convert the config to a Pydantic
model object. This validates the provided config and conforms it into a
representation that is easy to work with, and has methods for interpolating etc.

Within the simulation we have the concept of a ``runtime parameter``, which is
the input to the simulation at a specific time step. Currently we have the
concept of ``static`` and ``dynamic`` runtime parameters. ```Static``` runtime
parameters are those that do not change during the simulation, such as the main
ion names and changing these between runs will result in a recompilation of the
JAX functions. ``Dynamic`` runtime parameters are those that do change during
the simulation such as Ip. The |torax.config.runtime_params_slice|_ module
contains both of these.


orchestration
^^^^^^^^^^^^^

|run_simulation.py|_ is the main entrypoint for running a TORAX simulation.
It takes a ToraxConfig and returns the xarray.Dataset of the simulation as
described in :ref:`output`. It creates the various models, providers and initial
state needed for the simulation and creates a ``StepFunction``
which steps the simulation over time.

|initial_state.py|_ contains the logic for creating the initial state as well as
the logic for restarting a simulation from a previous state file.

|step_function.py|_ contains ``StepFunction`` which is the class used to step
the simulation. This is currently within ``_src`` as it may be subject to API
changes but users who want to write their own simulation loops can experiment
with this. If you want to use this in more permanent code, please reach out to
our team to help us understand your use case.

output_tools
^^^^^^^^^^^^

The |torax.output_tools|_ module contains the structures of the outputs of
a TORAX simulation. These are an ``xarray.Dataset`` and a ``StateHistory``
object which can be used for debugging.

state
^^^^^

The |torax.state|_ module describes the internal state of TORAX.
|torax.sim_state.ToraxSimState|_ is used to keep track of the internal state
throughout a simulation. Each time step generates a new state.

geometry
^^^^^^^^

|Geometry|_ describes the geometry of the torus and fields.
See the ``torax.geometry`` package for different ways to build a
``Geometry`` and its child classes.

Each ``Geometry`` object represents the geometry at a single point in time.
A sequence of different geometries at different timesteps can be provided at
config, and the various ``Geometry``s are constructed upon initialization.
These are packaged together into a ``GeometryProvider`` which interpolates a
new ``Geometry`` at each timestep allowing for dynamic time stepping.

solver
^^^^^^^

|torax.solver|_ contains PDE time solvers that discretize the PDE in time and
solve for the next time step with linear or nonlinear methods.

Inside the |Solver|_ implementations is where JAX is actually used to compute
Jacobians or do optimization-based solving. See the implementations for more
details.

.. _structure-sources:

sources
^^^^^^^

The |torax.sources|_ module contains all source models plugged into TORAX. They
are packaged together into a |SourceModels|_ object, which is a simple container
to help access all the sources while stepping through the simulation.

A TORAX ``Source`` produces heat, particle, or current deposition profiles used
to compute PDE source/sink coefficients used while solving for the next
simulation state. TORAX provides several default source model implementations,
all of which are configurable via the Python dict config.

See the |torax.sources|_ module for all implementations.

.. _structure-transport-model:

transport
^^^^^^^^^

A TORAX |TransportModel|_ computes the heat and particle turbulent transport
coefficients. |TransportModel|_ is an abstract class, and TORAX provides several
implementations, including |QLKNN|_.

See the |torax.transport_model|_ module for all implementations.

pedestal
^^^^^^^^

A TORAX |PedestalModel|_ imposes the plasma temperature and density at a desired
internal location. This is intended to correspond to the top of the H-mode
pedstal. The operation of the pedestal is controlled by a time-dependent
configuration attribute. |PedestalModel|_ is an abstract class, and TORAX
currently provides two simple implementations.

See the |torax.pedestal_model|_ module for all implementations.

mhd
^^^

The |torax.mhd|_ module currently just contains the sawtooth model which models
the crash in temperature and density at the centre of plasma. This is currently
only a simple analytical model and can be extended by more complex models for
trigger and redistribution in the future.

neoclassical
^^^^^^^^^^^^

The |torax.neoclassical|_ module contains the neoclassical conductivity and
bootstrap current models. It currently uses the Sauter model but can be extended
with more models in future. Near term work is also planned to add neoclassical
transport.

time_step_calculator
^^^^^^^^^^^^^^^^^^^^

|torax.time_step_calculator|_ contains the interface and default implementations
of |TimeStepCalculator|_, the base class which computes the duration of the next
time step in TORAX and decides when the simulation is over.

.. |run_simulation_main.py| replace:: ``run_simulation_main.py``
.. _run_simulation_main.py: https://github.com/google-deepmind/torax/blob/main/run_simulation_main.py
.. |torax/examples/| replace:: ``torax/examples/``
.. _torax/examples/: https://github.com/google-deepmind/torax/tree/main/torax/examples
.. |torax.sim.run_simulation()| replace:: ``torax.sim.run_simulation()``
.. _torax.sim.run_simulation(): https://github.com/google-deepmind/torax/blob/main/torax/orchestration/run_simulation.py
.. |TimeStepCalculator| replace:: ``TimeStepCalculator``
.. _TimeStepCalculator: https://github.com/google-deepmind/torax/blob/main/torax/_src/time_step_calculator/time_step_calculator.py
.. |Solver| replace:: ``Solver``
.. _Solver: https://github.com/google-deepmind/torax/blob/main/torax/_src/stepper/stepper.py
.. |SourceModels| replace:: ``SourceModels``
.. _SourceModels: https://github.com/google-deepmind/torax/blob/main/torax/_src/sources/source_models.py
.. |TransportModel| replace:: ``TransportModel``
.. _TransportModel: https://github.com/google-deepmind/torax/blob/main/torax/_src/transport_model/transport_model.py
.. |PedestalModel| replace:: ``PedestalModel``
.. _PedestalModel: https://github.com/google-deepmind/torax/blob/main/torax/_src/pedestal_model/pedestal_model.py
.. |torax.state| replace:: ``torax.state``
.. _torax.state: https://github.com/google-deepmind/torax/blob/main/torax/_src/state.py
.. |torax.sim_state.ToraxSimState| replace:: ``torax.sim_state.ToraxSimState``
.. _torax.sim_state.ToraxSimState: https://github.com/google-deepmind/torax/blob/main/torax/_src/state.py
.. |Geometry| replace:: ``Geometry``
.. _Geometry: https://github.com/google-deepmind/torax/blob/main/torax/_src/geometry/geometry.py
.. |torax.config.runtime_params_slice| replace:: ``torax.config.runtime_params_slice``
.. _torax.config.runtime_params_slice: https://github.com/google-deepmind/torax/blob/main/torax/_src/config/runtime_params_slice.py
.. |torax.solver| replace:: ``torax.solver``
.. _torax.solver: https://github.com/google-deepmind/torax/tree/main/torax/_src/stepper
.. |torax.sources| replace:: ``torax.sources``
.. _torax.sources: https://github.com/google-deepmind/torax/tree/main/torax/_src/sources
.. |QLKNN| replace:: ``QLKNN``
.. _QLKNN: https://github.com/google-deepmind/torax/blob/main/torax/_src/transport_model/qlknn_transport_model.py
.. |torax.transport_model| replace:: ``torax.transport_model``
.. _torax.transport_model: https://github.com/google-deepmind/torax/blob/main/torax/_src/transport_model
.. |torax.pedestal_model| replace:: ``torax.pedestal_model``
.. _torax.pedestal_model: https://github.com/google-deepmind/torax/blob/main/torax/_src/pedestal_model
.. |torax.time_step_calculator| replace:: ``torax.time_step_calculator``
.. _torax.time_step_calculator: https://github.com/google-deepmind/torax/blob/main/torax/_src/time_step_calculator
.. |torax.output_tools| replace:: ``torax.output_tools``
.. _torax.output_tools: https://github.com/google-deepmind/torax/blob/main/torax/_src/output_tools
.. |step_function.py| replace:: ``step_function.py``
.. _step_function.py: https://github.com/google-deepmind/torax/blob/main/torax/_src/orchestration/step_function.py
.. |initial_state.py| replace:: ``initial_state.py``
.. _initial_state.py: https://github.com/google-deepmind/torax/blob/main/torax/_src/orchestration/initial_state.py
.. |run_simulation.py| replace:: ``run_simulation.py``
.. _run_simulation.py: https://github.com/google-deepmind/torax/blob/main/torax/_src/orchestration/run_simulation.py
.. |torax.sim.run_simulation()| replace:: ``torax.sim.run_simulation()``
.. _torax.sim.run_simulation(): https://github.com/google-deepmind/torax/blob/main/torax/_src/orchestration/run_simulation.py
.. |torax.ToraxConfig| replace:: ``torax.ToraxConfig``
.. _torax.ToraxConfig: https://github.com/google-deepmind/torax/blob/main/torax/_src/torax_pydantic/model_config.py
.. |torax.mhd| replace:: ``torax.mhd``
.. _torax.mhd: https://github.com/google-deepmind/torax/blob/main/torax/_src/mhd
.. |torax.neoclassical| replace:: ``torax.neoclassical``
.. _torax.neoclassical: https://github.com/google-deepmind/torax/blob/main/torax/_src/neoclassical
