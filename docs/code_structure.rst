.. _structure:

TORAX code structure
####################

This page describes the structure of the TORAX repository and gives pointers for
where to find different components in the code base.

This page is just an overview. For more information on the logical flow of
TORAX, read through the code and reach out to our team if anything is unclear!
We strive to make the code readable and understandable with the help of this
guide, so feedback is appreciated :)

Runnable entrypoint
-------------------

|run_simulation_main.py| is the main runnable entrypoint
for trying out TORAX. If you are interested in getting started quickly, check
out :ref:`quickstart`.

TORAX library
-------------

The TORAX library contains modules to build and run a transport simulation. This
section describes several of the modules and how they fit together.

For reference, the standard workflow while using TORAX is:


#.
   Configure a simulation with a dictionary of Python primitives, numpy arrays,
   or xarray datasets (see |torax/examples/|). More on this config language
   in :ref:`configuration`.

#.
   Use |torax.ToraxConfig| to convert the config to a Pydantic model
   which is passed to the simulation. This is a class that contains all the
   information needed to run a simulation.

#.
   Use |torax.run_simulation()| to run the simulation. This takes a
   |torax.ToraxConfig| and returns an ``xarray.DataTree`` containing the
   simulation state as described in :ref:`output`.

The rest of this section details each of these components and a few more.

Note that although TORAX is designed such that you can bring your own models
for transport, sources, etc. Future work will expose this further for seamless
coupling. See the :ref:`model-integration` section for more details.

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
the input to the simulation at a specific time step. These are held in the
``RuntimeParams`` object and created by the ``RuntimeParamsProvider``.
These parameters are used to control the simulation and may or may not be
changed between time steps. The |config.runtime_params_slice| module contains
the container for these parameters for all the different models.

Because the ``RuntimeParams`` are used across the simulation they have to be
made to work with JAX compilation. For most parameters this works out the box
but for some parameters we have marked them as ``static`` according to JAX which
allows us to use them in standard python control flow. The impacts of this is
that ``static`` parameters are treated as compile time constants (see
`jax docs <https://docs.jax.dev/en/latest/jit-compilation.html#marking-arguments-as-static>`_
for details). These ``static`` parameters cause a recompilation of the JAX
functions for each new value they are given. An example of a ``static``
parameter is the ``mode`` of the ``Sources``.


orchestration
^^^^^^^^^^^^^

|torax.run_simulation| is the main entrypoint for running a TORAX simulation.
It takes a |torax.ToraxConfig| and returns the ``xarray.DataTree`` of the
simulation as described in :ref:`output`, as well as a |torax.StateHistory|
object containing a sequence of simulation states per timestep in the form of
internal TORAX data containers, which can be used for debugging.
|torax.run_simulation| creates the various models, providers and initial state
needed for the simulation and creates a ``StepFunction`` which steps the
simulation over time.

|orchestration.initial_state.py| contains the logic for creating the initial
state as well as the logic for restarting a simulation from a previous state
file.

|orchestration.step_function.py| contains ``StepFunction`` which is the class
used to step the simulation. This is currently within ``_src`` as it may be
subject to API changes but users who want to write their own simulation loops
can experiment with this. If you want to use this in more permanent code, please
reach out to our team to help us understand your use case.

output_tools
^^^^^^^^^^^^

The |output_tools| module contains the structures of the outputs of a TORAX
simulation. These are the ``xarray.DataTree`` and |torax.StateHistory| objects.

state
^^^^^

The |state| module describes the internal state of TORAX. |ToraxSimState| is
used to keep track of the internal state throughout a simulation. Each time step
generates a new state.

geometry
^^^^^^^^

|torax.Geometry| describes the geometry of the torus and fields. See the
|geometry| package for different ways to build a ``Geometry`` and its child
classes.

Each ``Geometry`` object represents the geometry at a single point in time.
A sequence of different geometries at different timesteps can be provided at
config, and the various ``Geometry`` objects are constructed upon
initialization. These are packaged together into a
|GeometryProvider| which, interpolates a new ``Geometry`` at each timestep
allowing for dynamic time stepping. For numerical geometries this is implemented
as a |StandardGeometryProvider|.

solver
^^^^^^^

|solver| contains PDE time solvers that discretize the PDE in time and solve
for the next time step with linear or nonlinear methods.

Inside the |Solver| implementations is where JAX is actually used to compute
Jacobians or do optimization-based solving. See the implementations for more
details.

.. _structure-sources:

sources
^^^^^^^

The |sources| subpackage contains all source models plugged into TORAX. They are
packaged together into a |SourceModels| object, which is a simple container to
help access all the sources while stepping through the simulation.

A TORAX ``Source`` produces heat, particle, or current deposition profiles used
to compute PDE source/sink coefficients used while solving for the next
simulation state. TORAX provides several default source model implementations,
all of which are configurable via the Python dict config.

See the |sources| subpackage for all implementations.

.. _structure-transport-model:

transport
^^^^^^^^^

A TORAX |TransportModel| computes the heat and particle turbulent transport
coefficients. |TransportModel| is an abstract class, and TORAX provides several
implementations, including |QLKNN|.

See the |transport_model| subpackage for all implementations.

pedestal
^^^^^^^^

A TORAX |PedestalModel| imposes the plasma temperature and density at a desired
internal location. This is intended to correspond to the top of the H-mode
pedestal. The operation of the pedestal is controlled by a time-dependent
configuration attribute. |PedestalModel| is an abstract class, and TORAX
currently provides two simple implementations.

See the |pedestal_model| modules for all implementations.

mhd
^^^

The |mhd| module currently just contains the sawtooth model which models the
crash in temperature and density at the centre of plasma. This is currently only
a simple analytical model and can be extended by more complex models for trigger
and redistribution in the future.

neoclassical
^^^^^^^^^^^^

The |neoclassical| module contains the neoclassical conductivity and bootstrap
current models. It currently uses the Sauter model but can be extended with more
models in future. Near term work is also planned to add neoclassical transport.

time_step_calculator
^^^^^^^^^^^^^^^^^^^^

|time_step_calculator| contains the interface and default implementations of
|TimeStepCalculator|, the base class which computes the duration of the next
time step in TORAX and decides when the simulation is over.
