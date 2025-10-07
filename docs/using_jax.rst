.. _using_jax:

##################
Using Jax in TORAX
##################

******************************
Using the Jax persistent cache
******************************

Torax is based on Jax. Each time we run the Python interpreter, Torax uses Jax
to "trace" or construct mathematic expressions, then compiles these into executable
programs. The tracing and compilation often take longer than the execution of the
program itself, especially if Torax isn't run for many timesteps.

It's possible to use a feature of Jax called the persistent cache to store the
output of compilation on the filesystem to avoid recompilation each time we
run Torax (or any other program using Jax). There are several limitations to this:
if Jax or Torax is updated, or if any config settings affecting the expressions
built by Torax change, Torax will build a different expression and need to compile
a new program. Also, as of this writing (May 2025), Jax caches only the
compilation step, not the tracing step.

See :ref:`how_to_install` for information on how to set your environment variables
to always use the cache by default.
The `Jax persistent cache documentation <https://docs.jax.dev/en/latest/persistent_compilation_cache.html#persistent-compilation-cache>`_
gives some more information.
Some particularly useful information includes:

* How to use command line flags or python config setter functions instead
  of environment variables to change cache settings on a case by case basis
* How to enable debugging logging information related to the cache, to get
  messages about whether / why not functions are written to / read from the cache

One Torax-specific cache gotcha is that the cache may not be used if Torax runtime
error handling is turned on (it is off by default)
via the ``TORAX_ERRORS_ENABLED`` environment variable.
This is because runtime error handling injects Python callbacks into the Jax
program, and Jax can't serialize arbitrary callable Python objects into its
cache. Most Torax tests have runtime error handling enabled to catch correctness
bugs, so many tests do not benefit from the speedup of caching.

Another interesting feature of the Jax persistent cache is that the
persistent cache key is a function of the built graph, not the args to
function where the jit decorator is applied. This means that our hash
functions for custom classes that are arguments to the outermost function
don't need to be designed to hash the same across runs of the Python
interpreter. An example of this is hashing by ``id`` which works with the
persistent cache even though the ``id`` of an object will change if the object is
recreated. See https://github.com/google-deepmind/torax/pull/276 for more
detail.

The `tests/persistent_cache.py` test gives some good examples of usage and
includes comments with advice about debugging cases of the cache unexpectedly
not being used.

*******************
Using JAX callbacks
*******************

In instances where you want to add a component to the TORAX simulation that
is not easily expressed in JAX, you can use JAX callbacks. This allows JAX to
execute regular Python code on the host and can be used to embed Python code
within a ``jax.jit`` scope.

See https://docs.jax.dev/en/latest/external-callbacks.html#external-callbacks
for more details.

For an example of how this is currently used in TORAX see the
``qualikiz_transport_model.py``.

########################################
Defining JIT compatible objects in TORAX
########################################

In TORAX we often make use of custom classes that are inputs or outputs of a
function that is decorated with ``jax.jit``. In order to allow this these
objects must either be registered as a
[Pytree](https://jax.readthedocs.io/en/latest/pytrees.html) or marked as static.

Objects marked as static must define a ``__hash__`` method and a
``__eq__`` method. These will be used to decide whether to re-execute
compilation (if the hash is the same and objects are equal) or re-use the cached
compilation.

**************
Custom pytrees
**************

For most cases where we want to define a custom pytree we make use of a regular
dataclass as well as the ``jax.tree_util.register_dataclass`` decorator. This
allows us to define a custom dataclass that contains both JAX array types and
regular python types that can be marked as static in the same object.

*********************
Custom static classes
*********************

We also define various classes for holding physics models that are marked as
static in ``jax.jit`` decorators. This includes the ``source_models``,
``transport_model``, ``pedestal_model``, ``neoclassical_models`` and
``mhd_models``.

Static arguments should be hashable, meaning both ``__hash__`` and ``__eq__``
are implemented, and immutable.

In TORAX the interface for the above physics models defines abstract methods for
``__hash__`` and ``__eq__`` that must be defined by any child classes. The child
classes hold the responsibility of what constitutes a JAX cache hit/miss in
these method implementations.

Immutability is currently enforced in these classes by a ``_frozen`` attribute
on the object that is set to ``True`` in the initializer. The object's
``__setattr__`` method is also overridden to raise an ``AttributeError`` if
``_frozen`` is ``True``.







