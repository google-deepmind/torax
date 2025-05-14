.. _cache:

Using the Jax persistent cache
##############################

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
via the TORAX_ERRORS_ENABLED environment variable or
`torax.jax_utils.enable_errors`.
This is because runtime error handling injects Python callbacks into the Jax
program, and Jax can't serialize arbitrary callable Python objects into its
cache. Most Torax tests have runtime error handling enabled to catch correctness
bugs, so many tests do not benefit from the speedup of caching.

Another interesting feature of the Jax persistent cache is that the
persistent cache key is a function of the built graph, not the args to
function where the jit decorator is applied. This means that our hash
functions for custom classes that are arguments to the outermost function
don't need to be designed to hash the same across runs of the Python
interpreter. An example of this is hashing by `id` which works with the
persistent cache even though the `id` of an object will change if the object is
recreated. See https://github.com/google-deepmind/torax/pull/276 for more
detail.

The `tests/persistent_cache.py` test gives some good examples of usage and
includes comments with advice about debugging cases of the cache unexpectedly
not being used.

