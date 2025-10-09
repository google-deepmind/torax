.. _jax_classes:

**************************************
Designing JAX-Compatible Classes
**************************************


* In TORAX we often make use of custom classes that are inputs or outputs of a
  function that is decorated with ``jax.jit``. In order to allow this these
  objects must either be registered as a
  `Pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_ or marked as static.

* To behave correctly, static arguments must be immutable, and must support
  ``__hash__`` and ``__eq__``, in order for the function defined by the static
  argument to be cached.

* See :doc:`using_jax` for a discussion of the Jax persistent cache. Torax
  classes that implement ``__hash__`` and ``__eq__`` need to do so in a way that
  works with the Jax persistent cache, which may introduce further considerations
  beyond those for caching within a single process.

* Jax does not enforce correct behavior. It is on us to do so.

* Fairly innocuous behavior can lead to silent computation of incorrect results,
  rather than errors. For example, if we were to use a traditional Python class
  as a static argument, this fails the immutability requirement, but there is no
  mutability test so no error is raised. Furthermore, Python supplies default
  implementations of ``__hash__`` and ``__eq__`` that are based on object
  identity, rather than mathematical properties, so the objects continue to hash
  and compare the same despite having different values. The following code
  illustrates this failure::

    import functools
    import jax

    @functools.partial(jax.jit, static_argnums=0)
    def f(x):
      return x.num_wheels

    class Vehicle:
      def __init__(self, num_wheels):
        self.num_wheels = num_wheels

    x = Vehicle(2)
    print(f(x))
    x.num_wheels = 4
    print(f(x))

  The user presumably intends for this to print a 2 and then a 4, but it prints a
  2 both times.

* Dataclasses provide a natural way of building classes with frozen fields
  and with hashing and comparison by value rather than identity. We recommend
  using dataclasses for all classes passed to jitted functions.

* We prefer not to use ``chex.dataclass`` but rather to apply
  ``jax.tree_util.register_dataclass`` to traditional dataclasses. This approach
  allows us to specify individual fields as static or dynamic. In the past we have
  used ``chex.dataclass`` but we are phasing this out.

* There is a constraint on the structure of the class hierarchy / the structure
  of the pytree: because static arguments must be hashable, and because dynamic
  arguments (Jax tracers / NumPy arrays) are not hashable, no static class can
  have dynamic children. This means every class must be factored into part that
  is static, and a part that is dynamic but may have static leaves.

* Throughout the library, many classes called "Config" or "RuntimeParams" are
  not only playing the role of user-specified configuration, but also playing the
  role of factoring out the dynamic-with-static-leaves component of the class
  hierarchy.

* In the past we used custom classes with our own implementation of frozen
  fields, ``__hash__``, and ``__eq__``. We are now phasing this out. It is more
  concise and less error prone to use dataclasses.

  * The implementation of freezing only needs to be done once.

  * We do not need to be concerned about maintaining the list of fields to be
    hashed, the dataclass does it automatically.

  * We do not need to make a new ``__hash__`` and ``__eq__`` method for each
    class.

  * In the past we avoided using dataclasses because some classes needed
    complicated logic in the ``__init__`` method. This is best handled by
    builder functions.

  * In the past we avoided using dataclasses because "culturally" dataclasses
    are intended to be used for objects that store data and don't have a lot of
    methods and polymorphic inheritance, while for TORAX polymorphic inheritance
    is common. We have found that there is no strong technical reason not to use
    dataclasses in these cases, and the Jax-specific reasons to use dataclasses
    are strong.

