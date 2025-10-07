.. _jax_classes:

**************************************
Designing JAX-compatible classes
**************************************


* In TORAX we often make use of custom classes that are inputs or outputs of a
  function that is decorated with ``jax.jit``. In order to allow this these
  objects must either be registered as a
  `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_ or marked as static.


* The requirements for classes used as arguments to jitted functions has been an
  ongoing area of complexity. This document is
  something of a mix of:

    * Documentation of what Torax *does*
    * A design document describing how new Torax classes *should be* written
    * Incomplete design thoughts about how we might change what Torax should do
      in the future

* As of October 2025, the Torax design standard is that static arguments must
  be immutable, and must support ``__hash__`` and ``__eq__``, in order for the
  function defined by the static argument to be cached. Additionally we hash
  and compare by value not by `id`. In particular, classes with polymorphism
  must hash the class id as part of the value.

     * It's actually quite hard to find a statement of the Jax requirements on this
       subject.
     * The `jax documentation <https://docs.jax.dev/en/latest/faq.html>`_ implies
       that it is sufficient for a static argument to either be immutable (and
       hash by id) *or* hash by value.
     * The Jax documentation does say that if your class is mutable then it 'is not
       really "static"' and should be made a pytree instead.
     * The Jax documentation does seem to imply that it's OK to have an immutable
       object that hashes by `id`.
     * None of our ``__hash__`` methods are used for the persistent cache at all.
       Every time the Python interpreter is started, the ``hash`` function gets
       a different random seed, for security reasons. Hashes are thus not stable
       across two processes, even if we hash by value rather than `id`.
       The persistent cache is queried after tracing, using some kind of
       intermediate representation, where our custom classes are no longer part
       of the interface to the cache.
       See more discussion in this
       `commit <https://github.com/google-deepmind/torax/commit/d73192ed0ea52c30fbafb01cc6e0d421550b22a4>`_,
       which removed a feature for hashing the code used to generate a trace.
     * Because it is difficult to correctly hash by value, and because it seems
       likely to be unnecessary to, we may wish to switch to hashing by `id` in
       the future.
     * Incorrect hashing by value usually causes hash collisions, which result
       in incorrect Jax computations. Usually this is from forgetting to hash
       a field or the class id, so two objects that are not the same appear
       the same due to not comparing the ommitted piece of information. This
       is very bad.
     * Hashing by id would make our code a lot less complex overall, eliminate
       the possibility of these hash collisions, and the worst that would
       happen is that we would get hash misses for objects that are
       equivalent, which is not something that happens much for the type of
       objects we use as static arguments. For example, we don't tend to
       construct two equivalent `Solver` objects in the same process.
       If a hash miss does occur, the result is just slowness from retracing,
       not a wrong computation.

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

* For static arguments, we recommend inheriting from
  ``jax._src.static_dataclass.StaticDataclass`` and taking care that all the fields
  of the new class also meet the requirements of a static argument.

  * In the past we used a variety of means of implementing custom classes with
    immutability and hashing by value, but these involved a lot of error-prone
    boilerplate (it was easy to forget to hash the class id, to correctly implement
    the hash the first time but forget to update it later when adding a field,
    to overwrite the hash method with a dataclasses decorator etc).
    ``StaticDataclass`` is the preferred method going forward because it
    automatically protects against many of these errors.

* For pytree arguments, we prefer not to use ``chex.dataclass`` but rather to apply
  ``jax.tree_util.register_dataclass`` to traditional dataclasses. This approach
  allows us to specify individual fields as static or dynamic.

* There is a constraint on the structure of the class hierarchy / the structure
  of the pytree: because static arguments must be hashable, and because dynamic
  arguments (Jax tracers / NumPy arrays) are not hashable, no static class can
  have dynamic children. This means every class must be factored into part that
  is static, and a part that is dynamic but may have static leaves.

* Throughout the library, many classes called "Config" or "RuntimeParams" are
  not only playing the role of user-specified configuration, but also playing the
  role of factoring out the dynamic-with-static-leaves component of the class
  hierarchy.

* If a class is intended for use as an argument to a jitted Jax function, we
  recommend saying so in the docstring of the class, and also saying whether
  it is intended to be a dynamic argument or static argument. (It is also
  possible to be both, by making a pytree with every field registered as static,
  but in this case you should probably just make the class a static argument)
  Why specify how the class is intended to be used?

  * Intended use isn't a local property that can be read from the class itself,
    it's necessary to analyze the library and see if the class is ever passed
    as a dynamic argument, and separately if it is ever passed as a static
    argument.
  * If you make a mistake in how you implement the class it will be easier to
    tell what was intended.
  * If we change how we implement dynamic or static arguments in the future,
    it will be easier to tell what an old format of class was doing.

