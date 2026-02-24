# TORAX Development Guide

This file contains conventions, patterns, and anti-patterns for the Torax
project.

## Code Quality

*   **Style Guide:** Follow the
    [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

*   **Linting:** ALWAYS clean linter errors before sending a PR. Run `pyink`
    for formatting, and check with your project linter.

*   **Small PRs:** Keep pull requests small and self-contained.
    One PR should address one thing.
    Smaller PRs are reviewed faster, more thoroughly, and are less likely to
    introduce bugs. Aim for <100 lines; 1000+ is almost always too large.
    Separate refactorings from feature changes.

*   **Reuse Existing Components:** Before writing new utilities, helpers, or
    abstractions, search the codebase for existing ones. Prefer reusing and
    extending existing code over duplicating functionality.

*   **Minimise Verbosity:** Prefer concise, readable code. Avoid unnecessary
    layers of abstraction, redundant variables, and boilerplate. If the
    standard library or JAX/NumPy already provides a function, use it.

*   **Comments:** Keep inline comments targeted and high-value. Avoid restating
    logic that is obvious from the code, and prefer clear naming over
    explanatory comments. However, **do** add inline comments to explain the
    physics reasoning or domain logic behind non-trivial operations—e.g.,
    why a particular equation form is used, what physical assumption a
    simplification relies on, or how a numerical trick improves stability.
    When implementing physics models, cite the source paper or textbook
    (e.g., author, title, equation number) near the relevant code.

*   **Docstrings:** All public functions, methods, and classes **must** have
    docstrings. Docstrings should describe the purpose, the arguments
    (including units and expected shapes where relevant), and the return
    value. For private helpers, a docstring is still encouraged whenever the
    intent or physics meaning is not immediately obvious from the name and
    signature.

## Build & test commands

For build and test instructions, see
[contribution_tips.rst](https://torax.readthedocs.io/en/latest/contribution_tips.html).

## Import & Execution conventions

*   **JAX Initialization:** In Python scripts, to avoid a `RuntimeError:
    Attempted call to JAX before absl.app.run()` error, encapsulate all
    module-level JAX calls inside a `main()` function and execute it using
    `absl.app.run(main)`.

*   **Absolute Imports:** Prefer `from torax._src.[module] import ...`.

## Patterns and Anti-patterns

*   **Tracer Type Errors:** NEVER use Python `if x < 10:` on JAX tracers; use
    `jnp.where` or `jax.lax.cond`. When passing a tracer to a strict `Enum` or
    `bool` dataclass field, append `# pytype: disable=wrong-arg-types`.

*   **No "God Objects" for I/O:** Do NOT add complex domain logic to generic
    output tools like `output.py`! Move identification/filtering logic (e.g.,
    root deduplication, convergence checking) to the individual `Outputs` data
    containers instead.

*   **Config Propagation:** Configuration flows from user-facing Pydantic models
    (e.g., `Numerics`, `ExtendedLengyelConfig`) to JAX-compatible, frozen
    `RuntimeParams` dataclasses via a `build_runtime_params(self, t)` method.
    The `t` argument evaluates time-dependent parameters (like
    `TimeVaryingScalar`) at a specific simulation time slice to yield concrete
    numerical values. Pydantic models use specialized types
    (e.g., `torax_pydantic.UnitInterval`), while `RuntimeParams` uses primitive
    types (`float`, `bool`) and explicit JAX-compatible type hints from
    `torax._src.array_typing` (e.g., `FloatScalar`, `FloatVector`) to ensure
    static shape and type clarity for JAX compilation.

*   **Pydantic Validators:** Use `@pydantic.model_validator(mode='before'|'after')`
    to validate complex inter-dependencies between config fields (e.g., ensuring
    `enrichment_factor` matches `seed_impurity_weights`). Always return `self`
    (or `data`) from these validators.

*   **Module-Level Constants:** Avoid using magic numbers within functions.
    Define thresholds, tolerances, or algorithm parameters as UPPER_SNAKE_CASE
    module-level constants (e.g., `_ROOT_UNIQUENESS_TOL = 1e-2`).

*   **Variable Naming:** Prioritize consistency and clarity in naming. Avoid
    excessive acronyms or contractions, but standard physics abbreviations
    (e.g., `T_e`, `n_e`, `R_major`, `Ip`) are encouraged where appropriate.
    Capital letters are permitted for physics variables, but you must add
    `# pylint: disable=invalid-name` near the top of the module to stop
    linter complaints.
    *   *Note:* Output keys in `output.py` use UPPER_SNAKE_CASE
        (e.g., `T_E`, `IP`), and integrated scalars in `scalars/` use
        formats like `P_SOL`.

*   **Dtype Conventions & Array Initialization:** Prefer `jnp.asarray` over
    `jnp.array` when wrapping existing data to avoid unnecessary copies. When
    explicitly specifying a precision (e.g., for `jnp.zeros` or `jnp.array`),
    use `jax_utils.get_dtype()`, `jax_utils.get_np_dtype()`, or
    `jax_utils.get_int_dtype()` rather than hardcoding `jnp.float64` or
    `jnp.float32`.

## Architecture invariants

*   **Staticity for Shapes:** Any parameter that determines array sizes, shapes,
    or loop counts (e.g., `num_guesses`, `maxiter`, `grid_size`) MUST be
    explicitly marked static:
    *   In the `@jax.jit` decorator (`static_argnames`).
    *   In dataclass fields using `metadata={'static': True}`.
    *   In Pydantic configurations using `torax_pydantic.JAX_STATIC`.

*   **Distinct Xarray Dimensions:** Ensure distinct xarray dimension names
    (e.g., `output.IMPURITY` vs `output.SEED_IMPURITY`) when grouping fields in
    a Dataset to prevent unintended shape alignments and `NaN` padding.

## Testing conventions

*   **Test Locality:** Preferably add to existing test modules instead of making
    new test modules.

*   **Test Scope:** Focus on having more, smaller tests that verify one specific
    thing or behavior, rather than large tests that cover a broad theme.

*   **Mock Variables:** Always use keyword arguments (never positional) when
    mocking or creating complex dataclasses like `CoreProfiles` or
    `EdgeModelOutputs`.

*   **Parameterized Setup:** Use `@parameterized.parameters` to split testing
    into explicit regimes, rather than combining all logic under boolean flags.

*   **JIT Mocks:** If mocking a dependency that a JIT-compiled function relies
    on, ensure `.clear_cache()` is called in `setUp()` to clear stale trace
    paths.

*   **Source Tests:** Source modules generally inherit from
    `test_lib.SourceTestCase`.

*   **Updating Numerical Test References:** When updating reference `.nc` files
    for numerical tests (`sim_test`, `compare_test`, `copy_sim_test`, etc.),
    see [contribution_tips.rst](https://torax.readthedocs.io/en/latest/contribution_tips.html)
    (section "Testing") for the full workflow.
