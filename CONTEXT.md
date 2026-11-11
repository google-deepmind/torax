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
    (specifying units in square brackets such as `[m]`, `[eV]`, or `[W/m^3]`
    rather than `"in units of ..."`, along with expected shapes where
    relevant), and the return value. For private helpers, a docstring is still
    encouraged whenever the intent or physics meaning is not immediately
    obvious from the name and signature.

*   **License Preamble:** Every source file must include the Apache 2.0
    license preamble. Always add it to newly created files with the copyright
    year set to the current year. Never remove the preamble from any file,
    and do **not** modify the copyright year in existing files.

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

*   **Config Propagation:** Configuration flows from user-facing Pydantic
    configs (e.g., `NumericsConfig`, `ExtendedLengyelConfig`) to JAX-compatible,
    frozen `RuntimeParams` dataclasses via a `build_runtime_params(self, t)`
    method.
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

## Configuration Documentation

User-facing Pydantic configuration reference pages under `docs/configuration/`
are auto-generated from class docstrings via the custom Sphinx directive
`.. torax_config::` (implemented in `docs/sphinxext/torax_config_autodoc.py`).
When adding or updating a Pydantic config model:

*   **Document every field in `Attributes:`:** Every field in `model_fields`
    (or its base class) must have a matching `field_name: ...` entry under an
    `Attributes:` section in the class docstring, and every entry under
    `Attributes:` must match an actual field or `ClassVar` on the class.
    `torax_config` emits a Sphinx warning (promoted to an error under `-W`)
    on missing or unknown attributes.
*   **Register new config classes in `docs/configuration/*.rst`:** Add
    `.. torax_config:: torax._src.<module>.<ConfigClass>` to the appropriate
    page in `docs/configuration/`.
*   **Code formatting in docstrings:** Use single backticks (`` `param` ``) for
    inline code/field names and triple-backtick fences (```` ``` ````) for code
    blocks (parsed as Python by default unless another language is specified).
*   **Units in square brackets:** Write physical units in square brackets
    (e.g., `[s]`, `[m]`, `[eV]`, `[W/m^3]`, `[$rho_norm$]`) rather than prose
    phrases like `"in units of ..."` or `"in seconds"`.
*   **Citations and external links (`|...|`):** Define external URLs and
    literature citations in `docs/links.rst` (automatically appended to
    `rst_epilog` in `docs/conf.py`) and reference them via `|...|`
    substitutions in config docstrings and `.rst` pages rather than inline
    `<https://...>`_ links.
*   **Code-math lookup table (`$...$` and `$$...$$`):** Write math in
    Python-like code syntax (`^` for exponents, `=` for equality, Greek names
    without backslashes, e.g., `$T_e$`, `$rho_norm$`,
    `$chi propto (R_over_L_Ti - R_over_L_Ti_crit)^alpha$`) rather than raw
    LaTeX backslashes. Every `$...$` and `$$...$$` expression is translated via
    the explicit substitution table in
    `docs/sphinxext/math_substitutions.toml` (with whitespace normalized).
    When adding or modifying a math expression in a docstring, add or update
    its entry in `docs/sphinxext/math_substitutions.toml`; unmapped expressions
    and unmatched `$` delimiters fail loudly under Sphinx `-W` and in
    `docs/tests/torax_config_autodoc_test.py`.
*   **Verification:** Run `pytest docs/tests/torax_config_autodoc_test.py` and
    `cd docs && make html SPHINXOPTS="-W"` to verify `math_substitutions.toml`
    coverage, whole-codebase Pydantic config coverage/docstrings, and
    warning-free Sphinx builds (ReadTheDocs builds also run with
    `SPHINXOPTS="-W"`).

*   **Module-Level Constants:** Avoid using magic numbers within functions.
    Define thresholds, tolerances, or algorithm parameters as UPPER_SNAKE_CASE
    module-level constants (e.g., `_ROOT_UNIQUENESS_TOL = 1e-2`).

*   **Variable Naming:** Prioritize consistency and clarity in naming. Avoid
    excessive acronyms or contractions, but standard physics abbreviations
    (e.g., `T_e`, `n_e`, `R_major`, `I_p`) are encouraged where appropriate.
    Capital letters are permitted for physics variables, but you must add
    `# pylint: disable=invalid-name` near the top of the module to stop
    linter complaints.
    *   *Note:* Output keys in `output.py` use UPPER_SNAKE_CASE
        (e.g., `T_E`, `I_P`), and integrated scalars in `scalars/` use
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
    *   In dataclass fields using `jax.tree.static()`.
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
