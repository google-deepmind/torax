# TORAX Development Guide

This file contains shared conventions, patterns, and anti-patterns for the
TORAX project.

## Code Quality

*   **Style Guide:** Follow the
    [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

*   **Linting & formatting:** ALWAYS clean linter errors before sending a PR.
    Run `pyink` for formatting, and check with your project linter. Respect the
    80-character line limit across both Python source files and `.rst`
    documentation.

*   **Small PRs & Concise Descriptions:** Keep pull requests small and
    self-contained. One PR should address one thing. Smaller PRs are reviewed
    faster, more thoroughly, and are less likely to introduce bugs. Aim for
    <100 lines; 1000+ is almost always too large. Separate refactorings and
    generic infrastructure changes (e.g., `output_tools`, `torax_pydantic`,
    `plotting`) into standalone PRs rather than bundling them with new physics
    features. Keep the top of each PR/commit description tight and
    human-readable: a concise single-line headline plus 2-4 lines (or crisp
    1-line bullets) explaining *what* changed and *why* — avoid verbose,
    line-by-line recaps of the diff. Place longer technical discussion,
    numerical deltas, or plot links lower down (required whenever a PR updates
    `sim_test` reference `.nc` files; see **Updating Numerical Test
    References**).

*   **Reuse Existing Components:** Before writing new utilities, helpers, or
    abstractions, search the codebase for existing ones. Prefer reusing and
    extending existing code over duplicating functionality.

*   **Minimise Verbosity & Structure Clearly:** Prefer concise, readable code.
    Avoid unnecessary layers of abstraction, redundant variables, boilerplate,
    assigned lambdas (`f = lambda ...`), and creating zero-filled output
    placeholders only to overwrite them with `dataclasses.replace`. If the
    standard library or JAX/NumPy already provides a function, use it. Decompose
    long monolithic physics functions (~>150 lines) into focused private helpers
    placed **below** the public entry point so readers see the high-level API
    first. Keep code, docstrings, and PR descriptions self-contained rather than
    relying on internal design-doc shorthand or labels.

*   **Error Handling vs. Assertions:** Use explicit `ValueError`/`TypeError` for
    runtime input and config checks rather than `assert` (which is stripped
    under `-O`). Reserve `assert` for static type-checker narrowing where
    needed.

*   **Type-Checker Suppressions (`pyrefly`):** Avoid adding broad
    `# pyrefly: ignore` suppressions when an explicit narrowing helper or union
    accessor works. When a suppression is genuinely required (e.g., passing a
    JAX tracer or working around a checker limitation), use a targeted
    `# pyrefly: ignore[code]` rather than legacy `# pytype: disable=...`.

*   **Comments:** Keep inline comments targeted and high-value. Avoid restating
    logic that is obvious from the code, and prefer clear naming over
    explanatory comments. However, **do** add inline comments to explain the
    physics reasoning or domain logic behind non-trivial operations—e.g.,
    why a particular equation form is used, what physical assumption a
    simplification relies on, or how a numerical trick improves stability.
    When implementing physics models, cite the source paper or textbook
    (author, title, and verified DOI or URL) in the class or function docstring
    (using `|...|` substitutions from `docs/links.rst` in Sphinx-exported
    config docstrings), and note specific equation numbers inline next to
    non-trivial formulas where helpful. Keep long prose derivations in
    `docs/physics_models/*.rst`, not inline comment walls. Do not put `TODO`
    markers inside public docstrings exported to Sphinx.

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
    `bool` dataclass field, append a targeted
    `# pyrefly: ignore[bad-argument-type]` comment rather than legacy
    `# pytype: disable=...`.

*   **Physics Precision & Scale-Safe Floors:** Be careful when using the
    dimensionless `CONSTANTS.eps` (`1e-7`) as a floor or division guard. It is
    acceptable for O(1) dimensionless quantities (e.g., `rho_norm`, `q`,
    `Z_eff`) and large-magnitude SI quantities where `1e-7` is negligible
    (e.g., `n_e ~ 1e20 m^-3`, `P_SOL ~ 1e6 W`, `B_0 ~ 1-10 T`), but **do not**
    use it on small-scale dimensional quantities (e.g., resistivity
    `eta ~ 1e-8 to 1e-7 Ohm*m`, short timescales, or small fluxes/gradients)
    where `1e-7` is comparable to the physical scale and silently biases
    results. In those cases, derive floors relative to a characteristic
    physical scale or `jnp.finfo(dtype).eps`. Also check sign conventions under
    reversed `B_0 < 0` / `I_p`, and verify both `float64` and `float32`
    (`JAX_PRECISION=f32`, where machine epsilon `~1.2e-7` exceeds
    `CONSTANTS.eps`) without overflow or underflow.

*   **Reverse-Mode Autodiff (`jax.grad`) Safety:** Guard zero denominators and
    non-smooth points inside *both* branches of `jnp.where` before computing
    operations like `sqrt` or division (`jax.grad` evaluates both branches).
    Reuse `torax._src.math_utils` helpers (`sqrt_with_zero_gradient_at_zero`,
    `safe_divide`, `cumulative_trapezoid`, `smooth_sqrt`,
    `smoothstep_transition`) rather than writing bespoke variants.

*   **Hot-Loop & Nested JIT Hygiene:** Do not decorate private helpers with
    `@jax.jit` when they are only called inside an already-JITted step loop,
    and avoid computing the same derived quantity twice on the same state per
    step.

*   **Keep Derived State Consistent:** When a frozen dataclass stores cached or
    derived quantities alongside its primary state, avoid partial
    `dataclasses.replace(...)` updates that can leave dependent fields stale;
    prefer existing factory constructors (e.g.,
    `PoloidalFluxProfiles.from_psi(...)`).

*   **No "God Objects" for I/O & Outputs:** Do NOT add complex domain logic to
    generic output tools like `output.py`! Move serialization (`to_xr_datatree`,
    `to_output_dict`), identification, and filtering logic onto individual
    `Outputs` data containers instead. Guard optional `time` dimensions
    (`output_keys.TIME in ds.dims`) before calling time-indexed xarray
    operations such as `drop_duplicates` or `reindex`. In `plotting/`, reference
    `output_keys.*` constants instead of raw strings, keep `default_plot_config`
    aligned when adding new current/transport components, and prefer `NaN` or
    finite bounds over `±inf` sentinels in exported arrays.

*   **Enums Over String Literals:** For discrete mode/option selectors inside
    `_src/` configs and models, prefer `enum.StrEnum` paired with `match`/`case`
    over bare `Literal[...]` string unions and `if`/`elif` chains.

*   **Discriminated Unions & Formulation Accessors:** When supporting multiple
    formulations behind one polymorphic config/model field (e.g.,
    `Field(discriminator='model_name')`), expose typed formulation accessors
    (one per layer: config, built models, `RuntimeParams`, `SimState`) rather
    than scattering `isinstance`/`getattr` checks across callers. Always return
    mutable nested configs via `pydantic.Field(default_factory=...)`, never a
    shared mutable default instance.

*   **Pydantic Config Defaults:** Only give a Pydantic config field a default
    value when that value is physically sensible across tokamaks/machines;
    leave machine-specific parameters required so callers set them explicitly.
    Similarly, require explicit values for fields whose default would otherwise
    be a silent no-op. Define each default in a single layer (the Pydantic
    config) instead of duplicating the same literal across Config, Model, and
    helper signatures.

*   **Stateful Model Serialization (`/model_state/*`):** Cross-timestep model
    state uses a symmetric DataTree pattern across `_src/model_state.py` and
    `_src/models.py` (see `pedestal_state.py` and `transport_state.py` for
    examples). Whenever adding a new field or state subclass to `ModelState`:
    *   Implement `to_xr_datatree(self, context) -> xr.DataTree | None` on the
        state dataclass, returning `None` when stateless so unaffected runs and
        reference `.nc` files stay untouched.
    *   Wire it inside `ModelState.to_xr_datatree` under
        `/model_state/<subsystem>`.
    *   Implement `initial_state_from_datatree(datatree, runtime_params)` on
        the matching model container and wire it in
        `Models.initial_state_from_datatree`.
    *   Add a file-restart round-trip test verifying exact state recovery across
        a step boundary.

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

*   **Bounded Time-Varying Types:** When configuring parameters with lower or
    upper bounds (such as positivity, interval `[0, 1]`, or custom bounds like
    `> 1.0`), prefer using `torax_pydantic`'s type helpers:
    *   `torax_pydantic.PositiveTimeVaryingScalar` (`> 0`) /
        `PositiveTimeVaryingArray`
    *   `torax_pydantic.NonNegativeTimeVaryingScalar` (`>= 0`) /
        `NonNegativeTimeVaryingArray`
    *   `torax_pydantic.UnitIntervalTimeVaryingScalar` (`[0, 1]`)
    *   For custom bounds: pair `Annotated[torax_pydantic.TimeVaryingScalar,`
        `...]` with `torax_pydantic.scalar_bounds_validator(...)` or
        `torax_pydantic.array_bounds_validator(...)`.
    Avoid writing bespoke, one-off `@pydantic.field_validator` or
    `AfterValidator` functions for simple scalar/array bound checking.

*   **Pydantic Validators:** Use
    `@pydantic.model_validator(mode='before'|'after')` to validate complex
    inter-dependencies between config fields (e.g., ensuring
    `enrichment_factor` matches `seed_impurity_weights`). Always return `self`
    (or `data`) from these validators.

*   **Module-Level Constants:** Avoid using magic numbers within functions.
    Define thresholds, tolerances, or algorithm parameters as `typing.Final`
    `UPPER_SNAKE_CASE` module-level constants (e.g.,
    `_ROOT_UNIQUENESS_TOL: Final[float] = 1e-2`).

*   **Variable Naming:** Prioritize consistency and clarity in naming. Use
    distinct names for physically distinct quantities rather than overloading
    one name with different definitions. Avoid excessive acronyms or
    contractions, but standard physics abbreviations (e.g., `T_e`, `n_e`,
    `R_major`, `I_p`) are encouraged where appropriate. Capital letters are
    permitted for physics variables, but you must add
    `# pylint: disable=invalid-name` near the top of the module (only when
    needed) to stop linter complaints.
    *   *Note:* Output keys in `output_keys.py` use `UPPER_SNAKE_CASE`
        (e.g., `T_E`, `I_P`), and integrated scalars under `/scalars` use names
        like `P_SOL_total`.

*   **Dtype Conventions & Array Initialization:** Prefer `jnp.asarray` over
    `jnp.array` when wrapping existing data to avoid unnecessary copies. When
    explicitly specifying a precision (e.g., for `jnp.zeros` or `jnp.array`),
    use `jax_utils.get_dtype()`, `jax_utils.get_np_dtype()`, or
    `jax_utils.get_int_dtype()` rather than hardcoding `jnp.float64` or
    `jnp.float32`.

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
    LaTeX backslashes. **Never wrap `$...$` inside backticks** (e.g., writing
    `` `$q$` `` matches the inline-code regex before math substitution in
    `torax_config_autodoc.py`, rendering literal dollar signs without raising a
    warning). Every `$...$` and `$$...$$` expression is translated via
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

## Architecture invariants

*   **Staticity for Shapes:** Any parameter that determines array sizes, shapes,
    or loop counts (e.g., `num_guesses`, `maxiter`, `grid_size`) MUST be
    explicitly marked static:
    *   In the `@jax.jit` decorator (`static_argnames`).
    *   In dataclass fields using `jax.tree.static()`.
    *   In Pydantic configurations using `torax_pydantic.JAX_STATIC`.

*   **Distinct Xarray Dimensions:** Ensure distinct xarray dimension names
    (e.g., `output_keys.IMPURITY` vs
    `extended_lengyel_standalone.SEED_IMPURITY`) when grouping fields in a
    Dataset to prevent unintended shape alignments and `NaN` padding.

## Testing conventions

*   **Test Locality:** Preferably add to existing test modules instead of making
    new test modules.

*   **Test Scope & Fast Execution:** Focus on having more, smaller tests that
    verify one specific thing or behavior, rather than large tests that cover a
    broad theme. Keep slow end-to-end `run_simulation()` tests to ~1-2 smoke
    tests per module; test edge cases and parameter branches via unit tests or
    direct step calls to keep the test suite fast.

*   **Avoid Trivial Tests:** Do not write tests for trivial behaviour such as
    dataclass construction, default field values, or straightforward `Config`
    to `RuntimeParams` passthroughs. Focus unit tests on non-trivial logic:
    physics equations, edge cases, config validation rules, time-dependent
    interpolation, and JAX/JIT recompilation invariants.

*   **Precision in Tests (`jax_enable_x64`):** Never unconditionally call
    `jax.config.update('jax_enable_x64', True)` at test module scope (it breaks
    `JAX_PRECISION=f32` runs). Gate any `float64` requirement on
    `jax_utils.get_dtype() == jnp.float64` or honor the configured dtype.

*   **Assertions & Logic in Tests:** Avoid `if`/`for` logic inside test bodies.
    Split multi-setup `self.subTest` blocks into separate tests or
    `@parameterized.parameters`. Prefer `assertRaisesRegex` over bare
    `assertRaises` to verify the error message, and use `assertIs` for enum
    identity assertions (e.g., `SimError.NO_ERROR`).

*   **Test Function Docstrings:** Prefer making each test name self-explanatory,
    and avoid redundant test function docstrings. If a test needs a long
    docstring, split it into multiple smaller tests.

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

*   **Updating Numerical Test References:** Do not blindly regenerate reference
    `.nc` files when `sim_test` fails. First investigate whether the shift is
    an intentional physics/numerical change or a bug/regression, and only
    update references for intentional model/numerical changes or output schema
    updates. Any PR that updates reference `.nc` files must clearly explain in
    the PR description why the numerical shift is expected and physically
    valid, and attach verification evidence (such as `compare_sim_tests` diff
    summaries and before/after profile plots from `plot_torax`). See
    [contribution_tips.rst](https://torax.readthedocs.io/en/latest/contribution_tips.html)
    (section "Testing") for the regeneration commands.
