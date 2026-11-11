# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `torax_config_autodoc` and whole-codebase config docs."""

import ast
from collections.abc import Iterator, Mapping, Sequence
import enum
import functools
import importlib
import io
import os
import pathlib
import pkgutil
import re
import sys
import tempfile
import textwrap
from typing import Annotated, Any, Literal

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pydantic
from sphinx import application
from torax import torax_pydantic
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic as internal_torax_pydantic

import importlib.util

_AUTODOC_PATH = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent
    / 'sphinxext'
    / 'torax_config_autodoc.py'
)
_spec = importlib.util.spec_from_file_location(
    'torax_config_autodoc', _AUTODOC_PATH
)
assert _spec is not None and _spec.loader is not None
torax_config_autodoc = importlib.util.module_from_spec(_spec)
sys.modules['torax_config_autodoc'] = torax_config_autodoc
_spec.loader.exec_module(torax_config_autodoc)


# Pydantic BaseModel subclasses in `torax` that are internal helpers,
# abstract base classes, or experimental models not meant to be top-level
# entries in `docs/configuration/*.rst`.
_INTERNAL_OR_BASE_CONFIG_CLASSES: frozenset[str] = frozenset({
    'torax.experimental.gas_puff_feedback_source.GasPuffFeedbackSourceConfig',
    'torax._src.geometry.chease.CheaseData',
    'torax._src.geometry.eqdsk.EQDSKData',
    'torax._src.geometry.imas.IMASSliceData',
    'torax._src.geometry.meq.MEQLData',
    'torax._src.geometry.meq.MEQLYBundle',
    'torax._src.geometry.meq.MEQLYSlice',
    'torax._src.geometry.tokamaker.TokaMakerEquilibrium',
    'torax._src.mhd.sawtooth.redistribution_base.RedistributionConfig',
    'torax._src.mhd.sawtooth.trigger_base.TriggerConfig',
    'torax._src.neoclassical.bootstrap_current.base.BootstrapCurrentModelConfig',
    'torax._src.neoclassical.conductivity.base.ConductivityModelConfig',
    'torax._src.neoclassical.config.NeoclassicalConfigBase',
    'torax._src.neoclassical.poloidal_velocity.base.PoloidalVelocityModelConfig',
    'torax._src.pedestal_model.config.PowerScalingFormationConfig',
    'torax._src.sources.ion_cyclotron_source.base.IonCyclotronSourceConfig',
    'torax._src.torax_pydantic.config.ToraxConfig',
    'torax._src.torax_pydantic.config_base.ConfigBase',
    'torax._src.torax_pydantic.interpolated_param_1d.TimeVaryingScalar',
    'torax._src.torax_pydantic.interpolated_param_1d.TimeVaryingScalarStep',
    'torax._src.torax_pydantic.interpolated_param_2d.Grid1D',
    'torax._src.torax_pydantic.interpolated_param_2d.SparseTimeVaryingArray',
    'torax._src.torax_pydantic.interpolated_param_2d.TimeVaryingArray',
    'torax._src.transport_model.qualikiz_based_transport_model.QualikizBasedTransportModelConfig',
    'torax._src.transport_model.tglf_based_transport_model.TGLFBasedTransportModelConfig',
})

# Pydantic BaseModel subclasses documented inline in `docs/configuration/*.rst`
# (e.g. discriminator-only submodels with no user-configurable parameters).
_INLINE_ONLY_MODEL_CLASSES: frozenset[str] = frozenset({
    'torax._src.core_profiles.plasma_composition.ion_mixture.IonMixture',
    'torax._src.edge.config.CombinedEdgeConfig',
    'torax._src.neoclassical.bootstrap_current.redl.RedlModelConfig',
    'torax._src.neoclassical.bootstrap_current.zeros.ZerosModelConfig',
    'torax._src.neoclassical.conductivity.redl.RedlModelConfig',
    'torax._src.neoclassical.poloidal_velocity.zeros.ZerosModelConfig',
    'torax._src.neoclassical.transport.zeros.ZerosModelConfig',
})


def _docs_dir() -> pathlib.Path:
  # Avoid `.resolve()` because test runfiles may symlink individual files,
  # which would lose the directory hierarchy.
  return pathlib.Path(os.path.abspath(__file__)).parent.parent


def _torax_package_dir() -> pathlib.Path:
  torax_config_autodoc._ensure_torax_importable()
  torax_mod = importlib.import_module('torax')
  assert torax_mod.__file__ is not None
  return pathlib.Path(os.path.abspath(torax_mod.__file__)).parent


def _iter_docstrings(tree: ast.AST) -> Iterator[tuple[str, str]]:
  """Yields `(qualname, docstring)` for all docstrings in a Python AST."""
  for node in ast.walk(tree):
    if isinstance(
        node,
        (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
    ):
      doc = ast.get_docstring(node, clean=False)
      if doc:
        yield getattr(node, 'name', '<module>'), doc
      if isinstance(node, ast.ClassDef):
        for stmt in node.body:
          if (
              isinstance(stmt, ast.Expr)
              and isinstance(stmt.value, ast.Constant)
              and isinstance(stmt.value.value, str)
          ):
            yield f'{node.name}.<attribute>', stmt.value.value


def _discover_all_torax_basemodels() -> dict[str, type[pydantic.BaseModel]]:
  """Imports all non-test `torax` modules and returns all `BaseModel` subclasses."""
  pkg_dir = _torax_package_dir()
  for _, modname, _ in pkgutil.walk_packages([str(pkg_dir)], prefix='torax.'):
    if (
        '.tests.' in modname
        or modname.endswith('_test')
        or '.scripts.' in modname
    ):
      continue
    try:
      importlib.import_module(modname)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  discovered: dict[str, type[pydantic.BaseModel]] = {}

  def _walk(cls: type[pydantic.BaseModel]) -> None:
    for sub in cls.__subclasses__():
      mod = sub.__module__
      if (
          mod.startswith('torax')
          and '.tests.' not in mod
          and not mod.endswith('_test')
          and '<locals>' not in sub.__qualname__
      ):
        qual = f'{mod}.{sub.__qualname__}'
        discovered[qual] = sub
      _walk(sub)

  _walk(pydantic.BaseModel)
  return discovered


def _discover_rst_torax_config_classes(docs_dir: pathlib.Path) -> set[str]:
  """Returns dotted BaseModel paths referenced by `.. torax_config::`."""
  referenced: set[str] = set()
  pattern = re.compile(r'^\s*\.\.\s+torax_config::\s+([^\s]+)', re.MULTILINE)
  for rst_path in sorted((docs_dir / 'configuration').glob('*.rst')):
    for match in pattern.finditer(rst_path.read_text(encoding='utf-8')):
      target_path = match.group(1)
      try:
        obj = torax_config_autodoc._import_symbol(target_path)
      except Exception:  # pylint: disable=broad-exception-caught
        referenced.add(target_path)
        continue
      if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
        referenced.add(target_path)
        continue
      union_models = torax_config_autodoc._extract_basemodels(obj)
      if torax_config_autodoc._collect_discriminator_literals(union_models):
        continue
      referenced.add(target_path)
  return referenced


def _build_sphinx_with_warning_capture(
    *,
    srcdir: pathlib.Path,
    confdir: pathlib.Path,
    outdir: pathlib.Path,
    doctreedir: pathlib.Path,
) -> tuple[int, str]:
  """Runs an in-process Sphinx HTML build with `warningiserror=True`."""
  status_buf = io.StringIO()
  warn_buf = io.StringIO()
  app = application.Sphinx(
      srcdir=str(srcdir),
      confdir=str(confdir),
      outdir=str(outdir),
      doctreedir=str(doctreedir),
      buildername='html',
      status=status_buf,
      warning=warn_buf,
      freshenv=True,
      warningiserror=True,
  )
  app.build(force_all=True)
  return app.statuscode, warn_buf.getvalue()


@absltest.skipIf(
    'pytest' in sys.modules,
    'Skipped in shared pytest shards due to optional-dependency import order.',
)
class ToraxConfigAutodocTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_var', 'T_e', 'T_e'),
      ('greek_symbol', 'chi', r'\chi'),
      ('hat_identifier', 'rho_norm', r'\hat{\rho}'),
      ('subscript_text', 'P_SOL', r'P_{\text{SOL}}'),
      ('primed_symbol', "beta_pol'", r"\beta_{\text{pol}}'"),
      ('inequality', 'P_SOL <= 0', r'P_{\text{SOL}} \le 0'),
      ('leading_compare', '>= 2', r'\ge 2'),
      ('prefix_propto', 'propto chi', r'\propto \chi'),
      (
          'propto_power_law',
          'chi propto (R_over_L_Ti - R_over_L_Ti_crit)^alpha',
          r'\chi \propto (R/L_{T_i} - R/L_{T_i,\text{crit}})^{\alpha}',
      ),
      (
          'inline_times',
          'n_e(rho_norm_ref, t) times multiplier',
          r'n_e(\hat{\rho}_{\text{ref}}, t) \times \text{multiplier}',
      ),
      (
          'partial_derivative',
          '-partial(beta_pol_local) / partial(psi_N)',
          r'-\partial \beta_{\text{pol,local}} / \partial \psi_N',
      ),
      ('standalone_star', '*', r'\times'),
      (
          'half_open_interval',
          '(rho_min, rho_max]',
          r'(\rho_{\text{min}}, \rho_{\text{max}}]',
      ),
      (
          'multiline_whitespace',
          'dt_base =\n  dx^2 / (2 * chi)',
          r'dt_{\text{base}} = \frac{dx^2}{2\chi}',
      ),
      (
          'ec_zeta_with_units',
          (
              'zeta = (e^2 * ln(Lambda_ee) / (8 * epsilon_0^2)) * (1 / avg(1 /'
              " R)) * (n_e['m^-3'] / T_e['eV']) * (I_ec['A'] / P_ec['W'])"
          ),
          (
              r'\zeta = \frac{e^2 \ln \Lambda_{ee}}{8\varepsilon_0^2} '
              r'\left\langle\frac{1}{R}\right\rangle^{-1} '
              r'\frac{n_e\;[\mathrm{m^{-3}}]}{T_e\;[\mathrm{eV}]} '
              r'\frac{I_{\text{ec}}\;[\mathrm{A}]}{P_{\text{ec}}\;[\mathrm{W}]}'
          ),
      ),
  )
  def test_code_to_latex_known_expressions(self, expr: str, expected: str):
    self.assertEqual(torax_config_autodoc._code_to_latex(expr), expected)

  @parameterized.named_parameters(
      ('empty', ''),
      ('whitespace_only', '   \n\t  '),
      ('raw_latex_symbol', r'\alpha'),
      ('raw_latex_frac', r'\frac{a}{b}'),
      ('unmapped_expression', 'unknown_var + 1'),
      ('operator_collision', 'T sim'),
  )
  def test_code_to_latex_fails_loudly_on_unknown_or_empty_math(self, expr: str):
    with self.assertRaises(ValueError):
      torax_config_autodoc._code_to_latex(expr)

  def test_code_to_latex_table_integrity(self):
    for key, val in torax_config_autodoc._CODE_TO_LATEX.items():
      self.assertTrue(key)
      self.assertEqual(re.sub(r'\s+', ' ', key.strip()), key)
      self.assertTrue(val)
      self.assertEqual(
          val.count('{'),
          val.count('}'),
          msg=f'Unbalanced braces in _CODE_TO_LATEX[{key!r}]: {val!r}',
      )
      self.assertIsNone(
          re.search(r'(?<!\\)\$', val),
          msg=f'Unescaped `$` in math_substitutions.toml[{key!r}]: {val!r}',
      )

  def test_load_code_to_latex_normalizes_and_validates_toml(self):
    with tempfile.TemporaryDirectory() as tmp:
      toml_path = pathlib.Path(tmp) / 'subs.toml'
      toml_path.write_text(
          textwrap.dedent("""\
              'rho_norm' = '\\hat{\\rho}'
              "beta_pol'" = '''
                \\beta_{\\text{pol}}'
              '''
          """),
          encoding='utf-8',
      )
      loaded = torax_config_autodoc._load_code_to_latex(toml_path)
      self.assertEqual(
          loaded,
          {
              'rho_norm': r'\hat{\rho}',
              "beta_pol'": r"\beta_{\text{pol}}'",
          },
      )

      toml_path.write_text("'a  b' = 'x'\n'a b' = 'y'\n", encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'Duplicate normalized'):
        torax_config_autodoc._load_code_to_latex(toml_path)

      toml_path.write_text("'a' = '   '\n", encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'Empty math substitution'):
        torax_config_autodoc._load_code_to_latex(toml_path)

  @parameterized.named_parameters(
      ('unknown_math_in_docstring', 'Temperature $unknown_var + 1$ in keV.'),
      ('raw_latex_in_math', r'Temperature $\alpha$ in keV.'),
      ('unclosed_dollar', 'Temperature $T_e + 1 in keV.'),
      ('spaced_inline_dollar', 'Temperature $ T_e + 1 $ in keV.'),
  )
  def test_parse_docstring_strict_fails_on_invalid_math(self, docstring: str):
    with self.assertRaises(ValueError):
      torax_config_autodoc._parse_docstring(docstring, strict=True)

  def test_parse_docstring_allows_escaped_dollar_and_fenced_dollar(self):
    docstring = textwrap.dedent("""\
        Costs \\$100 per run.

        ```bash
        echo $HOME
        ```
    """)
    summary, _ = torax_config_autodoc._parse_docstring(docstring, strict=True)
    self.assertIn('Costs $100 per run.', summary)
    self.assertIn('.. code-block:: bash', summary)
    self.assertIn('echo $HOME', summary)

  def test_sphinx_build_fails_loudly_on_invalid_math_in_docstring(self):
    class BadMathConfig(pydantic.BaseModel):
      """Config with broken math in docstring.

      Attributes:
        x: Value with invalid math $T sim$.
      """

      x: float = 1.0

    mod_name = '_torax_test_bad_math_mod'
    fake_mod = type(sys)(mod_name)
    setattr(fake_mod, 'BadMathConfig', BadMathConfig)
    sys.modules[mod_name] = fake_mod
    try:
      with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        srcdir = tmp_path / 'src'
        outdir = tmp_path / 'out'
        doctreedir = tmp_path / 'doctrees'
        srcdir.mkdir()
        (srcdir / 'conf.py').write_text(
            textwrap.dedent("""\
                extensions = ['torax_config_autodoc']
                main_doc = 'index'
            """),
            encoding='utf-8',
        )
        (srcdir / 'index.rst').write_text(
            textwrap.dedent("""\
                Test
                ====

                .. torax_config:: _torax_test_bad_math_mod.BadMathConfig
            """),
            encoding='utf-8',
        )
        statuscode, warnings = _build_sphinx_with_warning_capture(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=outdir,
            doctreedir=doctreedir,
        )
        self.assertNotEqual(statuscode, 0)
        self.assertIn('BadMathConfig.x', warnings)
        self.assertIn('T sim', warnings)
    finally:
      sys.modules.pop(mod_name, None)

  def test_format_type_fallbacks(self):
    def _str_or_dict_validator(val: Any) -> Any:
      if isinstance(val, str):
        return {val: 1.0}
      return val

    def _typed_str_validator(val: str | dict[str, int]) -> dict[str, int]:
      return {val: 1} if isinstance(val, str) else val

    def _non_str_validator(val: np.ndarray | tuple[str, list[Any]]) -> Any:
      return val

    class _SampleMode(enum.StrEnum):
      LINEAR = 'linear'
      STEP = 'step'

    class GeometryType(enum.IntEnum):
      CIRCULAR = 0
      CHEASE = 1

    class _NestedModel(pydantic.BaseModel):
      x: int = 1

    cases: list[tuple[Any, Sequence[Any], str]] = [
        (tuple[float, float], (), 'tuple[float, float]'),
        (tuple[()], (), 'tuple[()]'),
        (tuple, (), 'tuple'),
        (list[str], (), 'list[str]'),
        (list, (), 'list'),
        (Sequence[int], (), 'Sequence[int]'),
        (Sequence, (), 'Sequence'),
        (set[str], (), 'set[str]'),
        (frozenset[int], (), 'frozenset[int]'),
        (set, (), 'set'),
        (Mapping[str, float], (), 'Mapping[str, float]'),
        (
            dict[str, str | float | int | bool | None],
            (),
            'dict[str, str | float | int | bool | None]',
        ),
        (Any, (), 'Any'),
        (torax_pydantic.TimeVaryingScalar, (), 'TimeVaryingScalar'),
        (torax_pydantic.TimeVaryingScalarStep, (), 'TimeVaryingScalarStep'),
        (
            torax_pydantic.PositiveTimeVaryingScalar,
            (),
            'PositiveTimeVaryingScalar',
        ),
        (
            torax_pydantic.NonNegativeTimeVaryingScalar,
            (),
            'NonNegativeTimeVaryingScalar',
        ),
        (
            torax_pydantic.NonNegativeTimeVaryingScalarStep,
            (),
            'NonNegativeTimeVaryingScalarStep',
        ),
        (
            torax_pydantic.UnitIntervalTimeVaryingScalar,
            (),
            'UnitIntervalTimeVaryingScalar',
        ),
        (torax_pydantic.TimeVaryingArray, (), 'TimeVaryingArray'),
        (
            torax_pydantic.PositiveTimeVaryingArray,
            (),
            'PositiveTimeVaryingArray',
        ),
        (
            torax_pydantic.NonNegativeTimeVaryingArray,
            (),
            'NonNegativeTimeVaryingArray',
        ),
        (
            interpolated_param_2d.SparseTimeVaryingArray,
            (),
            'SparseTimeVaryingArray',
        ),
        (torax_pydantic.NumpyArray, (), 'NumpyArray'),
        (torax_pydantic.NumpyArray1D, (), 'NumpyArray1D'),
        (torax_pydantic.NumpyArray1DSorted, (), 'NumpyArray1DSorted'),
        (pydantic.PositiveFloat, (), 'PositiveFloat'),
        (
            Annotated[pydantic.NonNegativeInt, torax_pydantic.JAX_STATIC],
            (),
            'NonNegativeInt',
        ),
        (pydantic.FilePath, (), 'FilePath'),
        (internal_torax_pydantic.UnitInterval, (), 'float'),
        (internal_torax_pydantic.OpenUnitInterval, (), 'float'),
        (internal_torax_pydantic.Second, (), 'float'),
        (internal_torax_pydantic.Meter, (), 'PositiveFloat'),
        (internal_torax_pydantic.COCOSInt, (), 'int'),
        (
            Annotated[
                torax_pydantic.TimeVaryingScalar,
                torax_pydantic.scalar_bounds_validator(gt=1.0),
            ],
            (),
            'TimeVaryingScalar',
        ),
        (_SampleMode, (), "['linear', 'step']"),
        (GeometryType, (), "['circular', 'chease']"),
        (Literal['a', 'b'], (), "['a', 'b']"),
        (_NestedModel, (), 'dict'),
        (_NestedModel | None, (), 'dict | None'),
        (Sequence[_NestedModel], (), 'Sequence[dict]'),
        (os.PathLike, (), 'PathLike'),
        (pathlib.Path, (), 'PathLike'),
        (
            Annotated[
                dict[str, float],
                pydantic.BeforeValidator(
                    _str_or_dict_validator,
                    json_schema_input_type=str | dict[str, float],
                ),
            ],
            (),
            'dict[str, float] | str',
        ),
        (
            Annotated[
                dict[str, int],
                pydantic.BeforeValidator(_typed_str_validator),
            ],
            (),
            'dict[str, int] | str',
        ),
        (
            Annotated[
                np.ndarray,
                pydantic.BeforeValidator(_non_str_validator),
            ],
            (),
            'array',
        ),
    ]
    for ann, meta, expected in cases:
      with self.subTest(expected=expected):
        self.assertEqual(torax_config_autodoc._format_type(ann, meta), expected)

  def test_format_default_fallbacks(self):
    class SubConfig(pydantic.BaseModel):
      a: int = 1

    class DiscSubConfig(pydantic.BaseModel):
      model_name: Literal['zeros'] = 'zeros'
      x: float = 1.0

    class GeometryType(enum.IntEnum):
      CIRCULAR = 0

    def _make_list() -> list[int]:
      raise TypeError('requires context')

    def _make_set() -> set[str]:
      raise TypeError('requires context')

    sparse_multi = interpolated_param_2d.SparseTimeVaryingArray.model_validate(
        {0.0: {(0.2, 0.5): 1.5}}
    )
    sparse_single = interpolated_param_2d.SparseTimeVaryingArray.model_validate(
        2.5
    )

    class Holder(pydantic.BaseModel):
      sparse_multi_field: Any = sparse_multi
      sparse_single_field: Any = sparse_single
      nested_list_field: list[SubConfig] = [SubConfig()]
      nested_dict_field: dict[str, SubConfig] = {'k': SubConfig()}
      disc_model_field: DiscSubConfig = DiscSubConfig()
      ndarray_field: Any = np.array([1.0, 2.0])
      set_field: set[str] = {'alpha'}
      sci_float_field: float = 8e-5
      geom_enum_field: GeometryType = GeometryType.CIRCULAR
      partial_model_factory: Any = pydantic.Field(
          default_factory=functools.partial(SubConfig, a=2)
      )
      typed_list_factory: Any = pydantic.Field(default_factory=_make_list)
      typed_set_factory: Any = pydantic.Field(default_factory=_make_set)

    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['sparse_multi_field']
        ),
        '{0.0: {(0.2, 0.5): 1.5}}',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['sparse_single_field']
        ),
        '2.5',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['nested_list_field']
        ),
        '[{}]',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['nested_dict_field']
        ),
        "{'k': {}}",
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['disc_model_field']
        ),
        "{'model_name': 'zeros'}",
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['ndarray_field']
        ),
        'np.array([1.0, 2.0])',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(Holder.model_fields['set_field']),
        "{'alpha'}",
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['sci_float_field']
        ),
        '8e-05',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['geom_enum_field']
        ),
        "'circular'",
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['partial_model_factory']
        ),
        "{'a': 2}",
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['typed_list_factory']
        ),
        '[]',
    )
    self.assertEqual(
        torax_config_autodoc._format_default(
            Holder.model_fields['typed_set_factory']
        ),
        'set()',
    )

  def test_all_codebase_docstrings_have_valid_math_and_fences(self):
    pkg_dir = _torax_package_dir()
    py_files = [
        p for p in sorted(pkg_dir.rglob('*.py')) if 'tests' not in p.parts
    ]
    self.assertGreater(len(py_files), 200)
    errors: list[str] = []
    used_exprs: set[str] = set()
    for py_path in py_files:
      source = py_path.read_text(encoding='utf-8')
      tree = ast.parse(source, filename=str(py_path))
      rel = py_path.relative_to(pkg_dir)
      for qual, doc in _iter_docstrings(tree):
        rst_lines = torax_config_autodoc._convert_fenced_code_blocks(
            doc.splitlines()
        )
        try:
          torax_config_autodoc._normalize_inline_code(
              rst_lines,
              context=f'{rel}:{qual}',
              strict=True,
          )
        except ValueError as exc:
          errors.append(f'{rel}:{qual}: {exc}')

        def _collect_math(para: str) -> str:
          for m in torax_config_autodoc._INLINE_MARKUP_RE.finditer(para):
            raw = m.group(2) if m.group(2) is not None else m.group(3)
            if raw is not None:
              used_exprs.add(re.sub(r'\s+', ' ', raw.strip()))
          return para

        torax_config_autodoc._transform_prose_paragraphs(
            rst_lines, _collect_math
        )
    self.assertEmpty(
        errors,
        msg='Found invalid math or unmatched `$` in docstrings:\n'
        + '\n'.join(errors),
    )
    unused_entries = sorted(
        set(torax_config_autodoc._CODE_TO_LATEX) - used_exprs
    )
    self.assertEmpty(
        unused_entries,
        msg=(
            'Unused entries in `math_substitutions.toml` (remove stale'
            ' mappings):\n'
            + '\n'.join(f'  - {entry!r}' for entry in unused_entries)
        ),
    )

  def test_all_pydantic_models_covered_by_docs(self):
    docs_dir = _docs_dir()
    all_models = _discover_all_torax_basemodels()
    rst_referenced = _discover_rst_torax_config_classes(docs_dir)

    allowlisted = (
        _INTERNAL_OR_BASE_CONFIG_CLASSES | _INLINE_ONLY_MODEL_CLASSES
    )

    uncovered = sorted(set(all_models) - rst_referenced - allowlisted)
    self.assertEmpty(
        uncovered,
        msg=(
            'Pydantic BaseModel classes missing from `.. torax_config::` in '
            '`docs/configuration/*.rst` (add `.. torax_config::` or, if '
            'internal/inline-only, add to the allowlist in '
            '`docs/tests/torax_config_autodoc_test.py`):\n'
            + '\n'.join(f'  - {name}' for name in uncovered)
        ),
    )

    stale_allowlist = sorted(rst_referenced & allowlisted)
    self.assertEmpty(
        stale_allowlist,
        msg=(
            'Classes already documented via `.. torax_config::` should be '
            'removed from the allowlist in `torax_config_autodoc_test.py`:\n'
            + '\n'.join(f'  - {name}' for name in stale_allowlist)
        ),
    )

    missing_models = sorted(rst_referenced - set(all_models))
    self.assertEmpty(
        missing_models,
        msg=(
            'Documented config classes could not be imported or are not '
            'Pydantic BaseModel subclasses:\n'
            + '\n'.join(f'  - {name}' for name in missing_models)
        ),
    )

  def test_all_documented_configs_have_complete_attributes_and_clean_types(
      self,
  ):
    docs_dir = _docs_dir()
    all_models = _discover_all_torax_basemodels()
    target_names = sorted(_discover_rst_torax_config_classes(docs_dir))

    issues: list[str] = []
    for class_path in target_names:
      cls = all_models[class_path]
      attr_docs, own_documented = torax_config_autodoc._collect_mro_attributes(
          cls, strict=True
      )
      for field_name in cls.model_fields:
        if not attr_docs.get(field_name):
          issues.append(
              f'{class_path}: missing `Attributes:` entry for {field_name!r}'
          )
      class_vars = set(getattr(cls, '__class_vars__', ()))
      for doc_name in sorted(own_documented):
        if doc_name not in cls.model_fields and doc_name not in class_vars:
          issues.append(
              f'{class_path}: `Attributes:` documents unknown field'
              f' {doc_name!r}'
          )
      for field_name, field_info in cls.model_fields.items():
        type_str = torax_config_autodoc._format_type(
            field_info.annotation, field_info.metadata
        )
        default_str = torax_config_autodoc._format_default(field_info)
        for bad_tok in (
            'torax.',
            'typing.',
            'Annotated',
            'Union',
            'Optional',
            'Literal',
            'Any',
            'Callable',
        ):
          if bad_tok in type_str:
            issues.append(
                f'{class_path}.{field_name}: raw type fallback {type_str!r}'
            )
        if default_str is not None and (
            '<' in default_str or 'object at' in default_str
        ):
          issues.append(
              f'{class_path}.{field_name}: raw default fallback {default_str!r}'
          )

    self.assertEmpty(
        issues,
        msg='Config docstring / type formatting issues:\n' + '\n'.join(issues),
    )

  def test_central_links_bibliography_used_across_docs_and_configs(self):
    docs_dir = _docs_dir()
    links_path = docs_dir / 'links.rst'
    links_text = links_path.read_text(encoding='utf-8')
    defined_targets = set(
        re.findall(r'^\.\.\s+_([^\s:]+):\s+https?://', links_text, re.MULTILINE)
    )
    referenced_targets = set(
        re.findall(
            r'^\.\.\s+\|[^|]+\|\s+replace::\s+`[^`]*<([^`>]+)_>`__',
            links_text,
            re.MULTILINE,
        )
    )
    self.assertEqual(defined_targets, referenced_targets)

    rst_issues: list[str] = []
    for rst_path in sorted(docs_dir.rglob('*.rst')):
      if any(
          part in ('_build', 'build', '_autosummary') for part in rst_path.parts
      ):
        continue
      rel = rst_path.relative_to(docs_dir)
      text = rst_path.read_text(encoding='utf-8')
      if re.search(r'^\s*\.\.\s+include::\s+.*links\.rst\b', text, re.M):
        rst_issues.append(
            f'{rel}: manual `.. include:: ...links.rst` (loaded automatically'
            ' via `rst_epilog` in `conf.py`)'
        )
      if rst_path == links_path:
        continue
      for kind, lines in torax_config_autodoc._iter_rst_blocks(
          text.splitlines()
      ):
        if kind != 'prose':
          continue
        block = '\n'.join(lines)
        for url_match in re.finditer(r'https?://[^\s>`)]+', block):
          rst_issues.append(
              f'{rel}: hardcoded external URL {url_match.group(0)!r} (move to'
              ' `docs/links.rst`)'
          )
    self.assertEmpty(
        rst_issues,
        msg='External link / include issues in RST docs:\n'
        + '\n'.join(rst_issues),
    )

    docstring_issues: list[str] = []
    for class_path, cls in sorted(_discover_all_torax_basemodels().items()):
      doc = cls.__doc__ or ''
      if re.search(r'<https?://', doc):
        docstring_issues.append(
            f'{class_path}: inline `<http...>` link in docstring (use a'
            ' `|...|` substitution from `docs/links.rst`)'
        )
      if re.search(r'^\s*References:\s*$', doc, re.MULTILINE):
        docstring_issues.append(
            f'{class_path}: per-class `References:` section in docstring (use'
            ' `|...|` substitutions from `docs/links.rst`)'
        )
    self.assertEmpty(
        docstring_issues,
        msg='BaseModel docstring link issues:\n' + '\n'.join(docstring_issues),
    )

  def test_full_sphinx_build_succeeds_without_warnings(self):
    docs_dir = _docs_dir()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = pathlib.Path(tmp)
      confdir = tmp_path / 'conf'
      outdir = tmp_path / 'html'
      doctreedir = tmp_path / 'doctrees'
      confdir.mkdir()

      # Load project `conf.py` settings (`rst_epilog`, `suppress_warnings`,
      # etc.) while stubbing optional third-party UI/plot extensions and
      # disabling network intersphinx fetches for hermetic test execution.
      (confdir / 'conf.py').write_text(
          textwrap.dedent("""\
              import os
              from docutils import nodes
              from docutils.parsers import rst

              _ORIG_DOCS_DIR = {docs_dir!r}

              with open(
                  os.path.join(_ORIG_DOCS_DIR, 'conf.py'), encoding='utf-8'
              ) as _f:
                exec(compile(_f.read(), 'conf.py', 'exec'), globals())

              extensions = [
                  'sphinx.ext.autosectionlabel',
                  'sphinx.ext.mathjax',
                  'torax_config_autodoc',
              ]
              html_theme = 'basic'
              html_theme_options = {{}}
              html_static_path = []
              html_logo = None
              html_favicon = None


              class _StubDirective(rst.Directive):
                has_content = True
                optional_arguments = 10
                option_spec = {{
                    'name': rst.directives.unchanged,
                    'class': rst.directives.unchanged,
                    'class-card': rst.directives.unchanged,
                    'class-container': rst.directives.unchanged,
                    'class-title': rst.directives.unchanged,
                    'class-body': rst.directives.unchanged,
                    'color': rst.directives.unchanged,
                    'icon': rst.directives.unchanged,
                    'link': rst.directives.unchanged,
                    'link-type': rst.directives.unchanged,
                    'margin': rst.directives.unchanged,
                    'padding': rst.directives.unchanged,
                    'gutter': rst.directives.unchanged,
                    'columns': rst.directives.unchanged,
                    'shadow': rst.directives.unchanged,
                    'open': rst.directives.flag,
                    'selected': rst.directives.flag,
                    'sync': rst.directives.unchanged,
                    'context': rst.directives.flag,
                    'include-source': rst.directives.flag,
                    'format': rst.directives.unchanged,
                    'caption': rst.directives.unchanged,
                }}

                def run(self):
                  if self.arguments:
                    title_node = nodes.rubric(text=' '.join(self.arguments))
                    self.add_name(title_node)
                    node = nodes.container()
                    if self.content:
                      self.state.nested_parse(
                          self.content, self.content_offset, node
                      )
                    return [title_node, node]
                  node = nodes.container()
                  self.add_name(node)
                  if self.content:
                    self.state.nested_parse(
                        self.content, self.content_offset, node
                    )
                  return [node]


              def _noop_role(
                  name, rawtext, text, lineno, inliner, options=None, content=()
              ):
                del name, rawtext, lineno, inliner, options, content
                return [nodes.inline(text=text)], []


              def setup(app):
                for directive_name in (
                    'grid',
                    'grid-item-card',
                    'tab-set',
                    'tab-item',
                    'dropdown',
                    'card',
                    'button-ref',
                    'button-link',
                    'plot',
                ):
                  app.add_directive(directive_name, _StubDirective)
                for role_name in ('octicon', 'fas', 'fab', 'bdg-primary'):
                  app.add_role(role_name, _noop_role)
          """).format(
              docs_dir=str(docs_dir),
          ),
          encoding='utf-8',
      )

      statuscode, warnings = _build_sphinx_with_warning_capture(
          srcdir=docs_dir,
          confdir=confdir,
          outdir=outdir,
          doctreedir=doctreedir,
      )
      self.assertEqual(
          statuscode,
          0,
          msg=f'Sphinx build failed with status {statuscode}:\n{warnings}',
      )
      self.assertEqual(
          warnings,
          '',
          msg=f'Sphinx build emitted warnings:\n{warnings}',
      )
      mhd_html = (outdir / 'configuration' / 'mhd.html').read_text(
          encoding='utf-8'
      )
      self.assertIn('torax-config-dict-section', mhd_html)
      self.assertIn('torax-config-nested-row', mhd_html)
      solver_html = (outdir / 'configuration' / 'solver.html').read_text(
          encoding='utf-8'
      )
      self.assertIn('torax-config-variant-section', solver_html)
      self.assertIn('torax-variant-tab-set', solver_html)
      self.assertIn("['linear', 'newton_raphson', 'optimizer']", solver_html)
      pedestal_html = (outdir / 'configuration' / 'pedestal.html').read_text(
          encoding='utf-8'
      )
      self.assertIn('torax-goto-link', pedestal_html)
      self.assertNotIn('(see below)', pedestal_html)
      self.assertIn('[Snyder et al, Nucl. Fusion 2004]', pedestal_html)
      self.assertIn('https://github.com/aaronkho/epednn_mit', pedestal_html)
      transport_html = (outdir / 'configuration' / 'transport.html').read_text(
          encoding='utf-8'
      )
      self.assertIn(
          'Transport model identifier selecting the component implementation.',
          transport_html,
      )
      self.assertIn('torax-nested-dict-heading', transport_html)
      neoclassical_html = (
          outdir / 'configuration' / 'neoclassical.html'
      ).read_text(encoding='utf-8')
      self.assertIn(
          'Sets neoclassical transport coefficients to zero.',
          neoclassical_html,
      )
      profile_html = (
          outdir / 'configuration' / 'profile_conditions.html'
      ).read_text(encoding='utf-8')
      self.assertIn(
          'Disables internal boundary conditions (default).',
          profile_html,
      )
      geometry_html = (outdir / 'configuration' / 'geometry.html').read_text(
          encoding='utf-8'
      )
      self.assertIn('torax-variant-tab-set', geometry_html)
      self.assertIn('Time-dependent geometry', geometry_html)
      self.assertIn(
          'Constructs geometry from in-memory',
          geometry_html,
      )
      self.assertIn(
          'https://openfusiontoolkit.github.io/OpenFUSIONToolkit/doc_tokamaker_main.html',
          geometry_html,
      )
      tsc_html = (
          outdir / 'configuration' / 'time_step_calculator.html'
      ).read_text(encoding='utf-8')
      self.assertIn(
          'Adaptive time step calculator scaled to parabolic',
          tsc_html,
      )
      sources_html = (outdir / 'configuration' / 'sources.html').read_text(
          encoding='utf-8'
      )
      self.assertIn("['mavrin_fit', 'P_in_scaled_flat_profile']", sources_html)
      self.assertIn("['gaussian', 'feedback']", sources_html)
      self.assertIn(
          "['exponential', 'gaussian', 'prescribed']", sources_html
      )

  def test_time_varying_meta_tokens_linked_in_sphinx_html(self):
    class LinkTestConfig(pydantic.BaseModel):
      """Test config with time-varying types.

      Attributes:
        s: Scalar field.
        a: Array field.
      """

      s: torax_pydantic.PositiveTimeVaryingScalar | None = None
      a: torax_pydantic.TimeVaryingArray | None = None

    mod_name = '_torax_test_link_meta_mod'
    fake_mod = type(sys)(mod_name)
    setattr(fake_mod, 'LinkTestConfig', LinkTestConfig)
    sys.modules[mod_name] = fake_mod
    try:
      with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        srcdir = tmp_path / 'src'
        outdir = tmp_path / 'out'
        doctreedir = tmp_path / 'doctrees'
        (srcdir / 'configuration').mkdir(parents=True)
        (srcdir / 'conf.py').write_text(
            textwrap.dedent("""\
                extensions = ['torax_config_autodoc']
                main_doc = 'index'
                suppress_warnings = [
                    'app.add_node',
                    'app.add_directive',
                    'app.add_role',
                ]
            """),
            encoding='utf-8',
        )
        (srcdir / 'index.rst').write_text(
            textwrap.dedent("""\
                Index
                =====

                .. toctree::

                   configuration
                   configuration/numerics
            """),
            encoding='utf-8',
        )
        (srcdir / 'configuration.rst').write_text(
            textwrap.dedent("""\
                Configuration
                =============

                .. _time_varying_scalars:

                Time-varying scalars
                --------------------

                .. _time_varying_arrays:

                Time-varying arrays
                -------------------
            """),
            encoding='utf-8',
        )
        (srcdir / 'configuration' / 'numerics.rst').write_text(
            textwrap.dedent("""\
                numerics
                ########

                .. torax_config:: _torax_test_link_meta_mod.LinkTestConfig
            """),
            encoding='utf-8',
        )
        statuscode, warnings = _build_sphinx_with_warning_capture(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=outdir,
            doctreedir=doctreedir,
        )
        self.assertEqual(statuscode, 0, msg=warnings)
        html = (outdir / 'configuration' / 'numerics.html').read_text(
            encoding='utf-8'
        )
        self.assertIn(
            'href="../configuration.html#time-varying-scalars"'
            '>PositiveTimeVaryingScalar</a>',
            html,
        )
        self.assertIn(
            'href="../configuration.html#time-varying-arrays"'
            '>TimeVaryingArray</a>',
            html,
        )
    finally:
      sys.modules.pop(mod_name, None)

  @parameterized.named_parameters(
      (
          'top_level_dict_key',
          ('mhd', 'sawtooth'),
          ['mhd', 'sawtooth'],
          None,
      ),
      (
          'nested_dict_key',
          ('mhd', 'sawtooth', 'trigger_model'),
          ['mhd', 'sawtooth', 'trigger_model'],
          None,
      ),
      (
          'discriminated_submodel_variant',
          ('pedestal', 'formation_model', 'martin_scaling'),
          ['pedestal', 'formation_model'],
          ('model_name', 'martin_scaling'),
      ),
      (
          'solver_discriminator',
          ('solver', 'newton_raphson'),
          ['solver'],
          ('solver_type', 'newton_raphson'),
      ),
      (
          'geometry_discriminator',
          ('geometry', 'chease'),
          ['geometry'],
          ('model_name', 'chease'),
      ),
      (
          'dict_container_variant',
          ('transport', 'qlknn'),
          ['transport', 'core_transport_models', '<name>'],
          ('model_name', 'qlknn'),
      ),
  )
  def test_resolve_section_schema(
      self,
      path_parts: Sequence[str],
      expected_keys: list[str],
      expected_discriminator: tuple[str, str] | None,
  ):
    dict_keys, discriminator = torax_config_autodoc._resolve_section_schema(
        path_parts
    )
    self.assertEqual(dict_keys, expected_keys)
    self.assertEqual(discriminator, expected_discriminator)


if __name__ == '__main__':
  absltest.main()
