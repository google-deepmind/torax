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

"""Sphinx extension for autodocumenting TORAX Pydantic configurations.

This extension generates TORAX's `docs/configuration/*` reference pages directly
from `pydantic.BaseModel` schemas and Google-style class docstrings, while also
adding markup post-processing.

Main components:

* Environment bootstrapping
  - Ensures `torax` and optional external packages (such as `qualikiz_tools`)
    are importable.
  - Lazily caches TORAX interpolated parameter types, public type aliases, and
    base array classes.

* Markup normalization & code-math translation
  - Converts triple-backtick blocks into Sphinx `.. code-block::` directives.
  - Converts standalone single-backtick spans into RST double-backtick literals.
  - Translates 'Python-like' math inside `$...$` and `$$...$$` into `:math:`
    and `.. math::` blocks via the explicit substitution table loaded from
    `docs/sphinxext/math_substitutions.toml`. Raises warnings on unmapped
    expressions or unmatched `$` delimiters.

* Extracting docs from Google-style docstrings
  - Splits class docstrings into a summary and `Attributes:` field map, merging
    inherited `Attributes:` across the `BaseModel` MRO so subclasses inherit
    base-class field docs while allowing overrides.

* Type & default-value formatting
  - Formats Python/Pydantic annotations into concise config types.
  - Unboxes default wrapper objects and `default_factory` values into Python
    config literals, and marks fields without defaults as `required`.

* `.. torax_config::`
  - Renders a `BaseModel` class in the docs.

* Configuration hierarchy & variant-tab post-processing
  - Resolves `configuration/*` section headings against `ToraxConfig` into
    dictionary key paths (such as `["pedestal"]["structure_model"]`) or
    discriminator badges (such as `model_name = 'qlknn'`).
  - Adds links to TORAX-specific types (such as `TimeVaryingArray`).
  - Folds model variants into tabs.
"""

import ast
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import enum
import functools
import importlib
import importlib.machinery
import importlib.util
import inspect
import os
import pathlib
import re
import sys
import tomllib
import types
import typing
from typing import Annotated, Any, Literal, Tuple, TypeAliasType, Union
from unittest import mock

from docutils import nodes
from docutils import statemachine
from docutils.parsers.rst import directives
import numpy as np
import pydantic
import pydantic.fields
import pydantic.types
from sphinx import application
from sphinx.util import docutils as sphinx_docutils
from sphinx.util import logging

logger = logging.getLogger(__name__)

_ATTRIBUTE_HEADER_RE = re.compile(
    r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:\s*(.*)$'
)
_SECTION_HEADER_RE = re.compile(
    r'^(Args|Arguments|Attributes|Returns|Yields|Raises|Note|Notes|'
    r'Example|Examples|References):\s*$'
)
# Matches inline markup so standalone single-backtick code spans (`foo`) are
# converted to ``foo`` and dollar-delimited LaTeX ($...$ / $$...$$) is
# converted to RST math while preserving existing double backticks, Sphinx roles
# (:ref:`...`, :math:`...`), and RST hyperlinks (`text <url>`_).
_INLINE_MARKUP_RE = re.compile(
    r'``(?:[^`]|`(?!`))+``'
    r'|:[a-zA-Z0-9_+:-]+:`[^`]+`'
    r'|`[^`]+`__?'
    r'|`([^`]+)`(?!_)'
    r'|(?<!\\)\$\$(.+?)(?<!\\)\$\$'
    r'|(?<![\\\w$])\$(?!\s)([^$\n]+?(?:\n[^$\n]+?)*?)(?<!\s)\$(?![\d$])',
    re.DOTALL,
)
_ENUM_LIST_ITEM_RE = re.compile(r'^(\d+)\.\s+')
_FENCE_OPEN_RE = re.compile(r'^([ \t]*)```([a-zA-Z0-9_.+-]*)[ \t]*$')
_FENCE_CLOSE_RE = re.compile(r'^[ \t]*```[ \t]*$')

_MATH_SUBSTITUTIONS_PATH = pathlib.Path(__file__).with_name(
    'math_substitutions.toml'
)


def _load_code_to_latex(path: pathlib.Path) -> Mapping[str, str]:
  """Loads normalized code-math to LaTeX substitutions from a TOML file."""
  with path.open('rb') as f:
    raw = tomllib.load(f)
  mapping: dict[str, str] = {}
  for raw_key, raw_val in raw.items():
    if not isinstance(raw_val, str):
      raise ValueError(
          f'Expected string LaTeX value for {raw_key!r} in {path}, got '
          f'{type(raw_val).__name__}.'
      )
    key = re.sub(r'\s+', ' ', raw_key.strip())
    val = re.sub(r'\s+', ' ', raw_val.strip())
    if not key or not val:
      raise ValueError(
          f'Empty math substitution key or value for {raw_key!r} in {path}.'
      )
    if key in mapping:
      raise ValueError(
          f'Duplicate normalized math expression {key!r} in {path}.'
      )
    mapping[key] = val
  return mapping


# Explicit lookup table mapping normalized code-math expressions (`$...$` and
# `$$...$$` in docstrings) to their exact LaTeX representations, loaded from
# `math_substitutions.toml`. Any math expression not listed fails loudly.
_CODE_TO_LATEX: Mapping[str, str] = _load_code_to_latex(
    _MATH_SUBSTITUTIONS_PATH
)


def _code_to_latex(expr: str) -> str:
  """Looks up the LaTeX rendering for a code-math expression.

  Args:
    expr: Math expression written inside `$...$` or `$$...$$` in a docstring.

  Returns:
    The corresponding LaTeX math string from `math_substitutions.toml`.

  Raises:
    ValueError: If `expr` is empty or does not have an entry in
      `math_substitutions.toml`.
  """
  key = re.sub(r'\s+', ' ', expr.strip())
  if not key:
    raise ValueError('Empty math expression.')
  if key not in _CODE_TO_LATEX:
    raise ValueError(
        f'Unknown math expression {key!r}. Add an explicit mapping to '
        'docs/sphinxext/math_substitutions.toml.'
    )
  return _CODE_TO_LATEX[key]


def _ensure_torax_importable() -> None:
  """Ensures `torax` and optional external dependencies are importable."""
  if (
      'qualikiz_tools' not in sys.modules
      and importlib.util.find_spec('qualikiz_tools') is None
  ):
    for mod_name in (
        'qualikiz_tools',
        'qualikiz_tools.qualikiz_io',
        'qualikiz_tools.qualikiz_io.inputfiles',
        'qualikiz_tools.qualikiz_io.qualikizrun',
    ):
      mock_mod = mock.MagicMock()
      mock_mod.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
      sys.modules[mod_name] = mock_mod

  if 'torax' in sys.modules:
    return
  try:
    importlib.import_module('torax')
    return
  except ModuleNotFoundError:
    pass

  # When docs/ lives directly inside the torax package directory (rather than a
  # parent repo directory containing torax/), register the package root
  # directly without adding parent directories to sys.path.
  docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  pkg_dir = os.path.dirname(docs_dir)
  init_py = os.path.join(pkg_dir, '__init__.py')
  if os.path.isfile(init_py):
    spec = importlib.util.spec_from_file_location(
        'torax', init_py, submodule_search_locations=[pkg_dir]
    )
    if spec is not None and spec.loader is not None:
      module = importlib.util.module_from_spec(spec)
      sys.modules['torax'] = module
      spec.loader.exec_module(module)


@functools.cache
def _get_torax_param_types() -> tuple[Any, type[Any], type[Any], type[Any]]:
  """Lazily imports TORAX parameter types once `torax` is importable."""
  _ensure_torax_importable()
  # pylint: disable=g-import-not-at-top
  from torax._src import array_typing
  from torax._src.torax_pydantic import interpolated_param_1d
  from torax._src.torax_pydantic import interpolated_param_2d
  # pylint: enable=g-import-not-at-top

  return (
      array_typing,
      interpolated_param_1d.TimeVaryingScalar,
      interpolated_param_2d.TimeVaryingArray,
      interpolated_param_2d.SparseTimeVaryingArray,
  )


@functools.cache
def _get_jax_array_bases() -> tuple[type[Any], ...]:
  """Returns available JAX/jaxtyping base array classes for `issubclass` checks."""
  bases: list[type[Any]] = []
  for mod_name, attr_name in (
      ('jax', 'Array'),
      ('jaxtyping', 'AbstractArray'),
  ):
    try:
      mod = importlib.import_module(mod_name)
      cls = getattr(mod, attr_name, None)
      if isinstance(cls, type):
        bases.append(cls)
    except ImportError:
      pass
  return tuple(bases)


def _import_symbol(dotted_path: str) -> Any:
  """Imports and returns a Python object from its dotted module path."""
  _ensure_torax_importable()
  module_path, _, attr_name = dotted_path.rpartition('.')
  if not module_path:
    raise ValueError(
        f'Expected fully qualified symbol path, got {dotted_path!r}.'
    )
  module = importlib.import_module(module_path)
  return getattr(module, attr_name)


def _import_class(dotted_path: str) -> type[Any]:
  """Imports and returns a class from its dotted module path."""
  cls = _import_symbol(dotted_path)
  if not isinstance(cls, type):
    raise TypeError(f'{dotted_path!r} is not a class.')
  return cls


def _find_closing_fence(lines: Sequence[str], start_idx: int) -> int | None:
  """Returns the index of the first closing ` ``` ` fence in `lines[start_idx:]`."""
  for idx in range(start_idx, len(lines)):
    if _FENCE_CLOSE_RE.match(lines[idx]) is not None:
      return idx
  return None


def _update_fence_state(
    line: str, lines: Sequence[str], next_idx: int, in_fence: bool
) -> tuple[bool, bool]:
  """Updates triple-backtick fence state and returns `(in_fence, is_fenced_line)`."""
  if not in_fence:
    if (
        _FENCE_OPEN_RE.match(line) is not None
        and _find_closing_fence(lines, next_idx) is not None
    ):
      return True, True
    return False, False
  if _FENCE_CLOSE_RE.match(line) is not None:
    return False, True
  return True, True


def _iter_rst_blocks(
    rst_lines: Sequence[str], *, extract_fences: bool = False
) -> Iterator[tuple[str, Any]]:
  """Partitions RST lines into verbatim blocks, optional code fences, and prose."""
  in_directive_block = False
  directive_indent = 0
  pending_literal_indent: int | None = None
  paragraph_lines: list[str] = []

  def _flush_paragraph(
      *, check_literal: bool = True
  ) -> Iterator[tuple[str, Any]]:
    nonlocal pending_literal_indent
    if not paragraph_lines:
      return
    if check_literal:
      non_empty = [l for l in paragraph_lines if l.strip()]
      if non_empty:
        last_stripped = non_empty[-1].lstrip()
        if last_stripped.rstrip().endswith(
            '::'
        ) and not last_stripped.startswith('.. '):
          pending_literal_indent = min(
              len(l) - len(l.lstrip()) for l in non_empty
          )
    yield ('prose', list(paragraph_lines))
    paragraph_lines.clear()

  i = 0
  n = len(rst_lines)
  while i < n:
    line = rst_lines[i]
    stripped = line.lstrip()
    indent = len(line) - len(stripped)

    if in_directive_block:
      if stripped and indent <= directive_indent:
        in_directive_block = False
      else:
        yield ('verbatim', [line])
        i += 1
        continue

    if pending_literal_indent is not None:
      if not stripped:
        yield ('verbatim', [line])
        i += 1
        continue
      if indent > pending_literal_indent:
        in_directive_block = True
        directive_indent = pending_literal_indent
        pending_literal_indent = None
        yield ('verbatim', [line])
        i += 1
        continue
      pending_literal_indent = None

    if stripped.startswith(('.. code-block::', '.. math::', '.. raw::')):
      yield from _flush_paragraph()
      in_directive_block = True
      directive_indent = indent
      yield ('verbatim', [line])
      i += 1
      continue

    if (
        extract_fences
        and (open_match := _FENCE_OPEN_RE.match(line)) is not None
    ):
      close_idx = _find_closing_fence(rst_lines, i + 1)
      if close_idx is not None:
        code_lines = rst_lines[i + 1 : close_idx]
        if any(cl.strip() for cl in code_lines):
          yield from _flush_paragraph(check_literal=False)
          has_following_text = close_idx + 1 < n and bool(
              rst_lines[close_idx + 1].strip()
          )
          yield ('fence', (open_match, code_lines, has_following_text))
          i = close_idx + 1
          continue

    if not stripped:
      yield from _flush_paragraph()
      yield ('verbatim', [line])
      i += 1
      continue

    paragraph_lines.append(line)
    i += 1

  yield from _flush_paragraph()


def _convert_fenced_code_blocks(rst_lines: Sequence[str]) -> list[str]:
  """Converts triple-backtick code fences into `.. code-block::` directives."""
  result: list[str] = []
  for kind, payload in _iter_rst_blocks(rst_lines, extract_fences=True):
    if kind != 'fence':
      result.extend(payload)
      continue
    open_match, code_lines, has_following_text = payload
    indent_str = open_match.group(1)
    fence_indent = len(indent_str)
    lang = open_match.group(2) or 'python'
    if result and result[-1].strip():
      result.append('')
    result.append(f'{indent_str}.. code-block:: {lang}')
    result.append('')
    for cl in code_lines:
      if not cl.strip():
        result.append('')
      elif len(cl) >= fence_indent and not cl[:fence_indent].strip():
        result.append(f'{indent_str}   {cl[fence_indent:]}')
      else:
        result.append(f'{indent_str}   {cl.lstrip()}')
    if has_following_text:
      result.append('')
  return result


def _transform_prose_paragraphs(
    rst_lines: Sequence[str], transform: Callable[[str], str]
) -> list[str]:
  """Applies `transform` to prose paragraphs while skipping directive blocks."""
  transformed: list[str] = []
  for kind, payload in _iter_rst_blocks(rst_lines):
    if kind == 'prose':
      transformed.extend(transform('\n'.join(payload)).split('\n'))
    else:
      transformed.extend(payload)
  return transformed


def _normalize_inline_code(
    rst_lines: Sequence[str],
    *,
    convert_single_backticks: bool = True,
    location: tuple[str, int | None] | str | None = None,
    context: str | None = None,
    strict: bool = False,
) -> list[str]:
  """Converts single-backtick code and `$`/`$$` math into RST markup."""
  ctx_suffix = f' in {context}' if context else ''

  def _report_error(msg: str) -> None:
    if strict:
      raise ValueError(msg)
    logger.warning('torax_config: %s', msg, location=location)

  def _process_plain(raw_plain: str) -> str:
    if re.search(r'(?<!\\)\$', raw_plain):
      _report_error(
          f'Unmatched or malformed `$` math delimiter{ctx_suffix}:'
          f' {raw_plain.strip()!r}'
      )
    return raw_plain.replace(r'\$', '$')

  def _translate_math(raw_math: str) -> str:
    try:
      return _code_to_latex(raw_math)
    except ValueError as exc:
      _report_error(
          f'Failed to parse math expression {raw_math!r}{ctx_suffix}: {exc}'
      )
      return raw_math.strip()

  def _normalize_segment(text: str, base_indent: int) -> str:
    indent_str = ' ' * base_indent
    parts: list[str] = []
    last = 0
    for match in _INLINE_MARKUP_RE.finditer(text):
      if match.start() > last:
        parts.append(_process_plain(text[last : match.start()]))
      if match.group(1) is not None:
        parts.append(
            f'``{match.group(1)}``'
            if convert_single_backticks
            else match.group(0)
        )
      elif match.group(2) is not None:
        if parts:
          parts[-1] = parts[-1].rstrip(' \t\n')
        converted_display = _translate_math(match.group(2).strip())
        math_lines = [
            l.strip() for l in converted_display.splitlines() if l.strip()
        ]
        rest_start = match.end()
        if (
            rest_start < len(text)
            and text[rest_start] in '.,;:'
            and (rest_start + 1 == len(text) or text[rest_start + 1].isspace())
        ):
          if math_lines:
            math_lines[-1] += text[rest_start]
          rest_start += 1
        prefix = '\n\n' if ''.join(parts).strip() else ''
        block = f'{prefix}{indent_str}.. math::\n\n' + '\n'.join(
            f'{indent_str}   {ml}' for ml in math_lines
        )
        parts.append(block)
        ws_match = re.match(r'[ \t]*(?:\n[ \t]*)?', text[rest_start:])
        if ws_match is not None:
          rest_start += ws_match.end()
        if rest_start < len(text):
          parts.append(f'\n\n{indent_str}')
        last = rest_start
        continue
      elif match.group(3) is not None:
        suffix = (
            r'\ '
            if match.end() < len(text) and text[match.end()].isalnum()
            else ''
        )
        rendered_math = _translate_math(match.group(3))
        parts.append(f':math:`{rendered_math}`{suffix}')
      else:
        parts.append(match.group(0))
      last = match.end()
    if last < len(text):
      parts.append(_process_plain(text[last:]))
    return ''.join(parts)

  def _normalize_paragraph(paragraph: str) -> str:
    first_non_empty = next((l for l in paragraph.splitlines() if l.strip()), '')
    stripped_first = first_non_empty.lstrip()
    base_indent = len(first_non_empty) - len(stripped_first)
    if stripped_first.startswith(('* ', '- ')) or _ENUM_LIST_ITEM_RE.match(
        stripped_first
    ):
      base_indent += 2
    return _normalize_segment(paragraph, base_indent)

  return _transform_prose_paragraphs(rst_lines, _normalize_paragraph)


def _dedent_and_trim(lines: Sequence[str], first_line: str = '') -> list[str]:
  """Dedents `lines`, prepends `first_line` if non-empty, and strips blank edges."""
  non_empty_indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
  dedent = min(non_empty_indents) if non_empty_indents else 0
  result: list[str] = []
  if first_line.strip():
    result.append(first_line.strip())
  result.extend(l[dedent:] if len(l) >= dedent else '' for l in lines)
  while result and not result[0].strip():
    result.pop(0)
  while result and not result[-1].strip():
    result.pop()
  return result


def _parse_docstring(
    docstring: str | None,
    *,
    location: tuple[str, int | None] | str | None = None,
    context: str | None = None,
    strict: bool = False,
) -> tuple[str, dict[str, str]]:
  """Extracts class summary and `Attributes:` from a docstring.

  Args:
    docstring: Raw class docstring.
    location: Optional Sphinx `(docname, lineno)` or document location for
      warnings.
    context: Optional qualified symbol name for warning messages.
    strict: If True, raises `ValueError` instead of logging a Sphinx warning on
      malformed `$` math expressions.

  Returns:
    A tuple of `(summary_rst, attributes_dict)` where `attributes_dict` maps
    each attribute name to its cleaned reStructuredText description.
  """
  if not docstring:
    return '', {}

  lines = inspect.cleandoc(docstring).splitlines()
  summary_lines: list[str] = []
  sections: dict[str, list[str]] = {}
  current_section: str | None = None
  in_fence = False

  for idx, line in enumerate(lines):
    in_fence, is_fenced = _update_fence_state(line, lines, idx + 1, in_fence)
    if (
        not is_fenced
        and len(line) == len(line.lstrip())
        and _SECTION_HEADER_RE.match(line)
    ):
      current_section = line.strip()
      sections.setdefault(current_section, [])
      in_fence = False
      continue
    if current_section is None:
      summary_lines.append(line)
    else:
      sections[current_section].append(line)

  def _render_lines(raw_lines: Sequence[str], *, ctx: str | None) -> str:
    return '\n'.join(
        _normalize_inline_code(
            _convert_fenced_code_blocks(raw_lines),
            location=location,
            context=ctx,
            strict=strict,
        )
    )

  while summary_lines and not summary_lines[-1].strip():
    summary_lines.pop()
  summary_rst = _render_lines(summary_lines, ctx=context)

  attributes: dict[str, str] = {}
  attr_lines = sections.get('Attributes:')
  if attr_lines:
    attr_indent: int | None = None
    raw_entries: list[tuple[str, str, list[str]]] = []
    in_fence = False

    for offset, line in enumerate(attr_lines):
      in_fence, is_fenced = _update_fence_state(
          line, attr_lines, offset + 1, in_fence
      )
      if is_fenced:
        if raw_entries:
          raw_entries[-1][2].append(line)
        continue

      stripped = line.lstrip()
      if not stripped:
        if raw_entries:
          raw_entries[-1][2].append('')
        continue

      indent = len(line) - len(stripped)
      if attr_indent is None:
        attr_indent = indent

      if indent == attr_indent and (
          match := _ATTRIBUTE_HEADER_RE.match(stripped)
      ):
        raw_entries.append((match.group(1), match.group(2), []))
      elif raw_entries:
        raw_entries[-1][2].append(line)

    for name, first_line, cont_lines in raw_entries:
      entry_lines = _dedent_and_trim(cont_lines, first_line)
      attr_context = f'{context}.{name}' if context else name
      attributes[name] = _render_lines(entry_lines, ctx=attr_context)

  return summary_rst, attributes


def _collect_mro_attributes(
    cls: type[Any],
    *,
    location: tuple[str, int | None] | str | None = None,
    strict: bool = False,
) -> tuple[dict[str, str], set[str]]:
  """Collects attribute docstrings across MRO."""
  combined: dict[str, str] = {}
  own_documented: set[str] = set()
  for base in reversed(cls.__mro__):
    if base is pydantic.BaseModel or base is object:
      continue
    if issubclass(base, pydantic.BaseModel):
      base_context = f'{base.__module__}.{base.__qualname__}'
      _, attrs = _parse_docstring(
          base.__doc__,
          location=location,
          context=base_context,
          strict=strict,
      )
      combined.update(attrs)
      if base is cls:
        own_documented = set(attrs.keys())
  return combined, own_documented


def _annotation_includes_str(annotation: Any) -> bool:
  """Returns whether `annotation` accepts a top-level `str`."""
  if annotation is None:
    return False
  if annotation is str:
    return True
  origin = typing.get_origin(annotation)
  if origin is Annotated:
    args = typing.get_args(annotation)
    return bool(args) and _annotation_includes_str(args[0])
  if origin is Union or origin is types.UnionType:
    return any(
        _annotation_includes_str(arg) for arg in typing.get_args(annotation)
    )
  return False


def _unwrap_annotation(
    annotation: Any, *, unwrap_type_alias: bool = False
) -> Any:
  """Strips outer `Annotated[...]` (and optionally `TypeAliasType`) wrappers."""
  while True:
    if unwrap_type_alias and isinstance(annotation, TypeAliasType):
      annotation = annotation.__value__
      continue
    if typing.get_origin(annotation) is Annotated:
      annotation = typing.get_args(annotation)[0]
      continue
    return annotation


def _unbox_enum(member: enum.Enum) -> Any:
  """Unboxes an `Enum` member into its TORAX configuration literal value."""
  if type(member).__name__ == 'GeometryType':
    return member.name.lower()
  return member.value


def _validator_accepts_str(meta: Any) -> bool:
  """Returns whether a Pydantic BeforeValidator declares `str` input support."""
  if not isinstance(meta, pydantic.BeforeValidator):
    return False
  if _annotation_includes_str(getattr(meta, 'json_schema_input_type', None)):
    return True
  func = getattr(meta, 'func', None)
  while isinstance(func, functools.partial):
    func = func.func
  if func is None or not callable(func):
    return False
  try:
    params = list(inspect.signature(func).parameters.values())
  except (TypeError, ValueError):
    return False
  if not params:
    return False
  first_param = params[0]
  try:
    type_hints = typing.get_type_hints(func)
  except Exception:  # pylint: disable=broad-exception-caught
    type_hints = {}
  first_ann = type_hints.get(first_param.name, first_param.annotation)
  return first_ann is not inspect.Parameter.empty and _annotation_includes_str(
      first_ann
  )


_IGNORED_METADATA_MARKERS: frozenset[str] = frozenset({
    '_pydantic_jax_static_field',
    '_pydantic_time_invariant_field',
})


def _clean_type_metadata(metadata: Iterable[Any]) -> tuple[Any, ...]:
  """Strips internal static/time-invariant markers and flattens `FieldInfo`."""
  cleaned: list[Any] = []
  for item in metadata:
    if isinstance(item, str) and item in _IGNORED_METADATA_MARKERS:
      continue
    if isinstance(item, pydantic.fields.FieldInfo):
      cleaned.extend(_clean_type_metadata(item.metadata))
      continue
    cleaned.append(item)
  return tuple(cleaned)


def _callable_equal(fn_a: Any, fn_b: Any) -> bool:
  """Compares two callables (including `functools.partial` wrappers)."""
  if fn_a is fn_b:
    return True
  if isinstance(fn_a, functools.partial) and isinstance(
      fn_b, functools.partial
  ):
    return (
        _callable_equal(fn_a.func, fn_b.func)
        and fn_a.args == fn_b.args
        and fn_a.keywords == fn_b.keywords
    )
  return bool(fn_a == fn_b)


def _metadata_item_equal(a: Any, b: Any) -> bool:
  """Compares two `Annotated` metadata items safely."""
  if a is b:
    return True
  if type(a) is not type(b):
    return False
  try:
    if a == b:
      return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False
  func_a = getattr(a, 'func', None)
  func_b = getattr(b, 'func', None)
  if func_a is not None and func_b is not None:
    return _callable_equal(func_a, func_b)
  return False


@functools.cache
def _get_public_annotated_aliases() -> tuple[
    tuple[Any, tuple[Any, ...], str], ...
]:
  """Returns `(base_type, cleaned_metadata, name)` for public type aliases."""
  _ensure_torax_importable()
  # pylint: disable=g-import-not-at-top
  from torax import torax_pydantic
  # pylint: enable=g-import-not-at-top

  entries: list[tuple[Any, tuple[Any, ...], str]] = []
  for mod in (torax_pydantic, pydantic.types):
    for name in getattr(mod, '__all__', ()):
      obj = getattr(mod, name, None)
      if isinstance(obj, TypeAliasType):
        obj = obj.__value__
      if typing.get_origin(obj) is not Annotated:
        continue
      args = typing.get_args(obj)
      cleaned_meta = _clean_type_metadata(args[1:])
      if not cleaned_meta:
        continue
      entries.append((args[0], cleaned_meta, name))
  return tuple(entries)


def _match_public_annotated_alias(
    base_type: Any, cleaned_meta: tuple[Any, ...]
) -> str | None:
  """Returns the unqualified public alias name matching `(base_type, meta)`."""
  for cand_base, cand_meta, name in _get_public_annotated_aliases():
    if base_type == cand_base and len(cleaned_meta) == len(cand_meta):
      if all(
          _metadata_item_equal(a, b)
          for a, b in zip(cleaned_meta, cand_meta, strict=True)
      ):
        return name
  return None


def _format_type(annotation: Any, metadata: Sequence[Any] = ()) -> str:
  """Formats a Python/Pydantic type annotation into a readable type string."""
  if isinstance(annotation, TypeAliasType):
    return _format_type(annotation.__value__, metadata)

  origin = typing.get_origin(annotation)
  if origin is Annotated:
    args = typing.get_args(annotation)
    return _format_type(args[0], (*args[1:], *metadata))

  (
      array_typing,
      time_varying_scalar_cls,
      time_varying_array_cls,
      sparse_tva_cls,
  ) = _get_torax_param_types()
  accepts_str = any(_validator_accepts_str(meta) for meta in metadata)

  def _maybe_append_str(formatted: str) -> str:
    if accepts_str and formatted != 'str' and not formatted.endswith(' | str'):
      return f'{formatted} | str'
    return formatted

  cleaned_meta = _clean_type_metadata(metadata)
  if cleaned_meta:
    matched_alias = _match_public_annotated_alias(annotation, cleaned_meta)
    if matched_alias is not None:
      return _maybe_append_str(matched_alias)

  def _join_unique_arms(arm_annotations: Iterable[Any]) -> str:
    arms: list[str] = []
    seen: set[str] = set()
    literal_idx: int | None = None
    literal_items: list[str] = []
    for arg in arm_annotations:
      rendered = _format_type(arg)
      if rendered.startswith('[') and rendered.endswith(']'):
        for tok in rendered[1:-1].split(','):
          tok = tok.strip()
          if tok and tok not in literal_items:
            literal_items.append(tok)
        if literal_idx is None:
          literal_idx = len(arms)
          arms.append('')
        continue
      if rendered not in seen:
        seen.add(rendered)
        arms.append(rendered)
    if literal_idx is not None:
      arms[literal_idx] = f'[{", ".join(literal_items)}]'
    if accepts_str and 'str' not in seen:
      arms.append('str')
    return ' | '.join(arms)

  if annotation is Any:
    return _maybe_append_str('Any')
  for scalar_alias, scalar_name in (
      (array_typing.FloatScalar, 'float'),
      (array_typing.BoolScalar, 'bool'),
      (array_typing.IntScalar, 'int'),
  ):
    if annotation is scalar_alias:
      return _maybe_append_str(scalar_name)

  if origin is Union or origin is types.UnionType:
    return _join_unique_arms(typing.get_args(annotation))

  if origin is Literal:
    lit_args = typing.get_args(annotation)
    if not lit_args:
      return _maybe_append_str('Literal')
    rendered_vals: list[str] = []
    for lit_arg in lit_args:
      unboxed_lit = (
          _unbox_enum(lit_arg) if isinstance(lit_arg, enum.Enum) else lit_arg
      )
      tok = repr(unboxed_lit)
      if tok not in rendered_vals:
        rendered_vals.append(tok)
    return _maybe_append_str(f'[{", ".join(rendered_vals)}]')

  if origin in (dict, Mapping):
    container_name = 'Mapping' if origin is Mapping else 'dict'
    dict_args = typing.get_args(annotation)
    if len(dict_args) == 2:
      return _maybe_append_str(
          f'{container_name}[{_format_type(dict_args[0])},'
          f' {_format_type(dict_args[1])}]'
      )
    return _maybe_append_str(container_name)

  if origin is tuple:
    seq_args = typing.get_args(annotation)
    if annotation is Tuple and not seq_args:
      return _maybe_append_str('tuple')
    if not seq_args or (
        len(seq_args) == 1
        and isinstance(seq_args[0], tuple)
        and not seq_args[0]
    ):
      return _maybe_append_str('tuple[()]')
    if len(seq_args) == 2 and seq_args[1] is Ellipsis:
      return _maybe_append_str(f'tuple[{_format_type(seq_args[0])}, ...]')
    inner = ', '.join(_format_type(arg) for arg in seq_args)
    return _maybe_append_str(f'tuple[{inner}]')

  if origin in (list, Sequence, set, frozenset):
    container_names = {
        list: 'list',
        Sequence: 'Sequence',
        set: 'set',
        frozenset: 'frozenset',
    }
    container_name = container_names[origin]
    elem_args = typing.get_args(annotation)
    if len(elem_args) == 1:
      return _maybe_append_str(
          f'{container_name}[{_format_type(elem_args[0])}]'
      )
    return _maybe_append_str(container_name)

  if origin is Callable:
    return _maybe_append_str('Callable')

  if annotation is os.PathLike or origin is os.PathLike:
    return _maybe_append_str('PathLike')

  if annotation is types.NoneType or annotation is None:
    return _maybe_append_str('None')

  if isinstance(annotation, type):
    if issubclass(
        annotation,
        (time_varying_scalar_cls, time_varying_array_cls, sparse_tva_cls),
    ):
      return _maybe_append_str(annotation.__name__)
    if issubclass(annotation, (np.ndarray, *_get_jax_array_bases())):
      return _maybe_append_str('array')
    if issubclass(annotation, np.floating):
      return _maybe_append_str('float')
    if issubclass(annotation, np.integer):
      return _maybe_append_str('int')
    if issubclass(annotation, np.bool_):
      return _maybe_append_str('bool')
    if callable(getattr(annotation, '__fspath__', None)):
      return _maybe_append_str('PathLike')
    if issubclass(annotation, enum.Enum):
      enum_vals = [repr(_unbox_enum(m)) for m in annotation]
      if not enum_vals:
        return _maybe_append_str(annotation.__name__)
      dedup_vals = list(dict.fromkeys(enum_vals))
      return _maybe_append_str(f'[{", ".join(dedup_vals)}]')
    if annotation in (bool, int, float, str):
      return _maybe_append_str(annotation.__name__)
    if issubclass(annotation, pydantic.BaseModel):
      return _maybe_append_str('dict')
    return _maybe_append_str(annotation.__name__)

  return _maybe_append_str(str(annotation))


class _RawRepr(str):
  """String wrapper whose `repr()` renders unquoted in formatted defaults."""

  def __repr__(self) -> str:
    return str(self)


def _single_literal_discriminator(
    cls: type[pydantic.BaseModel],
) -> tuple[str, Any] | None:
  """Returns `(field_name, literal_val)` if `cls` has a 1-value `Literal` field."""
  for fname, finfo in cls.model_fields.items():
    ann = _unwrap_annotation(finfo.annotation)
    if typing.get_origin(ann) is Literal:
      lit_args = typing.get_args(ann)
      if len(lit_args) == 1:
        return fname, lit_args[0]
  return None


def _unbox_basemodel_default(value: pydantic.BaseModel) -> dict[str, Any]:
  """Unboxes a nested `BaseModel` default into a minimal config `dict`."""
  cls = type(value)
  result: dict[str, Any] = {}
  disc = _single_literal_discriminator(cls)
  if disc is not None:
    disc_name, _ = disc
    result[disc_name] = _unbox_default_value(getattr(value, disc_name))
  for fname in cls.model_fields:
    if fname in result or fname not in value.model_fields_set:
      continue
    actual = _unbox_default_value(getattr(value, fname))
    finfo = cls.model_fields[fname]
    if finfo.is_required() or _format_default(finfo) != repr(actual):
      result[fname] = actual
  return result


def _unbox_default_value(value: Any) -> Any:
  """Converts TORAX Pydantic wrapper objects into plain Python primitives."""
  _, time_varying_scalar_cls, time_varying_array_cls, sparse_tva_cls = (
      _get_torax_param_types()
  )

  if isinstance(value, enum.Enum):
    return _unbox_enum(value)

  if isinstance(value, time_varying_scalar_cls):
    if len(value.value) == 1 and float(value.time[0]) == 0.0:
      if value.is_bool_param:
        return bool(value.value[0] > 0.5)
      return float(value.value[0])
    return {
        float(t): float(v) for t, v in zip(value.time, value.value, strict=True)
    }

  if isinstance(value, time_varying_array_cls):
    if len(value.value) == 1 and 0.0 in value.value:
      rho_arr, val_arr = value.value[0.0]
      if len(rho_arr) == 1 and float(rho_arr[0]) == 0.0:
        return float(val_arr[0])
    return {
        float(t): {
            float(r): float(v) for r, v in zip(rho_arr, val_arr, strict=True)
        }
        for t, (rho_arr, val_arr) in value.value.items()
    }

  if isinstance(value, sparse_tva_cls):
    if len(value.values) == 1 and value.values[0][0] == (0.0, 1.0):
      return _unbox_default_value(value.values[0][1])
    by_time: dict[float, dict[float | tuple[float, float], Any]] = {}
    for point_or_range, tva in value.values:
      for t, (rho_arr, val_arr) in tva.value.items():
        if len(rho_arr) == 1 and float(rho_arr[0]) == 0.0:
          loc_val: Any = float(val_arr[0])
        else:
          loc_val = {
              float(r): float(v) for r, v in zip(rho_arr, val_arr, strict=True)
          }
        by_time.setdefault(float(t), {})[point_or_range] = loc_val
    return by_time

  if isinstance(value, np.ndarray):
    return _RawRepr(f'np.array({value.tolist()!r})')

  if type(value) in (tuple, list, set, frozenset):
    return type(value)(_unbox_default_value(v) for v in value)

  if isinstance(value, tuple):
    return tuple(_unbox_default_value(v) for v in value)

  if isinstance(value, Mapping):
    return {k: _unbox_default_value(v) for k, v in value.items()}

  if isinstance(value, pydantic.BaseModel):
    return _unbox_basemodel_default(value)

  return value


def _empty_constructor_repr(tp: Any) -> str | None:
  """Returns an empty constructor literal (`[]`, `{}`, etc.) for `tp`."""
  origin = typing.get_origin(tp) or tp
  if origin is list:
    return '[]'
  if origin is set:
    return 'set()'
  if origin is tuple:
    return '()'
  if origin is dict:
    return '{}'
  if isinstance(origin, type) and issubclass(origin, pydantic.BaseModel):
    disc = _single_literal_discriminator(origin)
    if disc is not None:
      return repr({disc[0]: _unbox_default_value(disc[1])})
    return '{}'
  return None


def _format_default(field_info: Any) -> str | None:
  """Formats the default value of a Pydantic FieldInfo, or None if required."""
  if field_info.is_required():
    return None

  default = field_info.default
  if default is pydantic.fields.PydanticUndefined:
    if field_info.default_factory is None:
      return None
    try:
      default = field_info.default_factory()
    except TypeError:
      try:
        default = field_info.default_factory({})
      except Exception:  # pylint: disable=broad-exception-caught
        factory = field_info.default_factory
        while isinstance(factory, functools.partial):
          factory = factory.func
        if (rendered := _empty_constructor_repr(factory)) is not None:
          return rendered
        if callable(factory):
          try:
            ret_ann = typing.get_type_hints(factory).get('return')
          except Exception:  # pylint: disable=broad-exception-caught
            ret_ann = None
          if (rendered := _empty_constructor_repr(ret_ann)) is not None:
            return rendered
        return '{}'

  return repr(_unbox_default_value(default))


def _csv_option(argument: str | None) -> list[str]:
  if not argument:
    return []
  return [item.strip() for item in argument.split(',') if item.strip()]


class ToraxConfigDirective(sphinx_docutils.SphinxDirective):
  """Directive `.. torax_config:: <dotted.symbol.Path>` for Pydantic configs."""

  required_arguments = 1
  optional_arguments = 0
  has_content = True
  option_spec = {
      'include-class-doc': directives.flag,
      'own-fields-only': directives.flag,
      'exclude-fields': _csv_option,
      'include-fields': _csv_option,
  }

  def _resolve_field_names(self, cls: type[Any]) -> list[str]:
    exclude_fields = set(self.options.get('exclude-fields', []))
    include_fields = self.options.get('include-fields', [])
    if include_fields:
      return [f for f in include_fields if f not in exclude_fields]
    if 'own-fields-only' in self.options:
      own_names = set(cls.__annotations__.keys())
      return [
          f
          for f in cls.model_fields.keys()
          if f in own_names and f not in exclude_fields
      ]
    return [f for f in cls.model_fields.keys() if f not in exclude_fields]

  def run(self) -> list[nodes.Node]:
    class_path = self.arguments[0].strip()
    target = _import_symbol(class_path)

    if not (
        isinstance(target, type) and issubclass(target, pydantic.BaseModel)
    ):
      union_models = _extract_basemodels(target)
      disc_info = _collect_discriminator_literals(union_models)
      if disc_info is None:
        raise self.error(
            f'{class_path!r} is neither a Pydantic BaseModel nor a'
            ' discriminated union of BaseModels.'
        )
      disc_field, lit_vals = disc_info
      dl_node = nodes.definition_list()
      dl_node['classes'].extend(['simple', 'torax-config-dl'])
      item_node = nodes.definition_list_item()
      term_node = nodes.term()
      _, _, symbol_name = class_path.rpartition('.')
      anchor_id = f'{symbol_name.lower()}-{disc_field}'
      if anchor_id not in self.state.document.ids:
        term_node['ids'].append(anchor_id)
      term_node += nodes.literal(text=disc_field, classes=['torax-config-name'])
      lit_type = f'[{", ".join(repr(v) for v in lit_vals)}]'
      term_node += nodes.inline(
          text=f'{lit_type} · required', classes=['torax-config-meta']
      )
      item_node += term_node
      def_node = nodes.definition()
      if self.content:
        self.state.nested_parse(self.content, self.content_offset, def_node)
      else:
        def_node += nodes.paragraph(
            '', f'Selects the {disc_field} implementation.'
        )
      item_node += def_node
      dl_node += item_node
      return [dl_node]

    cls = target

    location = (self.env.docname, self.lineno)
    summary_rst, _ = _parse_docstring(
        cls.__doc__, location=location, context=class_path
    )
    attr_docs, own_documented = _collect_mro_attributes(cls, location=location)
    field_names = self._resolve_field_names(cls)

    class_vars = set(getattr(cls, '__class_vars__', ()))
    for documented_name in sorted(own_documented):
      if (
          documented_name not in cls.model_fields
          and documented_name not in class_vars
      ):
        logger.warning(
            'torax_config: %s docstring documents unknown attribute %r.',
            class_path,
            documented_name,
            location=location,
        )

    result_nodes: list[nodes.Node] = []

    if 'include-class-doc' in self.options and summary_rst:
      summary_container = nodes.container()
      self.state.nested_parse(
          statemachine.StringList(
              summary_rst.splitlines(), source=class_path
          ),
          self.content_offset,
          summary_container,
      )
      result_nodes.extend(summary_container.children)

    dl_node = nodes.definition_list()
    dl_node['classes'].extend(['simple', 'torax-config-dl'])

    for name in field_names:
      field_info = cls.model_fields[name]
      description_rst = attr_docs.get(name, '')
      if not description_rst:
        logger.warning(
            'torax_config: Field %r on %s is missing from its `Attributes:`'
            ' docstring.',
            name,
            class_path,
            location=location,
        )

      item_node = nodes.definition_list_item()

      term_node = nodes.term()
      anchor_id = f'{cls.__name__.lower()}-{name}'
      if anchor_id not in self.state.document.ids:
        term_node['ids'].append(anchor_id)
      term_node += nodes.literal(text=name, classes=['torax-config-name'])

      type_str = _format_type(field_info.annotation, field_info.metadata)
      default_str = _format_default(field_info)
      if default_str is not None:
        meta_text = f'{type_str} = {default_str}'
      else:
        meta_text = f'{type_str} · required'
      term_node += nodes.inline(text=meta_text, classes=['torax-config-meta'])
      item_node += term_node

      def_node = nodes.definition()
      if description_rst:
        self.state.nested_parse(
            statemachine.StringList(
                description_rst.splitlines(),
                source=f'{class_path}.{name}',
            ),
            self.content_offset,
            def_node,
        )
      item_node += def_node
      dl_node += item_node

    if field_names:
      result_nodes.append(dl_node)

    return result_nodes


def _process_autodoc_docstring(
    app: application.Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
  """Converts fenced code blocks and `$` math in autodoc docstrings."""
  del what, obj, options
  if any('```' in line for line in lines):
    lines[:] = _convert_fenced_code_blocks(lines)
  if any('$' in line for line in lines):
    docname = getattr(getattr(app, 'env', None), 'docname', None)
    location: tuple[str, int | None] | str = (
        (docname, None) if docname else name
    )
    lines[:] = _normalize_inline_code(
        lines,
        convert_single_backticks=False,
        location=location,
        context=name,
    )


def _process_source_read(
    app: application.Sphinx,
    docname: str,
    source: list[str],
) -> None:
  """Converts triple-backtick fenced blocks in reStructuredText sources."""
  if '```' not in source[0]:
    return
  doc_path = str(app.env.doc2path(docname))
  if not doc_path.endswith('.rst'):
    return
  lines = source[0].splitlines()
  converted = _convert_fenced_code_blocks(lines)
  trailing_newline = '\n' if source[0].endswith('\n') else ''
  source[0] = '\n'.join(converted) + trailing_newline


_CONFIG_SECTION_KEY_RE = re.compile(r'^[a-z_][A-Za-z0-9_-]*$')

_LEAF_PYDANTIC_NAMES: frozenset[str] = frozenset({
    'TimeVaryingScalar',
    'TimeVaryingScalarStep',
    'TimeVaryingArray',
    'SparseTimeVaryingArray',
    'Grid1D',
})


def _extract_basemodels(annotation: Any) -> list[type[pydantic.BaseModel]]:
  """Returns all `pydantic.BaseModel` subclasses inside a type annotation."""
  if isinstance(annotation, TypeAliasType):
    return _extract_basemodels(annotation.__value__)
  if isinstance(annotation, type) and issubclass(
      annotation, pydantic.BaseModel
  ):
    if annotation.__name__ in _LEAF_PYDANTIC_NAMES:
      return []
    return [annotation]
  origin = typing.get_origin(annotation)
  if origin is None:
    return []
  models: list[type[pydantic.BaseModel]] = []
  for arg in typing.get_args(annotation):
    for m in _extract_basemodels(arg):
      if m not in models:
        models.append(m)
  return models


def _find_discriminator_match(
    models: Sequence[type[pydantic.BaseModel]], target_value: str
) -> tuple[str, str, str | None] | None:
  """Checks if `target_value` is a Literal discriminator on `models`."""
  for model_cls in models:
    for field_name, field_info in model_cls.model_fields.items():
      ann = _unwrap_annotation(field_info.annotation)
      if typing.get_origin(ann) is Literal:
        for lit_val in typing.get_args(ann):
          val_str = str(_unbox_default_value(lit_val))
          if val_str == target_value:
            return (field_name, val_str, None)

  # Also check flattened discriminated-union fields (e.g. `geometry_configs` on
  # `GeometryConfig`) and `dict[str, UnionOfModels]` fields (e.g.
  # `core_transport_models` on `TransportConfig`).
  for model_cls in models:
    for field_name, field_info in model_cls.model_fields.items():
      ann = _unwrap_annotation(field_info.annotation, unwrap_type_alias=True)
      sub_models = _extract_basemodels(ann)
      sub_match = _find_discriminator_match(sub_models, target_value)
      if sub_match is None:
        continue
      if typing.get_origin(ann) in (dict, Mapping):
        return (sub_match[0], sub_match[1], field_name)
      if sub_match[0] in model_cls.model_fields:
        return (sub_match[0], sub_match[1], None)
  return None


def _resolve_models_and_keys(
    path_parts: Sequence[str],
) -> tuple[
    list[str],
    tuple[str, str] | None,
    list[type[pydantic.BaseModel]],
]:
  """Resolves `path_parts` against `ToraxConfig` into keys, discriminator, and models."""
  if not path_parts:
    return [], None, []
  try:
    torax_cfg_cls = _import_class(
        'torax._src.torax_pydantic.config.ToraxConfig'
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return list(path_parts), None, []

  top_key = path_parts[0]
  if top_key not in torax_cfg_cls.model_fields:
    return list(path_parts), None, []

  current_models = _extract_basemodels(
      torax_cfg_cls.model_fields[top_key].annotation
  )
  dict_keys: list[str] = [top_key]
  discriminator: tuple[str, str] | None = None

  for part in path_parts[1:]:
    matched_field_models: list[type[pydantic.BaseModel]] = []
    is_field = False
    for model_cls in current_models:
      if part in model_cls.model_fields:
        is_field = True
        for m in _extract_basemodels(model_cls.model_fields[part].annotation):
          if m not in matched_field_models:
            matched_field_models.append(m)
    if is_field and matched_field_models:
      dict_keys.append(part)
      current_models = matched_field_models
      discriminator = None
      continue

    disc_match = _find_discriminator_match(current_models, part)
    if disc_match is not None:
      disc_field, disc_val, container_field = disc_match
      if container_field is not None:
        dict_keys.extend([container_field, '<name>'])
      discriminator = (disc_field, disc_val)
    else:
      dict_keys.append(part)
      if is_field:
        discriminator = None

  return dict_keys, discriminator, current_models


def _resolve_section_schema(
    path_parts: Sequence[str],
) -> tuple[list[str], tuple[str, str] | None]:
  """Resolves `path_parts` against `ToraxConfig` into keys and discriminator."""
  dict_keys, discriminator, _ = _resolve_models_and_keys(path_parts)
  return dict_keys, discriminator


def _collect_discriminator_literals(
    models: Sequence[type[pydantic.BaseModel]],
) -> tuple[str, list[str]] | None:
  """Returns `(disc_field, literal_strs)` when `models` form a discriminated union."""
  if len(models) < 2:
    return None
  for fname in models[0].model_fields:
    vals: list[str] = []
    valid = True
    for m in models:
      finfo = m.model_fields.get(fname)
      if finfo is None:
        valid = False
        break
      ann = _unwrap_annotation(finfo.annotation)
      if typing.get_origin(ann) is not Literal:
        valid = False
        break
      lit_args = typing.get_args(ann)
      if len(lit_args) != 1:
        valid = False
        break
      val_str = str(_unbox_default_value(lit_args[0]))
      if val_str not in vals:
        vals.append(val_str)
    if valid and len(vals) >= 2:
      return fname, vals
  return None


def _split_meta_type_and_suffix(meta_text: str) -> tuple[str, str]:
  """Splits `.torax-config-meta` text into `(type_part, suffix)`."""
  depth = 0
  in_quote: str | None = None
  for idx, ch in enumerate(meta_text):
    if in_quote is not None:
      if ch == in_quote and (idx == 0 or meta_text[idx - 1] != '\\'):
        in_quote = None
      continue
    if ch in ("'", '"'):
      in_quote = ch
    elif ch == '[':
      depth += 1
    elif ch == ']':
      depth = max(0, depth - 1)
    elif depth == 0:
      if meta_text.startswith(' = ', idx) or meta_text.startswith(' · ', idx):
        return meta_text[:idx], meta_text[idx:]
  return meta_text, ''


def _rewrite_meta_type_to_literal(
    item_node: nodes.definition_list_item,
    lit_values: Sequence[str],
    *,
    merge_existing: bool = False,
) -> None:
  """Rewrites an item's `.torax-config-meta` type to `['a', 'b', ...]`."""
  term_node = _dl_item_term(item_node)
  if term_node is None:
    return
  meta_node = next(
      (
          c
          for c in term_node.children
          if isinstance(c, nodes.inline)
          and 'torax-config-meta' in c.get('classes', [])
      ),
      None,
  )
  if meta_node is None:
    return
  type_part, suffix = _split_meta_type_and_suffix(meta_node.astext())
  nullable = type_part.endswith(' | None')
  core_type = type_part[: -len(' | None')] if nullable else type_part

  merged: list[str] = []
  if (
      merge_existing
      and core_type.startswith('[')
      and core_type.endswith(']')
  ):
    inner = core_type[1:-1]
    try:
      parsed = ast.literal_eval(f'({inner},)')
      for v in parsed:
        s = str(v)
        if s not in merged:
          merged.append(s)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  for v in lit_values:
    if v not in merged:
      merged.append(v)
  if not merged:
    return

  new_type = f'[{", ".join(repr(v) for v in merged)}]'
  if nullable:
    new_type = f'{new_type} | None'
  meta_node.clear()
  meta_node += nodes.Text(f'{new_type}{suffix}')


_TIME_VARYING_TYPE_TARGETS: Mapping[str, str] = {
    'TimeVaryingScalar': 'time_varying_scalars',
    'TimeVaryingScalarStep': 'time_varying_scalars',
    'PositiveTimeVaryingScalar': 'time_varying_scalars',
    'NonNegativeTimeVaryingScalar': 'time_varying_scalars',
    'NonNegativeTimeVaryingScalarStep': 'time_varying_scalars',
    'UnitIntervalTimeVaryingScalar': 'time_varying_scalars',
    'TimeVaryingArray': 'time_varying_arrays',
    'PositiveTimeVaryingArray': 'time_varying_arrays',
    'NonNegativeTimeVaryingArray': 'time_varying_arrays',
    'SparseTimeVaryingArray': 'time_varying_arrays',
}

_TIME_VARYING_TOKEN_RE = re.compile(
    r'\b('
    + '|'.join(sorted(_TIME_VARYING_TYPE_TARGETS, key=len, reverse=True))
    + r')\b'
)


def _link_time_varying_meta_tokens(
    app: application.Sphinx,
    doctree: nodes.document,
    docname: str,
) -> None:
  """Links `TimeVarying*` type tokens inside `.torax-config-meta` spans."""
  env = getattr(app, 'env', None)
  std_domain = env.get_domain('std') if env is not None else None
  labels = getattr(std_domain, 'labels', {})
  found_docs = getattr(env, 'found_docs', set())

  resolved_targets: dict[str, tuple[str, str]] = {}
  for label_name in ('time_varying_scalars', 'time_varying_arrays'):
    if label_name in labels:
      target_doc, label_id, _ = labels[label_name]
      resolved_targets[label_name] = (target_doc, label_id)
    elif 'configuration' in found_docs or docname.startswith('configuration/'):
      resolved_targets[label_name] = (
          'configuration',
          label_name.replace('_', '-'),
      )

  if not resolved_targets:
    return

  def _make_ref(token: str, target_info: tuple[str, str]) -> nodes.reference:
    target_doc, label_id = target_info
    if docname == target_doc:
      return nodes.reference('', token, internal=True, refid=label_id)
    rel_uri = ''
    builder = getattr(app, 'builder', None)
    if builder is not None:
      try:
        rel_uri = builder.get_relative_uri(docname, target_doc)
      except Exception:  # pylint: disable=broad-exception-caught
        rel_uri = ''
    if not rel_uri:
      rel_uri = (
          '../configuration.html' if '/' in docname else 'configuration.html'
      )
    return nodes.reference(
        '', token, internal=True, refuri=f'{rel_uri}#{label_id}'
    )

  for meta_node in doctree.findall(nodes.inline):
    if 'torax-config-meta' not in meta_node.get('classes', []):
      continue
    meta_text = meta_node.astext()
    type_part, suffix = _split_meta_type_and_suffix(meta_text)
    if not _TIME_VARYING_TOKEN_RE.search(type_part):
      continue
    meta_node.clear()
    last_end = 0
    for match in _TIME_VARYING_TOKEN_RE.finditer(type_part):
      start, end = match.span()
      if start > last_end:
        meta_node += nodes.Text(type_part[last_end:start])
      tok = match.group(1)
      target_info = resolved_targets.get(_TIME_VARYING_TYPE_TARGETS[tok])
      if target_info is not None:
        meta_node += _make_ref(tok, target_info)
      else:
        meta_node += nodes.Text(tok)
      last_end = end
    if last_end < len(type_part) or suffix:
      meta_node += nodes.Text(type_part[last_end:] + suffix)


def _section_ancestors(section: nodes.section) -> list[nodes.section]:
  """Returns ancestor `<section>` nodes from outermost to innermost."""
  ancestors: list[nodes.section] = []
  parent = section.parent
  while parent is not None:
    if isinstance(parent, nodes.section):
      ancestors.append(parent)
    parent = parent.parent
  ancestors.reverse()
  return ancestors


def _dl_item_term(item_node: nodes.definition_list_item) -> nodes.term | None:
  """Returns the `<term>` child of a `<definition_list_item>`."""
  return next(
      (c for c in item_node.children if isinstance(c, nodes.term)), None
  )


def _dl_item_definition(
    item_node: nodes.definition_list_item,
) -> nodes.definition | None:
  """Returns the primary `<definition>` child of a `<definition_list_item>`."""
  return next(
      (
          c
          for c in item_node.children
          if isinstance(c, nodes.definition)
          and 'torax-config-nested-row' not in c.get('classes', [])
      ),
      None,
  )


def _get_item_param_name(item_node: nodes.definition_list_item) -> str | None:
  """Returns the parameter name from a config `<definition_list_item>`."""
  term_node = _dl_item_term(item_node)
  if term_node is None or not term_node.children:
    return None
  first_child = term_node.children[0]
  if isinstance(first_child, (nodes.literal, nodes.reference)):
    return first_child.astext().strip()
  return None


def _get_or_create_nested_dd(
    item_node: nodes.definition_list_item, extra_class: str
) -> nodes.definition:
  """Returns or appends the `.torax-config-nested-row` `<definition>`."""
  for child in item_node.children:
    if isinstance(
        child, nodes.definition
    ) and 'torax-config-nested-row' in child.get('classes', []):
      if extra_class not in child['classes']:
        child['classes'].append(extra_class)
      return child
  nested_dd = nodes.definition(
      '', classes=['torax-config-nested-row', extra_class]
  )
  item_node += nested_dd
  return nested_dd


def _merge_adjacent_config_dls(container: nodes.Element) -> None:
  """Merges consecutive `dl.torax-config-dl` children in `container`."""
  idx = 0
  while idx < len(container.children) - 1:
    curr_child = container.children[idx]
    next_child = container.children[idx + 1]
    if (
        isinstance(curr_child, nodes.definition_list)
        and 'torax-config-dl' in curr_child.get('classes', [])
        and isinstance(next_child, nodes.definition_list)
        and 'torax-config-dl' in next_child.get('classes', [])
    ):
      for item in list(next_child.children):
        curr_child += item
      container.remove(next_child)
    else:
      idx += 1


def _find_dl_item_in_section(
    sec_node: nodes.Element,
    target_param: str,
    section_schema: Mapping[int, Any],
    before_child: nodes.Node | None = None,
) -> nodes.definition_list_item | None:
  """Finds the `<definition_list_item>` for `target_param` in `sec_node`."""
  for child in sec_node.children:
    if isinstance(child, nodes.definition_list):
      for item_node in child.children:
        if (
            isinstance(item_node, nodes.definition_list_item)
            and _get_item_param_name(item_node) == target_param
        ):
          return item_node
  if before_child is not None and before_child in sec_node.children:
    child_idx = sec_node.children.index(before_child)
    for prev_sib in reversed(sec_node.children[:child_idx]):
      if (
          isinstance(prev_sib, nodes.section)
          and id(prev_sib) not in section_schema
      ):
        found = _find_dl_item_in_section(prev_sib, target_param, section_schema)
        if found is not None:
          return found
  return None


def _link_goto_option_literals(
    doctree: nodes.document,
    subsection_targets: Mapping[str, tuple[str, nodes.section]],
) -> None:
  """Wraps `'option'` literals in `<dd>` and bullet lists with goto links."""

  def _is_inside_section(node: nodes.Node, target_sec: nodes.section) -> bool:
    curr: nodes.Node | None = node
    while curr is not None:
      if curr is target_sec:
        return True
      curr = curr.parent
    return False

  def _link_container(container: nodes.Node) -> None:
    linked_targets: set[str] = set()
    for lit_node in list(container.findall(nodes.literal)):
      if lit_node.parent is None or isinstance(
          lit_node.parent, nodes.reference
      ):
        continue
      raw_lit = lit_node.astext().strip()
      if (
          len(raw_lit) >= 3
          and raw_lit[0] in ("'", '"')
          and raw_lit[-1] == raw_lit[0]
      ):
        opt_name = raw_lit[1:-1]
        if opt_name in subsection_targets and opt_name not in linked_targets:
          opt_id, opt_section = subsection_targets[opt_name]
          if not _is_inside_section(lit_node, opt_section):
            goto_ref = nodes.reference(
                '',
                '',
                internal=True,
                refid=opt_id,
                classes=['torax-goto-link'],
            )
            goto_ref += lit_node.deepcopy()
            lit_node.parent.replace(lit_node, goto_ref)
            linked_targets.add(opt_name)

  for dl_node in doctree.findall(nodes.definition_list):
    for item_node in dl_node.children:
      if isinstance(item_node, nodes.definition_list_item):
        def_node = _dl_item_definition(item_node)
        if def_node is not None:
          _link_container(def_node)

  for b_list in doctree.findall(nodes.bullet_list):
    curr_anc: nodes.Node | None = b_list.parent
    in_dl = False
    while curr_anc is not None:
      if isinstance(curr_anc, nodes.definition_list):
        in_dl = True
        break
      curr_anc = curr_anc.parent
    if not in_dl:
      _link_container(b_list)


def _nest_config_sections(
    doctree: nodes.document,
    section_depths: Mapping[int, int],
    section_orig_titles: Mapping[int, str],
    section_schema: Mapping[int, tuple[list[str], tuple[str, str] | None]],
) -> None:
  """Nests sub-dict and variant sections into parent `<dl>` rows."""
  all_sections = sorted(
      doctree.findall(nodes.section),
      key=lambda s: section_depths.get(id(s), 0),
      reverse=True,
  )
  for section in all_sections:
    if id(section) not in section_schema or section.parent is None:
      continue
    parent_sec = section.parent
    if not isinstance(parent_sec, nodes.section):
      continue
    _, discriminator = section_schema[id(section)]
    title_text = section_orig_titles.get(id(section), '')

    if discriminator is None:
      target_item = _find_dl_item_in_section(
          parent_sec, title_text, section_schema, before_child=section
      )
      if target_item is not None:
        parent_sec.remove(section)
        section['classes'].append('torax-is-nested-inline')
        if section.children and isinstance(section.children[0], nodes.title):
          section.children[0]['classes'].append('torax-nested-dict-heading')
        nested_dd = _get_or_create_nested_dd(
            target_item, 'torax-config-nested-dict-row'
        )
        nested_dd += section
    else:
      disc_field, _ = discriminator
      target_item = _find_dl_item_in_section(
          parent_sec, disc_field, section_schema, before_child=section
      )
      if target_item is not None:
        parent_sec.remove(section)
        section['classes'].append('torax-is-nested-inline')
        nested_dd = _get_or_create_nested_dd(
            target_item, 'torax-config-nested-variants-row'
        )
        nested_dd += section


def _build_variant_tabs(
    doctree: nodes.document,
    section_schema: Mapping[int, tuple[list[str], tuple[str, str] | None]],
) -> None:
  """Converts nested `.torax-config-variant-section` blocks into CSS tabs."""
  tab_set_counter = 0
  tab_item_counter = 0

  for item_node in list(doctree.findall(nodes.definition_list_item)):
    nested_dd = next(
        (
            c
            for c in item_node.children
            if isinstance(c, nodes.definition)
            and 'torax-config-nested-row' in c.get('classes', [])
        ),
        None,
    )
    if nested_dd is None:
      continue

    variant_container: nodes.Element | None = None
    if any(
        isinstance(c, nodes.section)
        and 'torax-config-variant-section' in c.get('classes', [])
        for c in nested_dd.children
    ):
      variant_container = nested_dd
    elif (
        len(nested_dd.children) == 1
        and isinstance(nested_dd.children[0], nodes.section)
        and any(
            isinstance(c, nodes.section)
            and 'torax-config-variant-section' in c.get('classes', [])
            for c in nested_dd.children[0].children
        )
    ):
      variant_container = nested_dd.children[0]

    if variant_container is None:
      continue

    var_secs = [
        c
        for c in variant_container.children
        if isinstance(c, nodes.section)
        and 'torax-config-variant-section' in c.get('classes', [])
    ]
    if not var_secs:
      continue

    opt_order: list[str] = []
    default_disc_field = 'model_name'
    for s in var_secs:
      _, disc_tuple = section_schema.get(id(s), ([], None))
      if disc_tuple is not None:
        default_disc_field = disc_tuple[0]
        if disc_tuple[1] not in opt_order:
          opt_order.append(disc_tuple[1])

    set_id = f'torax-tab-set-{tab_set_counter}'
    tab_set_counter += 1
    tab_set = nodes.container(
        classes=[
            'torax-variant-tab-set',
            'torax-config-variant-section',
        ]
    )

    for idx, var_sec in enumerate(var_secs):
      _, disc_tuple = section_schema.get(id(var_sec), ([], None))
      if disc_tuple is None:
        continue
      disc_field, disc_val = disc_tuple

      input_id = f'torax-tab-item-{tab_item_counter}'
      tab_item_counter += 1
      input_node = _ToraxTabInput(
          '',
          ids=[input_id],
          classes=['torax-tab-input'],
          set_id=set_id,
          checked=(idx == 0),
      )

      label_ids: list[str] = []
      for sec_id in var_sec.get('ids', []):
        if sec_id not in label_ids:
          label_ids.append(sec_id)
      if not label_ids:
        label_ids.append(nodes.make_id(disc_val))

      label_text = disc_val if disc_val == 'None' else f"'{disc_val}'"
      label_node = _ToraxTabLabel(
          '',
          label_text,
          ids=label_ids,
          classes=['torax-tab-label'],
          input_id=input_id,
      )
      content_node = nodes.container(classes=['torax-variant-tab-content'])

      content_children = [
          c for c in var_sec.children if not isinstance(c, nodes.title)
      ]
      first_dl_idx = next(
          (
              i
              for i, c in enumerate(content_children)
              if isinstance(c, nodes.definition_list)
          ),
          len(content_children),
      )
      desc_children: list[nodes.Node] = list(content_children[:first_dl_idx])
      trailing_nodes = content_children[first_dl_idx:]

      # Strip redundant `disc_field` (`model_name`, `solver_type`, etc.) from
      # the variant's parameter `<dl>`.
      cleaned_trailing: list[nodes.Node] = []
      for t_node in trailing_nodes:
        if isinstance(t_node, nodes.definition_list):
          for sub_item in list(t_node.children):
            if isinstance(sub_item, nodes.definition_list_item):
              sub_param = _get_item_param_name(sub_item)
              if sub_param == disc_field:
                if not desc_children:
                  sub_dd = _dl_item_definition(sub_item)
                  if sub_dd is not None:
                    desc_children.extend(
                        sc.deepcopy() for sc in sub_dd.children
                    )
                t_node.remove(sub_item)
          if t_node.children:
            cleaned_trailing.append(t_node)
        else:
          cleaned_trailing.append(t_node)

      if not desc_children:
        desc_children.append(
            nodes.paragraph(
                '', f"Model configuration when {disc_field} = '{disc_val}'."
            )
        )

      for dc in desc_children:
        content_node += dc
      for ct in cleaned_trailing:
        content_node += ct

      tab_set += input_node
      tab_set += label_node
      tab_set += content_node
      variant_container.remove(var_sec)

    if (
        variant_container is nested_dd
        and _get_item_param_name(item_node) == default_disc_field
        and opt_order
    ):
      _rewrite_meta_type_to_literal(
          item_node, opt_order, merge_existing=True
      )
    variant_container += tab_set


def _enhance_config_hierarchy(
    app: application.Sphinx,
    doctree: nodes.document,
    docname: str,
) -> None:
  """Adds hierarchy classes, dict/discriminator paths, and variant tabs."""
  if not docname.startswith('configuration/'):
    _link_time_varying_meta_tokens(app, doctree, docname)
    return

  section_orig_titles: dict[int, str] = {}
  section_depths: dict[int, int] = {}
  section_schema: dict[int, tuple[list[str], tuple[str, str] | None]] = {}
  subsection_targets: dict[str, tuple[str, nodes.section]] = {}

  for section in doctree.findall(nodes.section):
    if not section.children or not isinstance(section.children[0], nodes.title):
      continue
    title_text = section.children[0].astext().strip()
    section_orig_titles[id(section)] = title_text
    depth = len(_section_ancestors(section))
    section_depths[id(section)] = depth

    if depth >= 1:
      section['classes'].append('torax-config-section')
      section['classes'].append(f'torax-config-depth-{min(depth, 3)}')
      if (
          _CONFIG_SECTION_KEY_RE.match(title_text)
          and section.get('ids')
          and title_text not in subsection_targets
      ):
        subsection_targets[title_text] = (section['ids'][0], section)

  for section in doctree.findall(nodes.section):
    if not section.children or not isinstance(section.children[0], nodes.title):
      continue
    title_node = section.children[0]
    title_text = section_orig_titles.get(id(section), '')
    if not _CONFIG_SECTION_KEY_RE.match(title_text):
      continue
    ancestors = _section_ancestors(section)
    ancestor_parts = [
        section_orig_titles[id(anc)]
        for anc in ancestors
        if _CONFIG_SECTION_KEY_RE.match(section_orig_titles.get(id(anc), ''))
    ]
    if not ancestor_parts:
      continue

    dict_keys, discriminator = _resolve_section_schema(
        [*ancestor_parts, title_text]
    )
    section_schema[id(section)] = (dict_keys, discriminator)
    title_node.clear()
    title_node['classes'].append('torax-config-heading')

    if discriminator is not None:
      disc_field, disc_val = discriminator
      section['classes'].append('torax-config-variant-section')
      title_node['classes'].append('torax-config-variant-heading')
      badge = nodes.inline(classes=['torax-heading-discriminator'])
      badge += nodes.inline(
          text=f'{disc_field} = ', classes=['torax-disc-field']
      )
      badge += nodes.inline(text=f"'{disc_val}'", classes=['torax-disc-val'])
      title_node += badge
    else:
      fmt_key = lambda k: f'[{k}]' if k.startswith('<') else f'["{k}"]'
      section['classes'].append('torax-config-dict-section')
      title_node += nodes.inline(
          text=''.join(fmt_key(k) for k in dict_keys[:-1]),
          classes=['torax-heading-path'],
      )
      title_node += nodes.inline(
          text=fmt_key(dict_keys[-1]), classes=['torax-heading-key']
      )

  _link_goto_option_literals(doctree, subsection_targets)

  for parent_el in list(doctree.findall(nodes.section)):
    _merge_adjacent_config_dls(parent_el)

  # Union sibling `Literal` values onto parent-section discriminator rows when
  # the section resolves to a discriminated union in `ToraxConfig`.
  for sec_node in doctree.findall(nodes.section):
    chain_secs = [*_section_ancestors(sec_node), sec_node]
    key_parts = [
        section_orig_titles[id(s)]
        for s in chain_secs
        if _CONFIG_SECTION_KEY_RE.match(section_orig_titles.get(id(s), ''))
    ]
    if not key_parts:
      continue
    non_key_titles = [
        section_orig_titles.get(id(s), '')
        for s in chain_secs
        if not _CONFIG_SECTION_KEY_RE.match(section_orig_titles.get(id(s), ''))
    ]
    _, sec_disc, cand_models = _resolve_models_and_keys(key_parts)
    if sec_disc is not None:
      continue
    disc_info = _collect_discriminator_literals(cand_models)
    if disc_info is None:
      continue
    disc_field, lit_vals = disc_info
    non_key_lower = ' '.join(non_key_titles).lower()
    if any(
        v.lower() in non_key_lower
        or v.lower().replace('_', ' ') in non_key_lower
        for v in lit_vals
    ):
      continue
    for child in sec_node.children:
      if isinstance(child, nodes.definition_list):
        for item_node in child.children:
          if (
              isinstance(item_node, nodes.definition_list_item)
              and _get_item_param_name(item_node) == disc_field
          ):
            _rewrite_meta_type_to_literal(item_node, lit_vals)

  _nest_config_sections(
      doctree, section_depths, section_orig_titles, section_schema
  )
  _build_variant_tabs(doctree, section_schema)

  for section in doctree.findall(nodes.section):
    _merge_adjacent_config_dls(section)

  _link_time_varying_meta_tokens(app, doctree, docname)


class _ToraxTabInput(nodes.Element, nodes.General):
  """Radio `<input>` node for pure-CSS model variant tabs."""


class _ToraxTabLabel(nodes.TextElement, nodes.General):
  """Tab `<label>` node for pure-CSS model variant tabs."""


def _visit_torax_tab_input(self: Any, node: _ToraxTabInput) -> None:
  """Emits opening HTML `<input type="radio">` tag for a variant tab."""
  attrs: dict[str, Any] = {
      'type': 'radio',
      'name': node['set_id'],
  }
  if node.get('checked'):
    attrs['checked'] = 'checked'
  self.body.append(self.starttag(node, 'input', suffix='', **attrs))


def _depart_torax_tab_input(self: Any, node: _ToraxTabInput) -> None:
  """No-op depart visitor for void `<input>` element."""
  del self, node


def _visit_torax_tab_label(self: Any, node: _ToraxTabLabel) -> None:
  """Emits opening HTML `<label>` tag for a variant tab."""
  self.body.append(
      self.starttag(node, 'label', suffix='', **{'for': node['input_id']})
  )


def _depart_torax_tab_label(self: Any, node: _ToraxTabLabel) -> None:
  """Emits closing HTML `</label>` tag for a variant tab."""
  del node
  self.body.append('</label>\n')


_TAB_HASH_SYNC_JS = """\
(function() {
  function syncTabFromHash(hash) {
    if (!hash || hash.length <= 1) return;
    var el = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!el) return;
    var panel = el.closest('.torax-variant-tab-content');
    var label = el.classList.contains('torax-tab-label')
      ? el
      : (panel && panel.previousElementSibling);
    var inp = label && label.previousElementSibling;
    if (inp && inp.classList.contains('torax-tab-input')) {
      inp.checked = true;
    }
  }
  window.addEventListener('hashchange', function() {
    syncTabFromHash(window.location.hash);
  });
  document.addEventListener('DOMContentLoaded', function() {
    syncTabFromHash(window.location.hash);
    document.addEventListener('click', function(e) {
      var t = e.target;
      var a = t && t.closest ? t.closest('a[href^="#"]') : null;
      if (a) syncTabFromHash(a.getAttribute('href'));
    });
  });
})();
"""


def setup(app: application.Sphinx) -> dict[str, Any]:
  """Registers the `torax_config` Sphinx directive and hooks."""
  app.add_node(
      _ToraxTabInput, html=(_visit_torax_tab_input, _depart_torax_tab_input)
  )
  app.add_node(
      _ToraxTabLabel, html=(_visit_torax_tab_label, _depart_torax_tab_label)
  )
  app.add_js_file(None, body=_TAB_HASH_SYNC_JS)
  app.add_directive('torax_config', ToraxConfigDirective)
  if 'autodoc-process-docstring' not in app.events.events:
    app.setup_extension('sphinx.ext.autodoc')
  app.connect(
      'autodoc-process-docstring', _process_autodoc_docstring, priority=400
  )
  app.connect('source-read', _process_source_read)
  app.connect('doctree-resolved', _enhance_config_hierarchy)
  return {
      'version': '0.1',
      'parallel_read_safe': True,
      'parallel_write_safe': True,
  }
