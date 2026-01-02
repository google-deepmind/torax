# Copyright 2024 DeepMind Technologies Limited
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

"""Functions to load a config from config file or directory."""

import importlib
import os
import pathlib
import sys
import types
import typing
from typing import Any, Literal, TypeAlias

from torax._src import path_utils
from torax._src.plotting import plotruns_lib
from torax._src.torax_pydantic import model_config

ExampleConfig: TypeAlias = Literal[
    "basic_config",
    "custom_pedestal_example",
    "iterhybrid_predictor_corrector",
    "iterhybrid_rampup",
    "step_flattop_bgb",
]

ExamplePlotConfig: TypeAlias = Literal[
    "default_plot_config",
    "global_params_plot_config",
    "simple_plot_config",
    "sources_plot_config",
]


def example_config_paths() -> dict[ExampleConfig, pathlib.Path]:
    """Returns a tuple of example config paths."""

    example_dir = path_utils.torax_path().joinpath("examples")
    assert example_dir.is_dir()

    def _get_path(path):
        path = example_dir.joinpath(path + ".py")
        assert path.is_file(), f"Path {path} to the example config does not exist."
        return path

    return {path: _get_path(path) for path in typing.get_args(ExampleConfig)}


def example_plot_config_paths() -> dict[ExamplePlotConfig, pathlib.Path]:
    """Returns a tuple of example plot config paths."""

    example_dir = path_utils.torax_path().joinpath("plotting", "configs")
    assert example_dir.is_dir(), f"Path {example_dir} is not a directory."

    def _get_path(path):
        path = example_dir.joinpath(path + ".py")
        assert path.is_file(), f"Path {path} to the example config does not exist."
        return path

    return {path: _get_path(path) for path in typing.get_args(ExamplePlotConfig)}


# Taken from
# https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def _import_from_path(module_name: str, file_path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if module is None:
        raise ValueError(f"No loader found for module {module_name}.")
    else:
        spec.loader.exec_module(module)  # pytype: disable=attribute-error
    return module


def import_module(path: str | pathlib.Path) -> dict[str, Any]:
    """Import a Python module from a file path as a dictionary.

    Relative paths are first resolved relative to the working directory, and then
    relative to the Torax directory.

    Args:
      path: The absolute or relative path to the Python module. The path can be
        represented as a string or a `pathlib.Path` object.

    Returns:
      All variables in the module as a dictionary.
    """

    path = pathlib.Path(path) if isinstance(path, str) else path

    if not path.is_file():
        # path.is_file() will return True if either path is an absolute path and it
        # exists, or if it is a relative path and it exists relative to the working
        # directory. In this case, neither was successful, so check whether this
        # path exists relative to the Torax directory.
        new_path = path_utils.torax_path().joinpath(path)
        if not new_path.is_file():
            raise ValueError(
                f"The file {path} could not be found. If it is a relative path, it"
                " could not be resolved relative to the working directory"
                f" {os.getcwd()} or the Torax directory {path_utils.torax_path()}."
            )
        path = new_path

    # An arbitrary module name is needed to import the config file.
    arbitrary_module_name = "_torax_temp_config_import"
    module = _import_from_path(arbitrary_module_name, path)
    return vars(module)


def build_torax_config_from_file(
    path: str | pathlib.Path,
) -> model_config.ToraxConfig:
    """Returns a ToraxConfig object from a config file.

    The config file is either a Python file containing a `CONFIG` dictionary, or
    a JSON file obtained via `ToraxConfig(...).model_dump_json`.

    Args:
      path: The absolute path to the config file. The path can be represented as a
        string or a `pathlib.Path` object.
    """

    path = pathlib.Path(path) if isinstance(path, str) else path

    match path.suffix:
        case ".json":
            raise NotImplementedError(
                "Loading ToraxConfig from JSON is not implemented yet."
            )
        case ".nc":
            raise NotImplementedError(
                "Loading ToraxConfig from serialized JSON is not implemented yet."
            )
        case ".py":
            cfg = import_module(path)
            if "CONFIG" not in cfg:
                raise ValueError(
                    f"The file {str(path)} is an invalid Torax config file, as it does"
                    " not have a `CONFIG` variable defined."
                )
            return model_config.ToraxConfig.from_dict(cfg["CONFIG"])
        case _:
            raise ValueError(f"Path {path} is not a valid Torax config file.")


def get_plot_config_from_file(
    path: str | pathlib.Path,
) -> plotruns_lib.FigureProperties:
    """Returns a FigureProperties object from a config file.

    The config file is a Python file with a `PLOT_CONFIG` variable.

    Args:
      path: The absolute path to the config file. If `None`, the default plot
        config file is used.

    Returns:
      A FigureProperties object.
    """

    path = pathlib.Path(path) if isinstance(path, str) else path

    cfg = import_module(path)
    if "PLOT_CONFIG" not in cfg:
        raise ValueError(
            f"The file {str(path)} is an invalid Torax plot config file, as it does"
            " not have a `PLOT_CONFIG` variable defined."
        )
    return cfg["PLOT_CONFIG"]
