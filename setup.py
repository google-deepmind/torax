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

"""Install script for setuptools."""

import os
from setuptools import setup  # pylint: disable=g-importing-member

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  with open(os.path.join(_CURRENT_DIR, 'torax', '__init__.py')) as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1 :].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `torax/__init__.py`')


def _parse_requirements(path):
  with open(os.path.join(_CURRENT_DIR, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


setup(
    name='torax',
    version=_get_version(),
    license='Apache 2.0',
    author='Google DeepMind',
    description='TORAX',  # TODO(b/323504363): Add desc.
    long_description=open(
        os.path.join(_CURRENT_DIR, 'README.md')
    ).read(),
    long_description_content_type='text/markdown',
    author_email='torax-team@google.com',
    packages=[
        'torax',
        'torax.fvm',
        'torax.sources',
        'torax.spectators',
        'torax.stepper',
        'torax.time_step_calculator',
        'torax.transport_model',
    ],
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements', 'requirements.txt')
    ),
    zip_safe=False,  # Required for full installation.
    include_package_data=True,
    python_requires='>=3.10',
)
