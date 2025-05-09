# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  with open(os.path.join(_CURRENT_DIR, 'dks', '__init__.py')) as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `dks/__init__.py`')


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


_VERSION = _get_version()

setup(
    name='dks',
    version=_VERSION,
    url='https://github.com/deepmind/dks',
    license='Apache 2.0',
    author='DeepMind',
    description=('A Python library implementing the DKS/TAT neural network '
                 'transformation method.'),
    long_description=open(os.path.join(_CURRENT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    author_email='dks-dev@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(exclude=['tests']),
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements.txt')),
    tests_require=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements_tests.txt')),
    extras_require={
        key: _parse_requirements(
            os.path.join(_CURRENT_DIR, f'requirements_{key}.txt'))
        for key in {'jax', 'pytorch', 'tf', 'tests'}
    },
    requires_python='>=3.8',
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
