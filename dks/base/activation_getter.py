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

"""Defines a getter for NumPy activation functions (with autograd support)."""

from autograd import numpy as np
from autograd import scipy as sp


_SELU_LAMBDA = 1.0507009873554804934193349852946
_SELU_ALPHA = 1.6732632423543772848170429916717


_sigmoid = sp.special.expit
_erf = sp.special.erf


def _elu(x, a=1.0, l=1.0):
  is_neg = x < 0
  is_not_neg = np.logical_not(is_neg)
  return l * (is_neg * a * (np.exp(x) - 1) + is_not_neg * x)


def _bentid(x):
  return (np.sqrt(x**2 + 1.) - 1.) / 2. + x


def _softplus(x):
  """Numerically-stable softplus."""
  return np.log(1. + np.exp(-np.abs(x))) + np.maximum(x, 0)


# aka Silu
def _swish(x):
  return x * _sigmoid(x)


def _leaky_relu(x, negative_slope=0.01):
  is_neg = x < 0
  is_not_neg = np.logical_not(is_neg)
  return negative_slope * is_neg * x + is_not_neg * x


_ACTIVATION_TABLE = {
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "erf": _erf,
    "relu": lambda x: np.maximum(0., x),
    "softplus": _softplus,
    "selu": lambda x: _elu(x, _SELU_ALPHA, _SELU_LAMBDA),
    "elu": _elu,
    "swish": _swish,
    "bentid": _bentid,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "square": lambda x: x**2,
    "softsign": lambda x: x / (1 + np.abs(x)),
    "leaky_relu": _leaky_relu,
}


def get_activation_function(name):
  if name in _ACTIVATION_TABLE:
    return _ACTIVATION_TABLE[name]
  else:
    raise ValueError(f"Unrecognized activation function name '{name}'.")
