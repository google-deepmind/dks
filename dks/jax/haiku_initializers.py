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

"""Haiku initializers for use with DKS/TAT."""

from dks.jax import parameter_sampling_functions
import haiku as hk


class ScaledUniformOrthogonal(hk.initializers.Initializer):
  """SUO (+ Delta) initializer for fully-connected and convolutional layers.

  Similar to hk.initializers.Orthogonal, except that it supports Delta
  initializations, and sampled weights are rescaled by
  ``max(sqrt(out_dim / in_dim), 1)`` so that the layer preserves q values at
  initialization time (assuming initial biases of zero).

  Should be used with a zeros initializer for the bias parameters for DKS/TAT.

  See "Parameter distributions" section of DKS paper
  (https://arxiv.org/abs/2110.01765) for a discussion of the SUO distribution
  and Delta initializations.
  """

  def __init__(self, scale=1.0, axis=-1, delta=True):
    """Construct a Haiku initializer which uses the SUO distribution.

    Args:
      scale: A float giving an additional scale factor applied on top of the
        standard rescaling used in the SUO distribution. This should be left
        at its default value when using DKS/TAT. (Default: 1.0)
      axis: An int giving the axis corresponding to the "output dimension" of
        the parameter tensor. (Default: -1)
      delta: A bool determining whether or not to use a Delta initialization
        (which zeros out all weights except those in the central location of
        convolutional filter banks). (Default: True)
    """

    if delta and axis != -1:
      raise ValueError("Invalid axis value for Delta initializations. "
                       "Must be -1.")
    self.scale = scale
    self.axis = axis
    self.delta = delta

  def __call__(self, shape, dtype):

    return parameter_sampling_functions.scaled_uniform_orthogonal(
        hk.next_rng_key(), shape, scale=self.scale, axis=self.axis,
        delta=self.delta, dtype=dtype)

