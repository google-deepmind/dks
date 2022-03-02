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

"""Sampling functions for parameter initialization in DKS/TAT with PyTorch."""

import numpy as np
import torch


def scaled_uniform_orthogonal_(weights, gain=1.0, delta=True):
  """Initializes fully-connected or conv weights using the SUO distribution.

  Similar to torch.nn.init.orthogonal_, except that it supports Delta
  initializations, and sampled weights are rescaled by
  ``max(sqrt(out_dim / in_dim), 1)``, so that the layer preserves q values at
  initialization-time (assuming initial biases of zero).

  Note that as with all PyTorch functions ending with '_', this function
  modifies the value of its tensor argument in-place.

  Should be used with a zeros initializer for the bias parameters for DKS/TAT.

  See "Parameter distributions" section of DKS paper
  (https://arxiv.org/abs/2110.01765) for a discussion of the SUO distribution
  and Delta initializations.

  Args:
    weights: A PyTorch Tensor corresponding to the weights to be randomly
      initialized.
    gain: A float giving an additional scale factor applied on top of the
      standard recaling used in the SUO distribution. This should be left
      at its default value when using DKS/TAT. (Default: 1.0)
    delta: A bool determining whether or not to use a Delta initialization
      (which zeros out all weights except those in the central location of
      convolutional filter banks). (Default: True)

  Returns:
    The ``weights`` argument (whose value will be initialized).
  """

  shape = list(weights.size())

  if delta and len(shape) != 2:
    # We assume 'weights' is a filter bank when len(shape) != 2

    # In PyTorch, conv filter banks have that shape
    # [in_dim, out_dim, loc_dim_1, loc_dim_2]
    in_dim = shape[0]
    out_dim = shape[1]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    nonzero_part = torch.nn.init.orthogonal_(weights.new_empty(in_dim, out_dim),
                                             gain=(rescale_factor * gain))

    if any(s % 2 != 1 for s in shape[2:]):
      raise ValueError("All spatial axes must have odd length for Delta "
                       "initializations.")

    midpoints = [(s - 1) // 2 for s in shape[2:]]
    indices = [slice(None), slice(None)] + midpoints

    with torch.no_grad():
      weights.fill_(0.0)
      weights.__setitem__(indices, nonzero_part)

      return weights

  else:

    # torch.nn.orthogonal_ flattens dimensions [1:] instead of [:-1], which is
    # the opposite of what we want here. So we'll first compute version with
    # the first two dimensions swapped, and then we'll transpose at the end.

    shape = [shape[1], shape[0]] + shape[2:]

    in_dim = np.prod(shape[1:])
    out_dim = shape[0]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    weights_t = torch.nn.init.orthogonal_(weights.new_empty(shape),
                                          gain=(rescale_factor * gain))
    with torch.no_grad():
      return weights.copy_(weights_t.transpose_(0, 1))
