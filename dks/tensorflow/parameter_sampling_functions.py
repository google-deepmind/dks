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

"""Sampling functions for parameter initialization in DKS/TAT with TF."""

import numpy as np
import tensorflow as tf


def _stateless_uniform_orthogonal(shape, seed, gain=1.0,
                                  dtype=tf.dtypes.float32):
  """Samples an orthogonal matrix from the uniform/Haar distribution."""

  # The implementation of this function is essentially copied from
  # tf.initializers.Orthogonal.

  # Check the shape
  if len(shape) < 2:
    raise ValueError("The tensor to initialize must be "
                     "at least two-dimensional. Received: "
                     f"shape={shape} of rank {len(shape)}.")
  # Flatten the input shape with the last dimension remaining
  # its original shape so it works for conv2d
  num_rows = 1
  for dim in shape[:-1]:
    num_rows *= dim
  num_cols = shape[-1]
  flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

  a = tf.random.stateless_normal(flat_shape, seed, dtype=dtype)

  # Compute the qr factorization
  q, r = tf.linalg.qr(a, full_matrices=False)

  # Make Q uniform
  d = tf.linalg.tensor_diag_part(r)
  q *= tf.sign(d)

  if num_rows < num_cols:
    q = tf.linalg.matrix_transpose(q)

  return gain * tf.reshape(q, shape)


def stateless_scaled_uniform_orthogonal(
    shape, seed, gain=1.0, delta=True, dtype=tf.dtypes.float32):
  """Initializes fully-connected or conv weights using the SUO distribution.

  Similar to a stateess functional version of tf.initializers.Orthogonal, except
  except that it supports Delta initializations, and sampled weights are
  rescaled by ``max(sqrt(out_dim / in_dim), 1)``, so that the layer preserves
  q values at initialization-time (assuming initial biases of zero).

  Note that this is a stateless function, and will produce the exact same output
  given the same arguments. A stateful random op can be created by passing the
  output of some stateful random op as the ``seed`` argument.

  Should be used with a zeros initializer for the bias parameters for DKS/TAT.

  See "Parameter distributions" section of DKS paper
  (https://arxiv.org/abs/2110.01765) for a discussion of the SUO distribution
  and Delta initializations.

  Args:
    shape: A list of integers giving the shape of the parameter tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype tf.int32 or tf.int64. (When using XLA, only tf.int32 is allowed.)
    gain: A float giving an additional scale factor applied on top of the
      standard recaling used in the SUO distribution. This should be left
      at its default value when using DKS/TAT. (Default: 1.0)
    delta: A bool determining whether or not to use a Delta initialization
      (which zeros out all weights except those in the central location of
      convolutional filter banks). (Default: True)
    dtype: a float dtype for the return value. (Default: jnp.float64 if
      jax_enable_x64 is true, otherwise jnp.float32).

  Returns:
    The sampled weights as a TF Tensor.
  """

  if delta and len(shape) != 2:
    # We assume 'weights' is a filter bank when len(shape) != 2

    # In TensorFlow, conv filter banks have the shape
    # [loc_dim_1, loc_dim_2, in_dim, out_dim]
    in_dim = shape[-2]
    out_dim = shape[-1]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    nonzero_part = _stateless_uniform_orthogonal(
        shape[-2:], seed, gain=(rescale_factor * gain), dtype=dtype)

    if any(s % 2 != 1 for s in shape[:-2]):
      raise ValueError("All spatial axes must have odd length for Delta "
                       "initializations.")

    midpoints = tuple((s - 1) // 2 for s in shape[:-2])

    return tf.scatter_nd((midpoints,), (nonzero_part,), shape)

  else:

    in_dim = np.prod(shape[:-1])
    out_dim = shape[-1]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    return _stateless_uniform_orthogonal(
        shape, seed, gain=(rescale_factor * gain), dtype=dtype)
