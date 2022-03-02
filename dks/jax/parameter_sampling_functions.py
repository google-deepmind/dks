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

"""Sampling functions for parameter initialization in DKS/TAT with JAX."""

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as np


def _get_default_float_dtype():
  if jax_config.jax_enable_x64:
    return jnp.float64
  else:
    return jnp.float32


def _uniform_orthogonal(key, shape, scale=1.0, axis=-1, dtype=None):
  """Samples an orthogonal matrix from the uniform/Haar distribution."""

  # The implementation of this function is essentially copied from
  # hk.initializers.Orthogonal.

  if dtype is None:
    dtype = _get_default_float_dtype()

  if len(shape) < 2:
    raise ValueError("Orthogonal initializer requires at least a 2D shape.")

  n_rows = shape[axis]
  n_cols = np.prod(shape) // n_rows
  matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)

  norm_dst = jax.random.normal(key, matrix_shape, dtype)

  q_mat, r_mat = jnp.linalg.qr(norm_dst)
  # Enforce Q is uniformly distributed
  q_mat *= jnp.sign(jnp.diag(r_mat))
  if n_rows < n_cols:
    q_mat = q_mat.T
  q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, axis)))
  q_mat = jnp.moveaxis(q_mat, 0, axis)

  return jax.lax.convert_element_type(scale, dtype) * q_mat


def scaled_uniform_orthogonal(key, shape, scale=1.0, axis=-1, delta=True,
                              dtype=None):
  """Initializes fully-connected or conv weights using the SUO distribution.

  Output is similar to that of hk.initializers.Orthogonal, except that it
  supports Delta initializations, and sampled weights are rescaled by
  ``max(sqrt(out_dim / in_dim), 1)``, so that the layer preserves q values at
  initialization-time (assuming initial biases of zero).

  Note that as with all JAX functions, this is pure and totally stateless
  function, and will produce the exact same output given the same arguments.

  Should be used with a zeros initializer for the bias parameters for DKS/TAT.

  See "Parameter distributions" section of DKS paper
  (https://arxiv.org/abs/2110.01765) for a discussion of the SUO distribution
  and Delta initializations.

  Args:
    key: A PRNG key used as the random key.
    shape: A list of integers giving the shape of the parameter tensor.
    scale: A float giving an additional scale factor applied on top of the
      standard recaling used in the SUO distribution. This should be left
      at its default value when using DKS/TAT. (Default: 1.0)
    axis: An int giving the axis corresponding to the "output dimension" of
      the parameter tensor. (Default: -1)
    delta: A bool determining whether or not to use a Delta initialization
      (which zeros out all weights except those in the central location of
      convolutional filter banks). (Default: True)
    dtype: a float dtype for the return value. (Default: jnp.float64 if
      jax_enable_x64 is true, otherwise jnp.float32).

  Returns:
    The sampled weights as a JAX ndarray.
  """

  if delta and axis != -1:
    raise ValueError("Invalid axis value for Delta initializations. "
                     "Must be -1.")

  if delta and len(shape) != 2:
    # We assume 'weights' is a filter bank when len(shape) != 2

    # In JAX, conv filter banks have the shape
    # [loc_dim_1, loc_dim_2, in_dim, out_dim]
    in_dim = shape[-2]
    out_dim = shape[-1]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    nonzero_part = _uniform_orthogonal(
        key, shape[-2:], scale=(rescale_factor * scale), axis=-1, dtype=dtype)

    if any(s % 2 != 1 for s in shape[:-2]):
      raise ValueError("All spatial axes must have odd length for Delta "
                       "initializations.")

    midpoints = tuple((s - 1) // 2 for s in shape[:-2])

    return jnp.zeros(shape, dtype).at[midpoints].set(nonzero_part)

  else:

    in_dim = np.prod(np.delete(shape, axis))
    out_dim = shape[axis]

    rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

    return _uniform_orthogonal(
        key, shape, scale=(rescale_factor * scale), axis=axis, dtype=dtype)


