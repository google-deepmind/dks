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

"""Tests the parameter sampling functions for each framework."""

from absl.testing import absltest
from absl.testing import parameterized

from dks.jax import parameter_sampling_functions as parameter_sampling_functions_jax
from dks.pytorch import parameter_sampling_functions as parameter_sampling_functions_pytorch
from dks.tensorflow import parameter_sampling_functions as parameter_sampling_functions_tf

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch


tf.compat.v1.enable_eager_execution()  # enable eager in case we are using TF1


class ParameterSamplingFunctionTest(parameterized.TestCase):
  """Perform some basic sanity checks for the parameter sampling functions.

  For each tensor programming framework, we test whether multiplication of the
  sampled parameters by random vectors behaves as expected regarding
  (approximate) empirical q value preservation. Note that only when the input
  dimension is smaller than the output dimension will empirical q values be
  exactly preserved (up to numerical precision).
  """

  @parameterized.parameters((5, 7, 9, 3, 5), (3, 5, 9, 9, 5), (8, 8, 5, 7, 5),
                            (3072, 1024, 5, 3, 1))
  def test_parameter_sampling_functions_jax(
      self, in_channels, out_channels, dim1, dim2, places):

    w = parameter_sampling_functions_jax.scaled_uniform_orthogonal(
        jnp.array([1, 2], dtype=jnp.uint32), [in_channels, out_channels])

    x = jax.random.normal(jnp.array([2, 3], dtype=jnp.uint32), [in_channels, 1])

    expected_rq = np.sqrt(out_channels/in_channels)

    self.assertAlmostEqual(
        jnp.linalg.norm(jnp.matmul(w.T, x)) / jnp.linalg.norm(x),
        expected_rq, places=places)

    w = parameter_sampling_functions_jax.scaled_uniform_orthogonal(
        jnp.array([3, 4], dtype=jnp.uint32),
        (dim1, dim2, in_channels, out_channels))

    self.assertAlmostEqual(
        (jnp.linalg.norm(
            jnp.matmul(w[(dim1-1) // 2, (dim2-1) // 2, :, :].T, x))
         / jnp.linalg.norm(x)),
        expected_rq, places=places)

    self.assertAlmostEqual(jnp.linalg.norm(
        jnp.matmul(w[(dim1-1) // 2 + 1, (dim2-1) // 2, :, :].T, x)),
                           0.0, places=places)

    self.assertAlmostEqual(jnp.linalg.norm(
        jnp.matmul(w[(dim1-1) // 2, (dim2-1) // 2 + 1, :, :].T, x)),
                           0.0, places=places)

    self.assertAlmostEqual(jnp.linalg.norm(
        jnp.matmul(w[(dim1-1) // 2 + 1, (dim2-1) // 2 + 1, :, :].T, x)),
                           0.0, places=places)

    if in_channels <= 200:
      out_channels *= dim1 * dim2

      w = parameter_sampling_functions_jax.scaled_uniform_orthogonal(
          jnp.array([4, 5], dtype=jnp.uint32),
          (dim1, dim2, in_channels, out_channels), delta=False)

      x = jax.random.normal(jnp.array([5, 6], dtype=jnp.uint32),
                            [dim1 * dim2 * in_channels, 1])

      self.assertAlmostEqual(
          (jnp.linalg.norm(jnp.matmul(jnp.reshape(
              w, [dim1 * dim2 * in_channels, out_channels]).T, x)
                           ) / jnp.linalg.norm(x)),
          np.sqrt(out_channels / (dim1 * dim2 * in_channels)), places=places)

  @parameterized.parameters((5, 7, 9, 3, 5), (3, 5, 9, 9, 5), (8, 8, 5, 7, 5),
                            (3072, 1024, 5, 3, 1))
  def test_parameter_sampling_functions_tensorflow(
      self, in_channels, out_channels, dim1, dim2, places):

    w = parameter_sampling_functions_tf.stateless_scaled_uniform_orthogonal(
        [in_channels, out_channels], [1, 2])

    x = tf.random.stateless_normal([in_channels, 1], [2, 3])

    expected_rq = np.sqrt(out_channels/in_channels)

    self.assertAlmostEqual(
        (tf.norm(tf.matmul(w, x, transpose_a=True)) / tf.norm(x)).numpy(),
        expected_rq, places=places)

    w = parameter_sampling_functions_tf.stateless_scaled_uniform_orthogonal(
        (dim1, dim2, in_channels, out_channels),
        [3, 4])

    self.assertAlmostEqual(
        (tf.norm(
            tf.matmul(w[(dim1-1) // 2, (dim2-1) // 2, :, :], x,
                      transpose_a=True))
         / tf.norm(x)).numpy(),
        expected_rq, places=places)

    self.assertAlmostEqual(
        tf.norm(tf.matmul(w[(dim1-1) // 2 + 1, (dim2-1) // 2, :, :], x,
                          transpose_a=True)).numpy(),
        0.0, places=places)

    self.assertAlmostEqual(
        tf.norm(tf.matmul(w[(dim1-1) // 2, (dim2-1) // 2 + 1, :, :], x,
                          transpose_a=True)).numpy(),
        0.0, places=places)

    self.assertAlmostEqual(
        tf.norm(tf.matmul(w[(dim1-1) // 2 + 1, (dim2-1) // 2 + 1, :, :], x,
                          transpose_a=True)).numpy(),
        0.0, places=places)

    if in_channels <= 200:
      out_channels *= dim1 * dim2

      w = parameter_sampling_functions_tf.stateless_scaled_uniform_orthogonal(
          (dim1, dim2, in_channels, out_channels), [4, 5], delta=False)

      x = tf.random.stateless_normal([dim1 * dim2 * in_channels, 1], [5, 6])

      self.assertAlmostEqual(
          (tf.norm(tf.matmul(tf.reshape(
              w, [dim1 * dim2 * in_channels, out_channels]), x,
                             transpose_a=True))
           / tf.norm(x)).numpy(),
          np.sqrt(out_channels / (dim1 * dim2 * in_channels)), places=places)

  @parameterized.parameters((5, 7, 9, 3, 5), (3, 5, 9, 9, 5), (8, 8, 5, 7, 5),
                            (3072, 1024, 5, 3, 1))
  def test_parameter_sampling_functions_pytorch(
      self, in_channels, out_channels, dim1, dim2, places):

    torch.manual_seed(123)

    def to_np(z):
      return z.detach().cpu().numpy()

    w = parameter_sampling_functions_pytorch.scaled_uniform_orthogonal_(
        torch.empty(in_channels, out_channels))

    x = torch.randn(in_channels, 1)

    expected_rq = np.sqrt(out_channels/in_channels)

    self.assertAlmostEqual(
        to_np(torch.norm(torch.matmul(w.t(), x)) / torch.norm(x)),
        expected_rq, places=places)

    w = parameter_sampling_functions_pytorch.scaled_uniform_orthogonal_(
        torch.empty(in_channels, out_channels, dim1, dim2))

    self.assertAlmostEqual(
        to_np(torch.norm(
            torch.matmul(w[:, :, (dim1-1) // 2, (dim2-1) // 2].t(), x))
              / torch.norm(x)),
        expected_rq, places=places)

    self.assertAlmostEqual(to_np(torch.norm(
        torch.matmul(w[:, :, (dim1-1) // 2 + 1, (dim2-1) // 2].t(), x))),
                           0.0, places=places)

    self.assertAlmostEqual(to_np(torch.norm(
        torch.matmul(w[:, :, (dim1-1) // 2, (dim2-1) // 2 + 1].t(), x))),
                           0.0, places=places)

    self.assertAlmostEqual(to_np(torch.norm(
        torch.matmul(w[:, :, (dim1-1) // 2 + 1, (dim2-1) // 2 + 1].t(), x))),
                           0.0, places=places)

    if in_channels <= 200:
      out_channels *= dim1 * dim2

      w = parameter_sampling_functions_pytorch.scaled_uniform_orthogonal_(
          torch.empty(in_channels, out_channels, dim1, dim2), delta=False)

      x = torch.randn(dim1 * dim2 * in_channels, 1)

      self.assertAlmostEqual(
          to_np(torch.norm(torch.matmul(torch.reshape(
              w.transpose(0, 1), [out_channels, dim1 * dim2 * in_channels]), x)
                           ) / torch.norm(x)),
          np.sqrt(out_channels / (dim1 * dim2 * in_channels)), places=places)


if __name__ == "__main__":
  absltest.main()
