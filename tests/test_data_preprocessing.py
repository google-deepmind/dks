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

"""Tests the data preprocessing module for each framework."""

import functools

from absl.testing import absltest
from dks.jax import data_preprocessing as data_preprocessing_jax
from dks.pytorch import data_preprocessing as data_preprocessing_pytorch
from dks.tensorflow import data_preprocessing as data_preprocessing_tf

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch


class DataPreprocessingTest(absltest.TestCase):
  """Test class for this module."""

  def _run_pln_test(self, module, to_framework, to_numpy, places):
    """Test PLN function outputs the correct values."""

    def assert_almost_equal(x, y):
      np.testing.assert_almost_equal(x, y, decimal=places)

    def pln(x, **kwargs):
      return to_numpy(module.per_location_normalization(
          to_framework(x), **kwargs))

    def check_is_normalized(x):
      assert_almost_equal(np.mean(np.square(x), axis=-1), np.ones(x.shape[:-1]))

    np.random.seed(123)

    shape_list = {(2, 5, 7, 2), (5, 6, 1), (3, 4), (8,), (1, 1), (1, 10)}

    for shape in shape_list:
      for homog_mode in {"one", "off", "avg_q"}:
        for homog_scale in {1.0, 3.2}:
          for has_batch_dim in {True, False}:

            if (homog_mode == "avg_q" and
                ((len(shape) <= 2 and has_batch_dim)
                 or (len(shape) <= 1 and not has_batch_dim))):  # pylint: disable=g-explicit-length-test

              with self.assertRaises(ValueError):
                y = pln(np.random.normal(size=shape), homog_mode=homog_mode,
                        homog_scale=homog_scale, has_batch_dim=has_batch_dim)

            else:

              y = pln(np.random.normal(size=shape), homog_mode=homog_mode,
                      homog_scale=homog_scale, has_batch_dim=has_batch_dim)

              check_is_normalized(y)

              expected_y_shape = list(shape).copy()

              if has_batch_dim and len(expected_y_shape) == 1:
                expected_y_shape = expected_y_shape + [1]

              if homog_mode != "off":
                expected_y_shape[-1] = expected_y_shape[-1] + 1

              assert y.shape == tuple(expected_y_shape)

    y = pln(0.3 * np.ones(()), homog_mode="one", has_batch_dim=False)
    assert_almost_equal(y, np.sqrt(2 / (1 + 0.3**2)) * np.array([0.3, 1.0]))

    with self.assertRaises(ValueError):
      pln(np.random.normal(size=()), homog_mode="avg_q", has_batch_dim=True)

    y = pln(0.7 * np.ones((10, 6, 3)), homog_mode="off")
    assert_almost_equal(y, np.ones((10, 6, 3)))

    y = pln(0.7 * np.ones((10, 6, 3)), homog_mode="one", homog_scale=2.5)
    assert_almost_equal(
        y, np.sqrt(4 / (3 + (2.5 / 0.7) ** 2)) * np.concatenate(
            [np.ones((10, 6, 3)), 2.5 / 0.7 * np.ones((10, 6, 1))], axis=-1))

    y = pln(0.7 * np.ones((10, 6, 3)), homog_mode="avg_q")
    assert_almost_equal(y, np.ones((10, 6, 4)))

    y = pln(0.7 * np.ones((10, 6, 3)), homog_mode="avg_q", homog_scale=2.0)
    assert_almost_equal(
        y, np.sqrt(4 / 7) * np.concatenate(
            [np.ones((10, 6, 3)), 2 * np.ones((10, 6, 1))], axis=-1))

  def test_per_location_normalization_jax(self):
    self._run_pln_test(
        data_preprocessing_jax, jnp.asarray, np.asarray, 5)

  def test_per_location_normalization_pytorch(self):
    self._run_pln_test(
        data_preprocessing_pytorch, torch.tensor,
        lambda x: x.detach().cpu().numpy(), 5)

  def test_per_location_normalization_tensorflow(self):
    self._run_pln_test(
        data_preprocessing_tf,
        functools.partial(tf.convert_to_tensor, dtype=tf.float32),
        lambda x: x.numpy(), 5)


if __name__ == "__main__":
  absltest.main()
