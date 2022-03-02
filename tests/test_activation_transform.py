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

"""Tests the activation function transform modules for each framework."""

from absl.testing import absltest

from dks.base import activation_transform as activation_transform_numpy
from dks.jax import activation_transform as activation_transform_jax
from dks.pytorch import activation_transform as activation_transform_pytorch
from dks.tensorflow import activation_transform as activation_transform_tf

import numpy as np
import torch
import tree


def _assert_structure_approx_equal(s1, s2):
  tree.map_structure(np.testing.assert_almost_equal, s1, s2)


def _subnet_max_func(x, r_fn, shortcut_weight=0.6):
  """The subnetwork maximizing function of the modified ResNet model."""

  blocks_per_group = (3, 4, 23, 3)

  for i in range(4):
    for j in range(blocks_per_group[i]):

      res_x = r_fn(r_fn(r_fn(x)))

      shortcut_x = r_fn(x) if (j == 0) else x

      x = (shortcut_weight**2 * shortcut_x + (1.0 - shortcut_weight**2) * res_x)

  x = r_fn(x)

  return x


class ActivationTransformTest(absltest.TestCase):
  """Test class for this module."""

  def test_parameter_computation(self):
    """Test that the correct transformation parameters are found."""

    def check(activation_names, method, expected_params_dict):
      params_dict = activation_transform_numpy._get_activations_params(  # pylint: disable=protected-access
          activation_names=activation_names,
          method=method,
          dks_params={"c_slope_1_target": 1.2},
          tat_params={"c_val_0_target": 0.75, "c_curve_target": 0.2},
          subnet_max_func=_subnet_max_func,
          )
      _assert_structure_approx_equal(params_dict, expected_params_dict)

    check(("softplus",), "DKS",
          {"softplus": {"input_scale": 0.18761008168267976,
                        "input_shift": 0.40688063442262007,
                        "output_shift": -0.9213466151411376,
                        "output_scale": 8.878672543665223}})

    check(("tanh", "softplus"), "DKS",
          {"tanh": {"input_scale": 0.07461073057540868,
                    "input_shift": 0.5566915199964182,
                    "output_shift": -0.5034378768692127,
                    "output_scale": 18.002635964442558},
           "softplus": {"input_scale": 0.18761008168267976,
                        "input_shift": 0.40688063442262007,
                        "output_shift": -0.9213466151411376,
                        "output_scale": 8.878672543665223}})

    check(("relu",), "DKS",
          {"relu": {"input_shift": 1.0,
                    "input_scale": 0.3685360046708044,
                    "output_shift": -1.000373858553784,
                    "output_scale": 2.721729988761473}})

    check(("leaky_relu",), "DKS",
          {"leaky_relu": {"negative_slope": 0.8761257073473065,
                          "input_shift": -6.372924855154692e-13,
                          "output_shift": -0.049418692997275034,
                          "output_scale": 1.0651832705682147}})

    check(("softplus",), "TAT",
          {"softplus": {"input_scale": 0.15011489794748328,
                        "input_shift": 0.5374599901127068,
                        "output_shift": -0.996481014465811,
                        "output_scale": 10.54880187880574}})

    check(("tanh", "softplus"), "TAT",
          {"tanh": {"input_scale": 0.0580205218578313,
                    "input_shift": 0.5218639804099805,
                    "output_shift": -0.4796430704354356,
                    "output_scale": 22.36066261763117},
           "softplus": {"input_scale": 0.15011489794748328,
                        "input_shift": 0.5374599901127068,
                        "output_shift": -0.996481014465811,
                        "output_scale": 10.54880187880574}})

    check(("leaky_relu",), "TAT",
          {"leaky_relu": {"output_scale": 1.196996549778802,
                          "negative_slope": 0.6291800290346146}})

  def _run_value_tests(self, module, to_framework, to_numpy, places):
    """Test that transformed activation functions compute the correct values."""

    val1 = to_framework(0.6)
    val2 = to_framework(-0.6)

    def check(activation_names, method, expected_values_dict):
      act_dict = module.get_transformed_activations(
          activation_names=activation_names,
          method=method,
          dks_params={"c_slope_1_target": 12.0, "local_q_slope_target": 0.97},
          tat_params={"c_val_0_target": 0.95, "c_curve_target": 4.5},
          subnet_max_func=_subnet_max_func
          )
      for name in expected_values_dict:
        self.assertAlmostEqual(
            to_numpy(act_dict[name](val1)), expected_values_dict[name][0],
            places=places)
        self.assertAlmostEqual(
            to_numpy(act_dict[name](val2)), expected_values_dict[name][1],
            places=places)

    check(("softplus",), "DKS",
          {"softplus": [0.5205088334781294, -0.6970897398761904]})

    check(("tanh", "softplus"), "DKS",
          {"tanh": [0.6910746968773931, -0.5122335409369118],
           "softplus": [0.5205088334781294, -0.6970897398761904]})

    check(("relu",), "DKS",
          {"relu": [0.6053339463256114, -0.6486764456443863]})

    check(("leaky_relu",), "DKS",
          {"leaky_relu": [0.5628988987673328, -0.6649764416535503]})

    check(("softplus",), "TAT",
          {"softplus": [0.6923574763139374, -0.4935166367766701]})

    check(("tanh", "softplus"), "TAT",
          {"tanh": [0.6860178437500071, -0.49030275427738146],
           "softplus": [0.6923574763139374, -0.4935166367766701]})

    check(("leaky_relu",), "TAT",
          {"leaky_relu": [0.7954047194402861, -0.2955187511683813]})

    check(("tanh", "softplus", "relu", "leaky_relu"), "untransformed",
          {"tanh": [0.5370495669980353, -0.5370495669980353],
           "softplus": [1.0374879504858856, 0.4374879504858857],
           "relu": [0.6, 0.0],
           "leaky_relu": [0.6, -0.006]})

  def test_transformed_activation_values_numpy(self):
    self._run_value_tests(
        activation_transform_numpy, lambda x: x, lambda x: x, 8)

  def test_transformed_activation_values_jax(self):
    self._run_value_tests(
        activation_transform_jax, lambda x: x, lambda x: x, 5)

  def test_transformed_activation_values_pytorch(self):
    self._run_value_tests(
        activation_transform_pytorch, torch.tensor,
        lambda x: x.detach().cpu().numpy(), 5)

  def test_transformed_activation_values_tensorflow(self):
    self._run_value_tests(
        activation_transform_tf, lambda x: x, lambda x: x.numpy(), 5)


if __name__ == "__main__":
  absltest.main()
