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

"""Tests the activation function transform module for each framework."""

import numpy as np
import torch
import jax.numpy as jnp
import tensorflow as tf
from pkg_resources import parse_version
from dks.base import activation_transform as activation_transform_base

def _assert_structure_approx_equal(s1, s2, places=8):
    # Add a tolerance check to the structure comparison
    tree.map_structure(lambda x, y: np.testing.assert_almost_equal(x, y, decimal=places), s1, s2)

def _run_value_tests(to_framework, to_numpy, places):
    # Move the value tests to a common function
    val1 = to_framework(0.6)
    val2 = to_framework(-0.6)

    def check(activation_names, method, expected_values_dict):
        act_dict = activation_transform_base.get_transformed_activations(
            activation_names=activation_names,
            method=method,
            dks_params={"c_slope_1_target": 12.0, "local_q_slope_target": 0.97},
            tat_params={"c_val_0_target": 0.95, "c_curve_target": 4.5},
        )
        for name in expected_values_dict:
            _assert_structure_approx_equal(
                to_numpy(act_dict[name](val1)),
                expected_values_dict[name][0],
                places=places
            )
            _assert_structure_approx_equal(
                to_numpy(act_dict[name](val2)),
                expected_values_dict[name][1],
                places=places
            )

    check(("softplus",), "DKS", {"softplus": [0.5205088334781294, -0.6970897398761904]})
    # Add more tests for different activation functions and methods

if __name__ == "__main__":
    _run_value_tests(np.asarray, lambda x: x, 8)
    _run_value_tests(jnp.asarray, np.asarray, 5)
    _run_value_tests(torch.tensor, lambda x: x.detach().cpu().numpy(), 5)
    _run_value_tests(tf.convert_to_tensor, lambda x: x.numpy(), 5)
