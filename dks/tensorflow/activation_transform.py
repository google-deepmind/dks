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

"""TF implementation of the activation transformations used in DKS/TAT."""

from dks.base import activation_transform
import tensorflow as tf


def _get_tf_activation_function(name):
  """Get activation function by name in TensorFlow."""

  if name == "bentid":
    return lambda x: (tf.sqrt(tf.square(x) + 1.) - 1.) / 2. + x
  elif name == "erf":
    return tf.math.erf
  elif name == "atan":
    return tf.math.atan
  elif name == "asinh":
    return tf.math.asinh
  elif name == "leaky_relu":
    return lambda x, negative_slope=0.01: tf.nn.leaky_relu(  # pylint: disable=g-long-lambda
        x, alpha=negative_slope)
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  else:
    raise ValueError(f"Unrecognized activation function name '{name}'.")


def get_transformed_activations(*args, **kwargs):
  """See ``dks.base.activation_transform.get_transformed_activations()``."""

  return activation_transform.get_transformed_activations(
      *args, **kwargs, activation_getter=_get_tf_activation_function)
