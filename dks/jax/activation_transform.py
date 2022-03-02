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

"""JAX implementation of the activation transformations used in DKS/TAT."""

from dks.base import activation_transform
import jax
import jax.numpy as jnp


def _get_jax_activation_function(name):
  """Get activation function by name in JAX."""

  if name == "bentid":
    return lambda x: (jnp.sqrt(jnp.square(x) + 1.) - 1.) / 2. + x
  elif name == "softsign":
    return jax.nn.soft_sign
  elif hasattr(jax.lax, name):
    return getattr(jax.lax, name)
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise ValueError(f"Unrecognized activation function name '{name}'.")


def get_transformed_activations(*args, **kwargs):
  """See ``dks.base.activation_transform.get_transformed_activations()``."""

  return activation_transform.get_transformed_activations(
      *args, **kwargs, activation_getter=_get_jax_activation_function)
