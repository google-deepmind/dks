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

"""PyTorch implementation of the activation transformations used in DKS/TAT."""

from dks.base import activation_transform
import torch
import torch.nn.functional as tfunc


def _get_pytorch_activation_function(name):
  """Get activation function by name in PyTorch."""

  if name == "bentid":
    return lambda x: (torch.sqrt(torch.square(x) + 1.) - 1.) / 2. + x
  elif hasattr(tfunc, name):
    return getattr(tfunc, name)
  elif hasattr(torch, name):
    return getattr(torch, name)
  else:
    raise ValueError(f"Unrecognized activation function name '{name}'.")


def get_transformed_activations(*args, **kwargs):
  """See ``dks.base.activation_transform.get_transformed_activations()``."""

  return activation_transform.get_transformed_activations(
      *args, **kwargs, activation_getter=_get_pytorch_activation_function)
