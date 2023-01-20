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

"""Data pre-processing functions for use with DKS/TAT in JAX."""

import functools
import operator

from absl import logging
import jax.numpy as jnp


_prod = lambda x: functools.reduce(operator.mul, x, 1)


def per_location_normalization(x, homog_mode="one", homog_scale=1.0,
                               has_batch_dim=True):
  """Applies Per-Location Normalization (PLN) to a given array.

  This function generalizes the idea of PLN from the DKS paper to arrays of
  arbitrary shape. Normalization is done over the last dimension (and only the
  last dimension), so that ``jnp.mean(PLN(x)**2, axis=-1, keepdims=True)`` is an
  array of ones. Note here that "normalization" does not correspond to making
  the vectors at each location have norm 1. Rather, they will have a squared
  norm given by ``x.shape[-1]``.

  All dimensions, except for the last, and possibly the first (which may be the
  batch dimension), are treated as indexing different "locations", analogous to
  how locations are indexed by the height and width dimensions in convolutional
  layers. The last dimension is always considered the "data" or "feature"
  dimension, analogous to the channels dimension in convolutional layers. For
  models where the dimensions don't have this interpretation, this type of
  preprocessing may not be suitable. (And it's likely that the rest of the
  ``dks`` package, and perhaps even the DKS/TAT method iself, won't be
  applicable either.)

  Before normalization occurs, a homogeneous coordinate may be appended to the
  last dimension of the array. If and how this depends on the value
  of the arguements ``homog_mode`` and ``homog_scale``, as described in the
  arguments section. This step is designed to preserve the information that
  would otherwise be lost due to normalization.

  The motivation for PLN is to ensure that the input "q values" to a network are
  always 1, which is a technical requirement of DKS/TAT. While DKS/TAT can often
  work well in practice without PLN, there are situations where using PLN will
  be crucial. In particular, if the input data, or particular samples from it,
  have an extreme scale that deviates from the typical ones seen in CIFAR and
  ImageNet (with the standard preprocessing applied). With CIFAR in particular
  we have observed that some pixels in some images have feature vectors that are
  exactly zero, which can lead to problems when using TAT with leaky ReLUs.

  See the section titled "Uniform q values via Per-Location Normalization" in
  the DKS paper (https://arxiv.org/abs/2110.01765) for a discussion of PLN.

  Args:
    x: A JAX array representing the input to a network to be normalized. If
      ``x`` has a batch dimension it must be the first one.
    homog_mode: A string indicating whether to append a homogeneous coordinate,
      and how to compute it. Can be ``one``, ``avg_q``, or ``off``. If
      ``one``, the coordinate will have the value 1. If ``avg_q``, it will be
      given by taking the mean squared value of ``x`` across the non-batch axes.
      If ``off`, no homogeneous coordinate will be added. (Default: "one")
    homog_scale: A float used to rescale homogenous coordinate (if used).
      (Default: 1.0)
    has_batch_dim: A boolean specifying whether ``x`` has a batch dimension
      (which will always be its first dimension). Note that many data processing
      pipelines will process training cases one at a time. Unless this is done
      with a singleton leading "dummy" batch dimension (which isn't typical)
      this argument should be set to False. (Default: True)

  Returns:
    A JAX array which is the result of applying PLN to ``x``, as described
    above.
  """

  def q_val(z, axis):
    return jnp.mean(jnp.square(z), axis=axis, keepdims=True)

  x_shape = list(x.shape)

  if len(x_shape) == 0 and has_batch_dim:  # pylint: disable=g-explicit-length-test
    raise ValueError("dataset doesn't appear to have a batch dimension.")

  if homog_mode == "avg_q" and ((len(x_shape) <= 2 and has_batch_dim)
                                or (len(x_shape) <= 1 and not has_batch_dim)):  # pylint: disable=g-explicit-length-test

    raise ValueError("homog_mode='avg_q' should not be used for datasets with "
                     "no time/location dimension, as it doesn't offer anything "
                     "beyond what homog_mode='off' would in such cases.")

  if ((len(x_shape) == 1 and has_batch_dim)
      or (len(x_shape) == 0 and not has_batch_dim)):  # pylint: disable=g-explicit-length-test

    x = jnp.expand_dims(x, axis=-1)
    x_shape += [1]

  # the threshold 20 is arbitrary
  if _prod(x_shape[1 if has_batch_dim else 0:]) < 20 and homog_mode == "avg_q":

    logging.warning("Using homog_mode='avg_q' for datasets with few total "
                    "degrees of freedom per batch element (taken over "
                    "time/location dimensions and the data dimension) is "
                    "dangerous. This is because it will remove one degree of "
                    "freedom, and possibly destroy important information. See "
                    "the discussion in the subsection of the DKS paper titled "
                    "'Uniform q values via Per-Location Normalization'.")

  if x_shape[-1] < 20 and homog_mode == "off":

    logging.warning("Using homog_mode='off' for datasets with a small data "
                    "dimension is dangerous. This is because it will remove "
                    "one degree of freedom in this dimension, and possibly "
                    "destroy important information. See the discussion in the "
                    "subsection of the DKS paper titled 'Uniform q values via "
                    "Per-Location Normalization'.")

  if homog_mode == "avg_q":

    homog = jnp.sqrt(q_val(x, axis=list(range(1 if has_batch_dim else 0,
                                              len(x_shape)))))

    if has_batch_dim:
      homog = jnp.tile(homog, [1] + x_shape[1:-1] + [1])

    else:
      homog = jnp.tile(homog, x_shape[:-1] + [1])

  elif homog_mode == "one":
    homog = jnp.ones(x_shape[:-1] + [1])

  elif homog_mode == "off":
    homog = None

  else:
    raise ValueError(f"Unrecognized value for homog_mode: {homog_mode}.")

  if homog_scale != 1.0 and homog is not None:
    homog = homog_scale * homog

  if homog is not None:
    x = jnp.concatenate([x, homog], axis=-1)

  x = x / jnp.sqrt(q_val(x, axis=-1))

  return x
