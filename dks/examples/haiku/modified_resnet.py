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

""""A "modified ResNet model" in Haiku with support for both DKS and TAT."""

import math
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from dks.jax import activation_transform
from dks.jax import haiku_initializers
import haiku as hk
import jax.numpy as jnp


FloatStrOrBool = Union[str, float, bool]

BN_CONFIG = {
    "create_offset": True,
    "create_scale": True,
    "decay_rate": 0.999,
}


class BlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bottleneck: bool,
      use_batch_norm: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      shortcut_weight: Optional[float],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.use_batch_norm = use_batch_norm
    self.shortcut_weight = shortcut_weight

    if self.use_projection and self.shortcut_weight != 0.0:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          w_init=w_init,
          with_bias=not use_batch_norm,
          padding="SAME",
          name="shortcut_conv")
      if use_batch_norm:
        self.proj_batchnorm = hk.BatchNorm(
            name="shortcut_batchnorm", **BN_CONFIG)

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        with_bias=not use_batch_norm,
        padding="SAME",
        name="conv_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        w_init=w_init,
        with_bias=not use_batch_norm,
        padding="SAME",
        name="conv_1")

    layers = (conv_0, conv_1)

    if use_batch_norm:
      bn_0 = hk.BatchNorm(name="batchnorm_0", **BN_CONFIG)
      bn_1 = hk.BatchNorm(name="batchnorm_1", **BN_CONFIG)
      bn_layers = (bn_0, bn_1)

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          w_init=w_init,
          with_bias=not use_batch_norm,
          padding="SAME",
          name="conv_2")

      layers = layers + (conv_2,)

      if use_batch_norm:
        bn_2 = hk.BatchNorm(name="batchnorm_2", **BN_CONFIG)
        bn_layers += (bn_2,)
        self.bn_layers = bn_layers

    self.layers = layers
    self.activation = activation

  def __call__(self, inputs, is_training, test_local_stats):
    out = shortcut = inputs

    if self.use_projection and self.shortcut_weight != 0.0:
      shortcut = self.proj_conv(shortcut)
      if self.use_batch_norm:
        shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

    for i, conv_i in enumerate(self.layers):
      out = conv_i(out)
      if self.use_batch_norm:
        out = self.bn_layers[i](out, is_training, test_local_stats)
      if i < len(self.layers) - 1:  # Don't apply activation on last layer
        out = self.activation(out)

    if self.shortcut_weight is None:
      return self.activation(out + shortcut)
    elif self.shortcut_weight != 0.0:
      return self.activation(
          math.sqrt(1 - self.shortcut_weight**2) * out +
          self.shortcut_weight * shortcut)
    else:
      return out


class BlockV2(hk.Module):
  """ResNet V2 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bottleneck: bool,
      use_batch_norm: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      shortcut_weight: Optional[float],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.use_batch_norm = use_batch_norm
    self.shortcut_weight = shortcut_weight

    if self.use_projection and self.shortcut_weight != 0.0:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          w_init=w_init,
          with_bias=not use_batch_norm,
          padding="SAME",
          name="shortcut_conv")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        with_bias=not use_batch_norm,
        padding="SAME",
        name="conv_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        w_init=w_init,
        with_bias=not use_batch_norm,
        padding="SAME",
        name="conv_1")

    layers = (conv_0, conv_1)

    if use_batch_norm:
      bn_0 = hk.BatchNorm(name="batchnorm_0", **BN_CONFIG)
      bn_1 = hk.BatchNorm(name="batchnorm_1", **BN_CONFIG)
      bn_layers = (bn_0, bn_1)

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          w_init=w_init,
          with_bias=not use_batch_norm,
          padding="SAME",
          name="conv_2")

      layers = layers + (conv_2,)

      if use_batch_norm:
        bn_2 = hk.BatchNorm(name="batchnorm_2", **BN_CONFIG)
        bn_layers += (bn_2,)
        self.bn_layers = bn_layers

    self.layers = layers
    self.activation = activation

  def __call__(self, inputs, is_training, test_local_stats):
    x = shortcut = inputs

    for i, conv_i in enumerate(self.layers):

      if self.use_batch_norm:
        x = self.bn_layers[i](x, is_training, test_local_stats)

      x = self.activation(x)

      if i == 0 and self.use_projection and self.shortcut_weight != 0.0:
        shortcut = self.proj_conv(x)

      x = conv_i(x)

    if self.shortcut_weight is None:
      return x + shortcut
    elif self.shortcut_weight != 0.0:
      return math.sqrt(
          1 - self.shortcut_weight**2) * x + self.shortcut_weight * shortcut
    else:
      return x


class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      use_batch_norm: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      shortcut_weight: Optional[float],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2 if resnet_v2 else BlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(
              channels=channels,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
              use_batch_norm=use_batch_norm,
              bottleneck=bottleneck,
              shortcut_weight=shortcut_weight,
              activation=activation,
              w_init=w_init,
              name="block_%d" % (i)))

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ModifiedResNet(hk.Module):
  """Modified version of an Imagenet ResNet model that supports DKS/TAT."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      200: {
          "blocks_per_group": (3, 24, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
  }

  def __init__(
      self,
      num_classes: int,
      depth: int,
      resnet_v2: bool = True,
      use_batch_norm: bool = False,
      shortcut_weight: Optional[float] = 0.0,
      activation_name: str = "softplus",
      w_init: Optional[Any] = haiku_initializers.ScaledUniformOrthogonal(
          delta=True),
      logits_config: Optional[Mapping[str, Any]] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      dropout_rate: float = 0.0,
      transformation_method: str = "DKS",
      dks_params: Optional[Mapping[str, FloatStrOrBool]] = None,
      tat_params: Optional[Mapping[str, FloatStrOrBool]] = None,
      name: Optional[str] = None,
  ):
    """Constructs a "modified ResNet model" with support for both DKS and TAT.

    By default, we construct the network *without* normalization layers or
    skip connections (making it a "vanilla network"), initialize the weights
    with the SUO distribution, and use DKS to transform the activation functions
    (which are "softplus" by default). These behaviors, and the option to use
    TAT, are controlled via the contructor arguments.

    This file was adapted from the original Haiku ResNet implementation:
    https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
    It is the end result of applying the rules described in the section titled
    "Summary of our method" in the DKS paper (https://arxiv.org/abs/2110.01765)
    to what is essentially a standard ResNet. See the section titled
    "Application to various modified ResNets" in the DKS paper for more details.
    The only departure from this is that we construct the "maximal C map
    function" instead of the "maximal slope function" (which can be computed
    from the former), which enables support for TAT.


    Args:
      num_classes: The number of classes to classify the inputs into.
      depth: The number of layers.
      resnet_v2: Whether to use the v2 ResNet implementation instead of v1.
        Defaults to ``True``.
      use_batch_norm: Whether to use Batch Normalization (BN). Note that DKS/TAT
        are not compatible with the use of BN. Defaults to ``False``.
      shortcut_weight: The weighting factor of shortcut branch, which must be
        a float between 0 and 1, or None. If not None, the shortcut branch is
        multiplied by ``shortcut_weight``, and the residual branch is multiplied
        by ``residual_weight``, where
                ``shortcut_weight**2 + residual_weight**2 == 1.0``.
        If None, no multiplications are performed (which corresponds to a
        standard ResNet), and compatibility with DKS/TAT is lost. Note that
        setting ``shortcut_weight`` to 0.0 effectively removes the skip
        connections from the network. Defaults to ``0.0``.
      activation_name: String name for activation function. To get TReLU from
        the TAT paper one should set this to ``leaky_relu``, and set
        the ``transformation_method`` argument to ``TAT``. Defaults to
        ``softplus``.
      w_init: Haiku initializer used to initialize the weights.
      logits_config: A dictionary of keyword arguments for the logits layer.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      dropout_rate: A float giving the dropout rate for penultimate layer of the
        network (i.e. right before the layer which produces the class logits).
        (Default: 0.0)
      transformation_method: A string representing the method used to transform
        the activation function. Can be ``DKS``, ``TAT``, or ``untransformed``.
        Defaults to ``DKS``.
      dks_params: A dictionary containing the parameters to use for DKS. See
        activation_transform.get_transformed_activations for more details.
        Defaults to ``None``.
      tat_params: A dictionary containing the parameters to use for TAT. See
        activation_transform.get_transformed_activations for more details.
        Defaults to ``None``.
      name: Name of the Sonnet module.
    """
    super().__init__(name=name)

    if shortcut_weight is not None and (shortcut_weight > 1.0
                                        or shortcut_weight < 0.0):
      raise ValueError("Unsupported value for shortcut_weight.")

    if (use_batch_norm and
        (transformation_method == "DKS" or transformation_method == "TAT")):
      raise ValueError("DKS and TAT are not compatible with the use of BN "
                       "layers.")

    if (shortcut_weight is None and
        (transformation_method == "DKS" or transformation_method == "TAT")):
      raise ValueError("Must specify a value for shortcut_weight when using "
                       "DKS or TAT.")

    self.depth = depth
    self.resnet_v2 = resnet_v2
    self.use_batch_norm = use_batch_norm
    self.shortcut_weight = shortcut_weight
    self.activation_name = activation_name
    self.dropout_rate = dropout_rate

    blocks_per_group = ModifiedResNet.CONFIGS[depth]["blocks_per_group"]
    channels_per_group = ModifiedResNet.CONFIGS[depth]["channels_per_group"]
    bottleneck = ModifiedResNet.CONFIGS[depth]["bottleneck"]
    use_projection = ModifiedResNet.CONFIGS[depth]["use_projection"]

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", w_init)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 64)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", not use_batch_norm)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")
    initial_conv_config.setdefault("w_init", w_init)

    act_dict = activation_transform.get_transformed_activations(
        [self.activation_name], method=transformation_method,
        dks_params=dks_params, tat_params=tat_params,
        subnet_max_func=self.subnet_max_func)

    self.activation = act_dict[self.activation_name]

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    if not self.resnet_v2 and use_batch_norm:
      self.initial_batchnorm = hk.BatchNorm(
          name="initial_batchnorm", **BN_CONFIG)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          BlockGroup(
              channels=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              resnet_v2=resnet_v2,
              bottleneck=bottleneck,
              use_batch_norm=use_batch_norm,
              use_projection=use_projection[i],
              shortcut_weight=shortcut_weight,
              activation=self.activation,
              w_init=w_init,
              name="block_group_%d" % (i)))

    if self.resnet_v2 and use_batch_norm:
      self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **BN_CONFIG)

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs, is_training, test_local_stats=False):
    out = inputs
    out = self.initial_conv(out)

    if not self.resnet_v2:
      if self.use_batch_norm:
        out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = self.activation(out)

    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:
      if self.use_batch_norm:
        out = self.final_batchnorm(out, is_training, test_local_stats)

      out = self.activation(out)

    out = jnp.mean(out, axis=(1, 2))

    if self.dropout_rate > 0.0 and is_training:
      out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)

    return self.logits(out)

  def subnet_max_func(self, x, r_fn):
    return subnet_max_func(x, r_fn, self.depth, self.shortcut_weight)


def subnet_max_func(x, r_fn, depth, shortcut_weight):
  """The subnetwork maximizing function of the modified ResNet model."""

  # See Appendix B of the TAT paper for a step-by-step procedure for how
  # to compute this function for different architectures.

  blocks_per_group = ModifiedResNet.CONFIGS[depth]["blocks_per_group"]
  bottleneck = ModifiedResNet.CONFIGS[depth]["bottleneck"]
  use_projection = ModifiedResNet.CONFIGS[depth]["use_projection"]

  res_branch_subnetwork_x = r_fn(r_fn(r_fn(x)))

  for i in range(4):
    for j in range(blocks_per_group[i]):
      if bottleneck:
        res_x = r_fn(r_fn(r_fn(x)))
      else:
        res_x = r_fn(r_fn(x))

      shortcut_x = r_fn(x) if (j == 0 and use_projection[i]) else x

      x = (shortcut_weight**2 * shortcut_x + (1.0 - shortcut_weight**2) * res_x)

  x = r_fn(x)

  return max(x, res_branch_subnetwork_x)
