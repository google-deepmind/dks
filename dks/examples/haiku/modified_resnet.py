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

"""A "modified ResNet model" in Haiku with support for both DKS and TAT."""

import inspect
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

from dks.jax import activation_transform
from dks.jax import haiku_initializers
import haiku as hk
import jax.numpy as jnp


FloatStrOrBool = Union[str, float, bool]


DEFAULT_BN_CONFIG = {
    "create_offset": True,
    "create_scale": True,
    "decay_rate": 0.99,
}


class BlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bottleneck: bool,
      norm_layers_ctor: Optional[Any],
      should_use_bias: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      shortcut_weight: Optional[float],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.use_projection = use_projection
    self.norm_layers_ctor = norm_layers_ctor
    self.shortcut_weight = shortcut_weight

    if self.use_projection and self.shortcut_weight != 0.0:

      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          w_init=w_init,
          with_bias=should_use_bias,
          padding="SAME",
          name="shortcut_conv")

      if norm_layers_ctor is not None:
        self.proj_norm = norm_layers_ctor(name="shortcut_norm")

    channel_div = 4 if bottleneck else 1

    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        with_bias=should_use_bias,
        padding="SAME",
        name="conv_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        w_init=w_init,
        with_bias=should_use_bias,
        padding="SAME",
        name="conv_1")

    layers = (conv_0, conv_1)

    if norm_layers_ctor is not None:

      norm_0 = norm_layers_ctor(name="norm_0")
      norm_1 = norm_layers_ctor(name="norm_1")

      self.norm_layers = (norm_0, norm_1)

    else:
      self.norm_layers = None

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          w_init=w_init,
          with_bias=should_use_bias,
          padding="SAME",
          name="conv_2")

      layers = layers + (conv_2,)

      if norm_layers_ctor:
        norm_2 = norm_layers_ctor(name="norm_2")
        self.norm_layers += (norm_2,)

    self.layers = layers
    self.activation = activation

  def __call__(self, inputs, is_training, test_local_stats):

    out = shortcut = inputs

    if self.use_projection and self.shortcut_weight != 0.0:

      shortcut = self.proj_conv(shortcut)

      if self.norm_layers is not None:
        shortcut = self.proj_norm(shortcut, is_training=is_training,
                                  test_local_stats=test_local_stats)

    for i, conv_i in enumerate(self.layers):

      out = conv_i(out)

      if self.norm_layers is not None:
        out = self.norm_layers[i](out, is_training=is_training,
                                  test_local_stats=test_local_stats)

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
      norm_layers_ctor: Optional[Any],
      should_use_bias: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      shortcut_weight: Optional[float],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.use_projection = use_projection
    self.norm_layers_ctor = norm_layers_ctor
    self.shortcut_weight = shortcut_weight

    if self.use_projection and self.shortcut_weight != 0.0:

      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          w_init=w_init,
          with_bias=should_use_bias,
          padding="SAME",
          name="shortcut_conv")

    channel_div = 4 if bottleneck else 1

    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        with_bias=should_use_bias,
        padding="SAME",
        name="conv_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        w_init=w_init,
        with_bias=should_use_bias,
        padding="SAME",
        name="conv_1")

    layers = (conv_0, conv_1)

    if norm_layers_ctor is not None:

      norm_0 = norm_layers_ctor(name="norm_0")
      norm_1 = norm_layers_ctor(name="norm_1")

      self.norm_layers = (norm_0, norm_1)

    else:
      self.norm_layers = None

    if bottleneck:

      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          w_init=w_init,
          with_bias=should_use_bias,
          padding="SAME",
          name="conv_2")

      layers = layers + (conv_2,)

      if norm_layers_ctor is not None:

        norm_2 = norm_layers_ctor(name="norm_2")
        self.norm_layers += (norm_2,)

    self.layers = layers
    self.activation = activation

  def __call__(self, inputs, is_training, test_local_stats):

    x = shortcut = inputs

    for i, conv_i in enumerate(self.layers):

      if self.norm_layers is not None:
        x = self.norm_layers[i](x, is_training=is_training,
                                test_local_stats=test_local_stats)

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
      norm_layers_ctor: Optional[Any],
      should_use_bias: bool,
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
              norm_layers_ctor=norm_layers_ctor,
              should_use_bias=should_use_bias,
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
      use_norm_layers: bool = False,
      norm_layers_ctor: Optional[Type[hk.SupportsCall]] = None,
      norm_layers_kwargs: Optional[Mapping[str, Any]] = None,
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
    TAT, are controlled via the constructor arguments.

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
      use_norm_layers: Whether to use normalization. Note that DKS/TAT are not
        compatible with the use of Batch Norm layers, but Layer Norm is fine.
        Defaults to ``False``.
      norm_layers_ctor: Haiku constructor to use for normalization layers (if
        enabled). Defaults to ``None``, which uses ``hk.BatchNorm``.
      norm_layers_kwargs: Keyword arguments to pass to the normalization layer
        constructor. Defaults to ``None``, which uses the default BN config.
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
        ``dks.base.activation_transform.get_transformed_activations`` for more
        details. Defaults to ``None``.
      tat_params: A dictionary containing the parameters to use for TAT. See
        ``dks.base.activation_transform.get_transformed_activations`` for more
        details. Defaults to ``None``.
      name: Name of the Sonnet module.
    """
    super().__init__(name=name)

    if use_norm_layers:

      norm_layers_ctor = norm_layers_ctor or hk.BatchNorm

      if norm_layers_ctor == hk.BatchNorm:
        should_use_bias = False
        if norm_layers_kwargs is None:
          norm_layers_kwargs = DEFAULT_BN_CONFIG

      elif norm_layers_kwargs is not None:
        should_use_bias = norm_layers_kwargs.get("create_offset", True)

      else:
        should_use_bias = True

      if norm_layers_kwargs is None:
        raise ValueError("Must specify 'norm_layers_kwargs' when using "
                         "non-BN normalization layers.")

      norm_layers_ctor_unwrapped = norm_layers_ctor
      norm_layers_ctor = lambda *a, **k: _filter_kwargs(
          norm_layers_ctor_unwrapped(*a, **k, **norm_layers_kwargs))  # pytype: disable=wrong-keyword-args

    else:
      norm_layers_ctor_unwrapped = None
      should_use_bias = True
      norm_layers_ctor = None

    if shortcut_weight is not None and (shortcut_weight > 1.0
                                        or shortcut_weight < 0.0):
      raise ValueError("Unsupported value for shortcut_weight.")

    if transformation_method in ("DKS", "TAT"):

      if norm_layers_ctor_unwrapped == hk.BatchNorm:
        raise ValueError("DKS and TAT are not compatible with the use of BN "
                         "layers.")

      if shortcut_weight is None:
        raise ValueError("Must specify a value for shortcut_weight when using "
                         "DKS or TAT.")

    self.depth = depth
    self.resnet_v2 = resnet_v2
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
    initial_conv_config.setdefault("with_bias", should_use_bias)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")
    initial_conv_config.setdefault("w_init", w_init)

    act_dict = activation_transform.get_transformed_activations(
        [self.activation_name], method=transformation_method,
        dks_params=dks_params, tat_params=tat_params,
        subnet_max_func=self.subnet_max_func)

    self.activation = act_dict[self.activation_name]

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    if not self.resnet_v2 and norm_layers_ctor is not None:
      self.initial_norm = norm_layers_ctor(name="initial_norm")
    else:
      self.initial_norm = None

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
              norm_layers_ctor=norm_layers_ctor,
              should_use_bias=should_use_bias,
              use_projection=use_projection[i],
              shortcut_weight=shortcut_weight,
              activation=self.activation,
              w_init=w_init,
              name="block_group_%d" % (i)))

    if self.resnet_v2 and norm_layers_ctor is not None:
      self.final_norm = norm_layers_ctor(name="final_norm")
    else:
      self.final_norm = None

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs, is_training, test_local_stats=False):

    out = inputs
    out = self.initial_conv(out)

    if not self.resnet_v2:

      if self.initial_norm is not None:
        out = self.initial_norm(out, is_training=is_training,
                                test_local_stats=test_local_stats)

      out = self.activation(out)

    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:

      if self.final_norm is not None:
        out = self.final_norm(out, is_training=is_training,
                              test_local_stats=test_local_stats)

      out = self.activation(out)

    out = jnp.mean(out, axis=(1, 2))

    if self.dropout_rate > 0.0 and is_training:
      out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)

    return self.logits(out)

  def subnet_max_func(self, x, r_fn):
    return subnet_max_func(x, r_fn, self.depth, self.shortcut_weight,
                           resnet_v2=self.resnet_v2)


def subnet_max_func(x, r_fn, depth, shortcut_weight, resnet_v2=True):
  """The subnetwork maximizing function of the modified ResNet model."""

  # See Appendix B of the TAT paper for a step-by-step procedure for how
  # to compute this function for different architectures.

  blocks_per_group = ModifiedResNet.CONFIGS[depth]["blocks_per_group"]
  bottleneck = ModifiedResNet.CONFIGS[depth]["bottleneck"]
  use_projection = ModifiedResNet.CONFIGS[depth]["use_projection"]

  if bottleneck and resnet_v2:
    res_fn = lambda z: r_fn(r_fn(r_fn(z)))

  elif (not bottleneck and resnet_v2) or (bottleneck and not resnet_v2):
    res_fn = lambda z: r_fn(r_fn(z))

  else:
    res_fn = r_fn

  res_branch_subnetwork = res_fn(x)

  for i in range(4):
    for j in range(blocks_per_group[i]):

      res_x = res_fn(x)

      if j == 0 and use_projection[i] and resnet_v2:
        shortcut_x = r_fn(x)
      else:
        shortcut_x = x

      x = (shortcut_weight**2 * shortcut_x + (1.0 - shortcut_weight**2) * res_x)

      if not resnet_v2:
        x = r_fn(x)

  x = r_fn(x)

  return max(x, res_branch_subnetwork)


def _filter_kwargs(fn_or_class):
  """Wraps a function or class to ignore over-specified arguments."""

  method_fn = (fn_or_class.__init__ if isinstance(fn_or_class, Type) else
               fn_or_class)

  if isinstance(method_fn, hk.Module):
    # Haiku wraps `__call__` and destroys the `argspec`. However, it does
    # preserve the signature of the function.
    fn_args = list(inspect.signature(method_fn.__call__).parameters.keys())
  else:
    fn_args = inspect.getfullargspec(method_fn).args

  if fn_args and "self" == fn_args[0]:
    fn_args = fn_args[1:]

  def wrapper(*args, **kwargs):

    common_kwargs = {}

    if len(args) > len(fn_args):
      raise ValueError("Too many positional arguments.")

    for k, v in zip(fn_args, args):
      common_kwargs[k] = v

    for k, v in kwargs.items():
      if k in common_kwargs:
        raise ValueError(
            "{} already specified as a positional argument".format(k))
      if k in fn_args:
        common_kwargs[k] = v

    return fn_or_class(**common_kwargs)

  return wrapper
