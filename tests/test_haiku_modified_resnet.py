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

"""Basic tests for the Haiku modified ResNet example model."""

from absl.testing import absltest
from absl.testing import parameterized

from dks.examples.haiku.modified_resnet import ModifiedResNet
import haiku as hk
import jax
import jax.numpy as jnp


class ModifiedResNetTest(parameterized.TestCase):

  @parameterized.parameters((50, 0.0, "tanh", False), (101, 0.5, "relu", True),
                            (152, 0.9, "leaky_relu", True))
  def test_model_instantiation_and_apply(self, depth, shortcut_weight, act_name,
                                         resnet_v2):
    """Tests that the model can be instantiated and applied on data."""

    def func(batch, is_training):
      model = ModifiedResNet(
          num_classes=1000,
          depth=depth,
          resnet_v2=resnet_v2,
          activation_name=act_name,
          shortcut_weight=shortcut_weight,
      )
      return model(batch, is_training=is_training)

    forward = hk.without_apply_rng(hk.transform_with_state(func))

    rng = jax.random.PRNGKey(42)

    image = jnp.ones([2, 224, 224, 3])
    params, state = forward.init(rng, image, is_training=True)
    logits, state = forward.apply(params, state, image, is_training=True)

    self.assertEqual(logits.shape, (2, 1000))


if __name__ == "__main__":
  absltest.main()
