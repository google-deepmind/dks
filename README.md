# Official Python package for Deep Kernel Shaping (DKS) and Tailored Activation Transformations (TAT)

This Python package implements the activation function transformations and
weight initializations used in the [DKS paper] and [TAT paper]. It supports the
JAX, PyTorch, and TensorFlow tensor programming frameworks.

Questions/comments about the code can be sent to
[dks-dev@google.com](mailto:dks-dev@google.com).

NOTE: we are not taking code contributions from Github at this time. All PRs
from Github will be rejected. Instead, please email us if you find a bug.

## Usage

For each tensor programming framework supported there is a corresponding
directory/subpackage which handles the activation function transformations and
weight initializations. It's up to the user to import these and use them
appropriately within their model code. Activation functions are transformed by
the function `get_transformed_activations()` in module `activation_transform` of
the appropriate subpackage. Weight sampling is done using functions inside of
the module `parameter_sampling_functions` of said subpackage.

In addition to using these functions, the user is responsble for ensuring that
their model meets the architectural requirements of DKS/TAT, and for converting
any weighted sums in their model to "normalized sums" (which are weighted sums
whoses non-trainable weights have a sum of squares equal to 1). This package
doesn't currently include an implementation of Per-Location Normalization (PLN)
data pre-processing. While not required for CIFAR or ImageNet, PLN could
possibly be important for other datasets. See the section titled "Summary of our
method" in the [DKS paper] for more details about the requirements and execution
steps of DKS. TAT is very similar except that it requires the maximal C map
function instead of the maximal slope function (which can be computed from the
former).

Note that ReLUs are only partially supported by DKS, and their use is *highly*
discouraged. Instead, one should use Leaky ReLUs, which are fully supported by
DKS, and work especially well with TAT.

Note that in order to avoid having to import all of the tensor frameworks, the
user is required to individually import whatever framework subpackage they want.
e.g. `import dks.jax`. Meanwhile, `import dks` won't actually do anything.

## Example

`dks.examples.haiku.modified_resnet` is a Haiku ResNet model which has been
modified as described in the DKS/TAT papers, and includes support for both DKS
and TAT. (By default, it removes the normalization layers or skip connections
found in ResNets, making it a "vanilla network".) It can be used as an
instructive example for how to construct DKS/TAT models using this package. See
the section titled "Application to various modified ResNets" from the
[DKS paper] for more details.

## Installation

**Insert instructions here once they are known**

It is strongly recommended that you run the tests for the library using `python
-m pytest -n "${N_JOBS}" tests` the same Python environment in which you plan to
use the library. Note that this is different from running the included `test.sh`
script, which creates a separate virtual environment with very specific versions
for the package's dependencies.

## Disclaimer

This is not an official Google product.

[DKS paper]: https://arxiv.org/abs/2110.01765
[TAT paper]: https://openreview.net/forum?id=U0k7XNTiFEq
