![CI status](https://github.com/deepmind/dks/workflows/ci/badge.svg)
![pypi](https://img.shields.io/pypi/v/dks)

# Official Python package for Deep Kernel Shaping (DKS) and Tailored Activation Transformations (TAT)

This Python package implements the activation function transformations, weight
initializations, and dataset preprocessing used in Deep Kernel Shaping (DKS) and
Tailored Activation Transformations (TAT). DKS and TAT, which were introduced in
the [DKS paper] and [TAT paper], are methods for constructing/transforming
neural networks to make them much easier to train. For example, these methods
can be used in conjunction with K-FAC to train deep vanilla deep convnets
(without skip connections or normalization layers) as fast as standard ResNets
of the same depth.

The package supports the JAX, PyTorch, and TensorFlow tensor programming
frameworks.

Questions/comments about the code can be sent to
[dks-dev@google.com](mailto:dks-dev@google.com).

**NOTE:** we are not taking code contributions from Github at this time. All PRs
from Github will be rejected. Instead, please email us if you find a bug.

## Usage

For each of the supported tensor programming frameworks, there is a
corresponding subpackage which handles the activation function transformations,
weight initializations, and (optional) data preprocessing. (These are `dks.jax`,
`dks.pytorch`, and `dks.tensorflow`.) It's up to the user to import these and
use them appropriately within their model code. Activation functions are
transformed by the function `get_transformed_activations()` in the module
`activation_transform` of the appropriate subpackage. Sampling initial
parameters is done using functions inside of the module
`parameter_sampling_functions` of said subpackage. And data preprocessing is
done using the function `per_location_normalization` inside of the module
`data_preprocessing` of said subpackage. Note that in order to avoid having to
import all of the tensor programming frameworks, the user is required to
individually import whatever framework subpackage they want. e.g. `import
dks.jax`. Meanwhile, `import dks` won't actually do anything.

`get_transformed_activations()` requires the user to pass either the "maximal
slope function" for DKS, the "subnet maximizing function" for TAT with Leaky
ReLUs, or the "maximal curvature function" for TAT with smooth activation
functions. (The subnet maximizing function also handles DKS and TAT with smooth
activations.) These are special functions that encode information about the
particular model architecture. See the section titled "Summary of our method" of
the [DKS paper] for a procedure to construct the maximal slope function for a
given model, or the appendix section titled "Additional details and pseudocode
for activation function transformations" of the [TAT paper] for procedures to
construct the other two functions.

In addition to these things, the user is responsible for ensuring that their
model meets the architectural requirements of DKS/TAT, and for converting any
weighted sums into "normalized sums" (which are weighted sums whose
non-trainable weights have a sum of squares equal to 1). See the section titled
"Summary of our method" of the [DKS paper] for more details.

Note that the data preprocessing method implemented, called Per-Location 
Normalization (PLN), may not always be needed in practice, but we have observed
certain situations where not using can lead to problems. (For example, training
on datasets that contain all-zero pixels, such as CIFAR-10.) Also
note that ReLUs are only partially supported by DKS, and unsupported by TAT, and
so their use is *highly* discouraged. Instead, one should use Leaky ReLUs, which
are fully supported by DKS, and work especially well with TAT.

## Example

`dks.examples.haiku.modified_resnet` is a [Haiku] ResNet model which has been
modified as described in the DKS/TAT papers, and includes support for both DKS
and TAT. When constructed with its default arguments, it removes the
normalization layers and skip connections found in standard ResNets, making it a
"vanilla network". It can be used as an instructive example for how to build
DKS/TAT models using this package. See the section titled "Application to
various modified ResNets" from the [DKS paper] for more details.

## Installation

This package can be installed directly from GitHub using `pip` with

```bash
pip install git+https://github.com/deepmind/dks.git
```

or

```bash
pip install -e git+https://github.com/deepmind/dks.git#egg=dks[<extras>]
```

Or from PyPI with

```bash
pip install dks
```

or

```bash
pip install dks[<extras>]
```

Here `<extras>` is a common-separated list of strings (with no spaces) that can
be passed to install extra dependencies for different tensor programming
frameworks. Valid strings are `jax`, `tf`, and `pytorch`. So for example, to
install `dks` with the extra requirements for JAX and PyTorch, one does

```bash
pip install dks[jax,pytorch]
```

## Testing

To run tests in a Python virtual environment with specific pinned versions of
all the dependencies one can do:

```bash
git clone https://github.com/deepmind/dks.git
cd dks
./test.sh
```

However, it is strongly recommended that you run the tests in the same Python
environment (with the same package versions) as you plan to actually use `dks`.
This can be accomplished by installing `dks` for all three tensors programming
frameworks (e.g. with `pip install dks[jax,pytorch,tf]` or some other
installation method), and then doing

```bash
pip install pytest-xdist
git clone https://github.com/deepmind/dks.git
cd dks
python -m pytest -n 16 tests
```

## Disclaimer

This is not an official Google product.

[DKS paper]: https://arxiv.org/abs/2110.01765
[TAT paper]: https://openreview.net/forum?id=U0k7XNTiFEq
[Haiku]: https://github.com/deepmind/dm-haiku
