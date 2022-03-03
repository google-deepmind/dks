![CI status](https://github.com/deepmind/dks/workflows/ci/badge.svg)
![pypi](https://img.shields.io/pypi/v/dks)

# Official Python package for Deep Kernel Shaping (DKS) and Tailored Activation Transformations (TAT)

This Python package implements the activation function transformations and
weight initializations used Deep Kernel Shaping (DKS) and Tailored Activation
Transformations (TAT). DKS and TAT, which were introduced in the [DKS paper] and
[TAT paper], are methods constructing/transforming neural networks to make them
much easier to train. For example, these methods can be used in conjunction with
K-FAC to train deep vanilla deep convnets (without skip connections or
normalization layers) as fast as standard ResNets of the same depth.

The package supports the JAX, PyTorch, and TensorFlow tensor programming
frameworks.

Questions/comments about the code can be sent to
[dks-dev@google.com](mailto:dks-dev@google.com).

**NOTE:** we are not taking code contributions from Github at this time. All PRs
from Github will be rejected. Instead, please email us if you find a bug.

## Usage

For each of the supported tensor programming frameworks, there is a
corresponding directory/subpackage which handles the activation function
transformations and weight initializations. It's up to the user to import these
and use them appropriately within their model code. Activation functions are
transformed by the function `get_transformed_activations()` in module
`activation_transform` of the appropriate subpackage. Weight sampling is done
using functions inside of the module `parameter_sampling_functions` of said
subpackage.

In addition to using these functions, the user is responsble for ensuring that
their model meets the architectural requirements of DKS/TAT, and for converting
any weighted sums in their model to "normalized sums" (which are weighted sums
whoses non-trainable weights have a sum of squares equal to 1). This package
doesn't currently include an implementation of Per-Location Normalization (PLN)
data pre-processing. While not required for CIFAR or ImageNet, PLN could
potentially be important for other datasets. See the section titled "Summary of
our method" in the [DKS paper] for more details about the requirements and
execution steps of DKS. To read about the additional requirements of TAT, such
as the subset maximizing function, refer to Appendix B of the [TAT paper].

Note that ReLUs are only partially supported by DKS, and unsupported by TAT, and
their use is *highly* discouraged. Instead, one should use Leaky ReLUs, which
are fully supported by DKS, and work especially well with TAT.

Note that in order to avoid having to import all of the tensor programming
frameworks, the user is required to individually import whatever framework
subpackage they want. e.g. `import dks.jax`. Meanwhile, `import dks` won't
actually do anything.

## Example

`dks.examples.haiku.modified_resnet` is a Haiku ResNet model which has been
modified as described in the DKS/TAT papers, and includes support for both DKS
and TAT. By default, it removes the normalization layers and skip connections
found in standard ResNets, making it a "vanilla network". It can be used as an
instructive example for how to build DKS/TAT models using this package. See the
section titled "Application to various modified ResNets" from the [DKS paper]
for more details.

## Installation

This package can be installed directly from GitHub using `pip` with

```bash
pip install git+https://github.com/deepmind/dks.git
```

or

```bash
pip install -e git+https://github.com/deepmind/dks.git#egg=dks[<extras>]
```

or from PyPI with

```bash
pip install dks
```

or

```bash
pip install dks[<extras>]
```

Here `<extras>` is a common-separated list (with no spaces) of strings that can
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
