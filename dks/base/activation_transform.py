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

"""NumPy implementation of the activation transformations used in DKS/TAT."""

import itertools
import os

from absl import logging

from autograd import elementwise_grad as egrad
from autograd import numpy as np
from dks.base.activation_getter import get_activation_function as _get_numpy_activation_function
import scipy.integrate as sp_int
import scipy.optimize as sp_opt
from scipy.special import roots_legendre

# pylint: disable=g-import-not-at-top
# This is a trick to achieve compatibility with multiple versions of SciPy:
try:
  from scipy.integrate._quadrature import _cached_roots_legendre
except ImportError:
  from scipy.integrate.quadrature import _cached_roots_legendre
# pylint: enable=g-import-not-at-top


# Random seed used to initialize activation function parameter searches after
# standard starting points fail (which almost never happens).
_RANDOM_SEED = 123

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Feel free to change the path of this file if needed. This file should ideally
# persist between experiments since it can take around 20 minutes to regenerate.
_ROOTS_CACHE_FILE = os.path.join(_CURRENT_DIR, "roots_{}.npy")

_QUADRATURE_ORDER = 100000

# These represent sensible values and ranges to initialize the activation
# function parameter search with. They are sensible because commonly used
# activation functions tend to have reasonable behavior for such values. If
# the searches fail for some activation function they may need to be
# tweaked/expanded.
_ALWAYS_TRY_VALUES = {"input_shift": (0.0, 1.0, -1.0),
                      "input_scale": (1.0, 0.1),
                      "output_shift": (0.0, 1.0, -1.0),
                      "negative_slope": (0.0, 0.5)}

_SAMPLE_RANGE_LOW = {"input_shift": -3.0,
                     "input_scale": 0.0,
                     "output_shift": -3.0,
                     "negative_slope": 0.0}

_SAMPLE_RANGE_HIGH = {"input_shift": 3.0,
                      "input_scale": 2.0,
                      "output_shift": 3.0,
                      "negative_slope": 1.0}


def _precompute_or_load_roots(order):
  """Compute or load the roots used by fixed_quad to save time."""

  if order not in _cached_roots_legendre.cache:

    roots_cache_file = _ROOTS_CACHE_FILE.format(order)

    if os.path.exists(roots_cache_file):

      with open(roots_cache_file, "rb") as fhandle:
        _cached_roots_legendre.cache[order] = np.load(fhandle,
                                                      allow_pickle=False)

    else:
      roots = roots_legendre(order)
      _cached_roots_legendre.cache[order] = roots

      with open(roots_cache_file, "wb") as fhandle:
        np.save(fhandle, roots, allow_pickle=False)


def _estimate_gaussian_mean(fn, order=_QUADRATURE_ORDER):
  """Estimate the mean of a function fn(x) where x ~ N(0,1)."""

  _precompute_or_load_roots(order)

  fn_weighted = lambda x: np.exp(-x**2 / 2) * fn(x)

  integral, _ = sp_int.fixed_quad(fn_weighted, -10., 10., n=order)

  return integral / np.sqrt(2*np.pi)


def _calc_c_map(activation, derivative_order=0, c=1.0, q_output=None):
  """Evaluate local C map value assuming an input q value of 1.

  Args:
    activation: A callable representing the activation function (applied
      elementwise) from which to define the local C map.
    derivative_order: An integer giving the order of the derivative of the C map
      to take before evaluating it. (Default: 0)
    c: A float giving the input point at which to evaluate the C map. Must
      be 0.0 or 1.0. (Default: 1.0)
    q_output: Float or None giving the output q value associated with
      ``activation``, if this is known. If None this will be computed from
      scratch. (Default: None)

  Returns:
     A float giving the value of the (derivative of) the local C map for the
     given activation function.
  """

  derivative = activation
  for _ in range(derivative_order):
    derivative = egrad(derivative)

  if c == 0.0:
    integral = _estimate_gaussian_mean(derivative)**2
  elif c == 1.0:
    integral = _estimate_gaussian_mean(lambda x: derivative(x)**2)
  else:
    raise NotImplementedError("Input c value must be 0.0 or 1.0.")

  if q_output is None:
    q_output = _estimate_gaussian_mean(lambda x: activation(x)**2)

  return integral / q_output


def _calc_c_slope(activation, c=1.0, q_output=None):
  """Evaluate local C map derivative assuming an input q value of 1."""
  return _calc_c_map(activation, derivative_order=1, c=c, q_output=q_output)


def _calc_c_curv(activation, c=1.0, q_output=None):
  """Evaluate local C map second derivative assuming an input q value of 1."""
  return _calc_c_map(activation, derivative_order=2, c=c, q_output=q_output)


def _calc_q_slope_1(activation):
  """Computes the derivative of a local Q map at q=1."""
  derivative = egrad(activation)
  return _estimate_gaussian_mean(lambda x: activation(x)*derivative(x)*x)


def _leaky_relu_cmap(c, negative_slope):
  """Evaluates the local C map for Leaky ReLU with the given negative slope."""
  return ((1 - negative_slope)**2 * (np.sqrt(1 - c**2)
                                     + (np.pi - np.arccos(c)) * c) / np.pi
          + 2 * negative_slope * c) / (1 + negative_slope**2)


def _compute_output_params(act, local_c_val_0_target):
  """Compute output params to achieve Q(1)=1 and C(0)=local_c_val_0_target."""

  if local_c_val_0_target is not None:
    output_shift = np.sqrt(local_c_val_0_target) - _estimate_gaussian_mean(act)
    act_shifted = lambda x: act(x) + output_shift
  else:
    output_shift = None
    act_shifted = act

  output_scale = 1. / np.sqrt(
      _estimate_gaussian_mean(lambda x: act_shifted(x)**2))

  return output_shift, output_scale


def _transform_activation(phi, params):
  """Transform an activation function phi using the given parameters."""

  params = params.copy()

  input_scale = params.pop("input_scale", None)
  input_shift = params.pop("input_shift", None)
  output_shift = params.pop("output_shift", None)
  output_scale = params.pop("output_scale", None)

  def activation(x):

    # Note: DO NOT use += and *= below! Bad things will happen.
    if input_scale is not None:
      x = x * input_scale
    if input_shift is not None:
      x = x + input_shift

    x = phi(x, **params)

    if output_shift is not None:
      x = x + output_shift
    if output_scale is not None:
      x = x * output_scale

    return x

  return activation


def _solve_for_activation_params(
    name, local_q_slope_target, local_c_val_0_target, local_c_slope_1_target,
    local_c_slope_0_target, local_c_curv_target, reject_condition=None,
    num_tries=50):
  """Computes activation function parameters to achieve the given targets."""

  # Making sure random starting points used in solvers will be the same for
  # each run, so that they find the same exact solutions (which is important
  # when using JAX with multiple processes).
  np.random.seed(_RANDOM_SEED)

  # RELU has 1 less degree of freedom so we use this special-case logic:
  if name == "relu":
    # The constant below is the maximum value of local_c_slope_1 that can be
    # achieved in ReLUs given a positive input shift value.
    if local_c_slope_1_target < 1.466942206924260361:
      input_shift = 1.0
    else:
      input_shift = -1.0

    constant_params = {"input_shift": input_shift}
    opt_params = ("input_scale",)

    local_q_slope_target = None  # we turn this off for ReLUs
    assert local_c_slope_0_target is None
    assert local_c_curv_target is None

  elif name == "leaky_relu":
    constant_params = {}
    opt_params = ("negative_slope", "input_shift")

  elif local_c_val_0_target is not None:
    constant_params = {}
    opt_params = ("input_scale", "input_shift")

  else:
    constant_params = {}
    opt_params = ("input_scale", "input_shift", "output_shift")

  def make_params_from_pvector(p):
    """Make a dictionary of activation parameters value from a given vector."""

    params = constant_params.copy()

    for i, pname in enumerate(opt_params):
      assert pname not in params
      params[pname] = p[i]

    # Here we compute output_scale, and sometimes also output_shift, separately
    # instead of taking them the vector.
    output_shift, output_scale = _compute_output_params(
        _transform_activation(_get_numpy_activation_function(name), params),
        local_c_val_0_target)

    if output_shift is not None:
      assert "output_shift" not in params
      params["output_shift"] = output_shift

    if output_scale is not None:
      assert "output_scale" not in params
      params["output_scale"] = output_scale

    return params

  def compute_error_vector(p):
    """Computes vector of errors (value - target) over relevant quantities."""

    params = make_params_from_pvector(p)
    phi_hat = _transform_activation(_get_numpy_activation_function(name),
                                    params)

    residual_array = []

    if local_q_slope_target is not None:
      local_q_slope = _calc_q_slope_1(phi_hat)
      residual_array += [local_q_slope - local_q_slope_target]

    if local_c_slope_1_target is not None:
      local_c_1_slope = _calc_c_slope(phi_hat, c=1.0, q_output=1.0)
      residual_array += [local_c_1_slope - local_c_slope_1_target]

    if local_c_slope_0_target is not None:
      local_c_0_slope = _calc_c_slope(phi_hat, c=0.0, q_output=1.0)
      residual_array += [local_c_0_slope - local_c_slope_0_target]

    if local_c_curv_target is not None:
      local_c_curv = _calc_c_curv(phi_hat, c=1.0, q_output=1.0)
      residual_array += [local_c_curv - local_c_curv_target]

    return np.asarray(residual_array)

  # Make the starting points for the search:
  always_try = tuple(_ALWAYS_TRY_VALUES[pname] for pname in opt_params)
  starting_points = list(itertools.product(*always_try))
  for _ in range(num_tries - len(starting_points)):
    starting_points += [
        tuple(np.random.uniform(low=_SAMPLE_RANGE_LOW[pname],
                                high=_SAMPLE_RANGE_HIGH[pname])
              for pname in opt_params)]

  # For each starting point we run sp_opt.root to try and find a solution:
  for starting_point in starting_points:

    sol = sp_opt.root(compute_error_vector, np.asarray(starting_point),
                      method="hybr", jac=False, options=None)
    if sol.success:
      params = make_params_from_pvector(sol.x)
      if reject_condition is None or not reject_condition(name, params):
        break

    logging.debug("Failed to find parameters from starting point %s.",
                  starting_point)

  if not sol.success:
    raise ValueError(f"Failed to find parameters for '{name}'!")

  logging.info("Found parameters for '%s': %s", name, params)

  return params


def _solve_increasing(fn, target, input_, min_, max_, tol=1e-8, max_eval=100):
  """Solves for x in fn(x)=target, where fn is an increasing function.

  Args:
    fn: A callable which takes a scalar input x and produces a scalar output.
      Must compute an increasing function.
    target: The target output value of ``fn``.
    input_: The initial guess for x.
    min_: A lower bound on the possible value of x.
    max_: An upper bound on the possible value of x.
    tol: A float which giving the acceptable tolerance for ``|fn(x) - target|``.
      (Default: 1e-8)
    max_eval: An integer giving the maximum number of times to evaluate ``fn``
      before giving up. (Default: 100)

  Returns:
    A float giving a value of x such that ``|fn(x) - target| < tol``.
  """

  # The method used to find the solution is a simple binary search in the
  # inteval [min_, max_], where max_ or min_ will change each iteration. If
  # max_ is infinity we double our current guess instead of averaging it with
  # max_.
  for _ in range(max_eval):

    value = fn(input_)

    logging.debug("binary search vals: min = %f, input = %f, max = %f, "
                  "target = %f, value = %f", min_, input_, max_, target, value)

    if np.abs(value - target) < tol:
      return input_

    if value > target:
      max_ = input_
      input_ = 0.5 * (input_ + min_)

    elif value < target:
      min_ = input_

      if np.isinf(max_):
        input_ = input_ * 2
      else:
        input_ = 0.5 * (input_ + max_)

  raise ValueError(f"Maximum evaluations ({max_eval}) exceeded while searching "
                   "for solution. This is probably due the specified targets "
                   "being unachievable for the given architecture.")


def _compute_local_c_slope_1_target(max_slope_func, target_value):
  return _solve_increasing(max_slope_func, target_value, 1.1, 1., np.inf)


def _compute_local_c_slope_0_target(max_slope_func, target_value):
  return _solve_increasing(max_slope_func, target_value, 0.99, 0., 1.0)


def _compute_local_c_curv_target(max_curv_func, target_value):
  return _solve_increasing(max_curv_func, target_value, 0.1, 0., np.inf)


def _compute_negative_slope_param(max_lrelu_c0, target_value):
  return _solve_increasing(
      lambda a: 1.0 - max_lrelu_c0(a), 1.0 - target_value, 0.5, 0., 1.0)


def _verify_params_dict_and_set_defaults(params_dict, defaults):
  """Verify keys in parameter dict and set any missing ones to the defaults."""

  bad_keys = set(params_dict.keys()).difference(set(defaults.keys()))

  if bad_keys:
    raise ValueError(
        f"Parameter dictionary had unrecognized keys: '{bad_keys}'")

  for key in defaults.keys():
    if key not in params_dict:
      params_dict[key] = defaults[key]


def _get_activations_params(
    activation_names, method="DKS", dks_params=None, tat_params=None,
    max_slope_func=None, max_curv_func=None, subnet_max_func=None):
  """Get dict of optimized parameters for given named activation functions."""

  if not isinstance(activation_names, (list, tuple)):
    raise ValueError("activation_names argument must be a list or tuple of "
                     "strings.")

  # Note that using dictionaries as defaults in the function def is bad, hence
  # we do this instead:
  if dks_params is None:
    dks_params = {}
  if tat_params is None:
    tat_params = {}

  _verify_params_dict_and_set_defaults(
      dks_params, {"c_slope_1_target": 1.5, "local_q_slope_target": 1.0})
  _verify_params_dict_and_set_defaults(
      tat_params, {"c_val_0_target": 0.9, "c_curve_target": 0.3})

  local_q_slope_target = None
  local_c_val_0_target = None
  local_c_slope_1_target = None
  local_c_slope_0_target = None
  local_c_curv_target = None
  reject_condition = None

  if method == "DKS":

    if "relu" in activation_names:
      logging.warning("The use of ReLUs with DKS is *highly* discouraged. You "
                      "are advised to use Leaky ReLUs instead.")

    c_slope_1_target = dks_params["c_slope_1_target"]

    if c_slope_1_target <= 1.0:
      raise ValueError("Invalid value for DKS 'c_slope_1_target' parameter. "
                       "Must be a float greater than 1.0.")

    if max_slope_func is None:
      if subnet_max_func is None:
        raise ValueError("Must supply 'subnet_max_func' if using DKS and not "
                         "passing in 'max_slope_func'.")
      # We can compute the maximal slope function by replacing composition
      # with multiplication in the maximal c value function.
      max_slope_func = lambda x: subnet_max_func(1.0, lambda y: x * y)

    # Three of the four conditions used by DKS. The remaining condition Q(1) = 1
    # is implied.
    local_q_slope_target = dks_params["local_q_slope_target"]
    local_c_val_0_target = 0.0
    # We set the local slope to achieve C'_f(1) <= target over all subnetworks
    # f:
    local_c_slope_1_target = _compute_local_c_slope_1_target(
        max_slope_func, c_slope_1_target)

    logging.info("Found 'local_c_slope_1_target': %s", local_c_slope_1_target)

  elif method == "TAT" and "leaky_relu" not in activation_names:

    if "relu" in activation_names:
      raise ValueError("Standard ReLU not supported with TAT. Use leaky "
                       "ReLU instead.")

    c_curve_target = tat_params["c_curve_target"]

    if c_curve_target <= 0.0:
      raise ValueError("Invalid value for TAT 'c_curve_target' parameter. Must "
                       "be a float greater than 0.0.")

    if max_curv_func is None:
      if subnet_max_func is None:
        raise ValueError("Must supply 'subnet_max_func' if using TAT with "
                         "smooth activations and not passing in "
                         "'max_curv_func'.")
      # We can compute the maximal curvature function by replacing composition
      # with addition in the maximal c value function.
      max_curv_func = lambda x: subnet_max_func(0.0, lambda y: x + y)

    # Three of the four conditions used by TAT in the smooth case. The remaining
    # condition Q(1) = 1 is implied.
    local_q_slope_target = 1.0
    local_c_slope_1_target = 1.0
    # We set the local second derivative to achieve C''_f(1) <= target over all
    # subnetworks f:
    local_c_curv_target = _compute_local_c_curv_target(
        max_curv_func, c_curve_target)

    logging.info("Found 'local_c_curv_target': %s", local_c_curv_target)

    # This is a hacky fix used to avoid certain 'bad' solutions we observed that
    # seem to have unstable Q maps and higher kernel approximation errors. It
    # should probably be replaced with something more principled.
    reject_condition = lambda name, params: (  # pylint: disable=g-long-lambda
        params["input_scale"] * params["output_scale"] >= 2.0)

  elif method == "TAT" and "leaky_relu" in activation_names:

    if len(activation_names) > 1:
      raise ValueError("When using Leaky ReLU with TAT it must be the only "
                       "activation function.")

    c_val_0_target = tat_params["c_val_0_target"]
    if c_val_0_target > 1.0 or c_val_0_target < 0.0:
      raise ValueError("Invalid value for TAT 'c_val_0_target' parameter. Must "
                       "be a float between 0.0 and 1.0.")

    if subnet_max_func is None:
      raise ValueError("Must supply 'subnet_max_func' if using TAT with Leaky "
                       "ReLU activation functions.")

    max_lrelu_c0 = lambda neg_slope: subnet_max_func(  # pylint: disable=g-long-lambda
        0.0, lambda c: _leaky_relu_cmap(c, neg_slope))

    # We set the negative slope parameter to achieve C_f(0) <= target over all
    # subnetworks f:
    negative_slope = _compute_negative_slope_param(
        max_lrelu_c0, c_val_0_target)

    # This is the value required to achieve Q(1) = 1 for Leaky ReLUs. See the
    # TAT paper for details.
    output_scale = np.sqrt(2.0 / (1.0 + negative_slope**2))

    logging.info("Found parameters for 'leaky_relu': negative_slope = %s, "
                 "output_scale = %s.", negative_slope, output_scale)

    return {"leaky_relu": {"output_scale": output_scale,
                           "negative_slope": negative_slope}}

  else:
    raise ValueError(f"Unrecognized value for argument 'method': {method}")

  params = {}
  for name in activation_names:

    params[name] = _solve_for_activation_params(
        name,
        local_q_slope_target=local_q_slope_target,
        local_c_val_0_target=local_c_val_0_target,
        local_c_slope_1_target=local_c_slope_1_target,
        local_c_slope_0_target=local_c_slope_0_target,
        local_c_curv_target=local_c_curv_target,
        reject_condition=reject_condition)

  return params


def get_transformed_activations(
    activation_names, method="DKS", dks_params=None, tat_params=None,
    max_slope_func=None, max_curv_func=None, subnet_max_func=None,
    activation_getter=_get_numpy_activation_function):
  """Gets transformed activation functions using the DKS or TAT method.

  See the DKS paper (https://arxiv.org/abs/2110.01765) and the TAT paper
  (https://openreview.net/forum?id=U0k7XNTiFEq) for details about what these
  are, how they are computed, and what their parameters mean.

  Note that if you are using the JAX, PyTorch, or TensorFlow frameworks, you
  probably want to be using the version of get_transformed_activations() in the
  corresponding subpackage. (These are basically thin wrappers around this
  function that pass a framework-specific value to the ``activation_getter``
  argument.)

  Args:
    activation_names: An iterable of string names for the activation functions.
      Supported names are the intersection of those supported by
      dks.base.activation_getter.get_activation_function, and those supported
      by the getter passed to the ``activation_getter`` argument (which defaults
      to dks.base.activation_getter.get_activation_function). The built-in
      getters in this package (for each framework) currently support the
      following names: "tanh", "softplus", "leaky_relu", "relu" (not
      recommended; use "leaky_relu" instead), "selu", "elu", "swish", "sigmoid",
      "erf", "bentid", "atan", "asinh", "square", and "softsign".
    method: A string representing the method used to transform the activation
      functions. Can be "DKS", "TAT", or "untransformed". The latter choice
      will return activation functions without any transformations.
      (Default: "DKS")
    dks_params: A dictionary containing the parameters to use for DKS. Keys
      should be a subset of {"c_slope_1_target", "local_q_slope_target"}.
      "c_slope_1_target" gives the target maximal slope value for the network
      (corresponding to "zeta" from the paper), and defaults to 1.5.
      "local_q_slope_target" gives the target value for the local Q map slope
      of each nonlinear layer (which is kept at 1.0 in the paper -- except in
      ablation tests), and defaults to 1.0. If ``dks_params`` is passed as None,
      it defaults to the empty dictionary (so that the parameters will use their
      default values). (Default: None)
    tat_params: A dictionary containing the parameters to use for TAT. Keys
      should be a subset of {"c_val_0_target", "c_curve_target"}.
      "c_val_0_target" gives the maximum value of ``C_f(0)`` over subnetworks f,
      which is used when transforming Leaky ReLUs (and corresponds to "eta" from
      the paper), and defaults to 0.9. "c_curve_target" gives the maximum value
      of ``C''_f(1)``, which is used for all other activation functions (and
      corresponds to "tau" from the paper), and defaults to 0.3. If
      ``tat_params`` is passed as None, it defaults to the empty dictionary (so
      that the parameters will use their default values). (Default: None)
    max_slope_func: A callable which computes the maximal slope function, as
      defined in the DKS paper. It should take a single argument representing
      the slope of each local C map at ``c=1``. If this is required (i.e. when
      using DKS) but is passed as None, it will be generated using
      ``subnet_max_func`` if possible. (Default: None)
    max_curv_func: A callable which computes the maximal curvature function. It
      should take a single parameter representing the second derivative of each
      local C map at c=1. If this is required (i.e. when using TAT with smooth
      activation functions) but is passed as None, it will be generated using
      ``subnet_max_func`` if possible. (Default: None)
    subnet_max_func: A callable which computes the subnetwork maximizing
      function of the network (denoted ``M_{f,r}(x)`` in the TAT paper). It
      should take two arguments: the input value ``x``, and a callable
      ``r_func`` which maps a float to a float. This is required when using TAT
      with Leaky ReLUs. (Default: None)
    activation_getter: A callable which takes a string name for an activation
      function and returns the (untransformed) activation function corresponding
      to this name. Defaults to one returning activation functions in NumPy
      (with autograd). Returned transformed activation functions will be based
      on the output of this callable. Other tensor frameworks can be supported
      by changing this argument. See the versions of
      get_transformed_activations() in the ``dks.jax``, ``dks.pytorch``, and
      ``dks.tensorflow`` subpackages.

  Returns:
    A dictionary mapping the activation function names to their corresponding
    transformed activation functions.
  """

  if method == "untransformed":
    return {name: activation_getter(name) for name in activation_names}

  params = _get_activations_params(
      activation_names, method=method, dks_params=dks_params,
      tat_params=tat_params, max_slope_func=max_slope_func,
      max_curv_func=max_curv_func, subnet_max_func=subnet_max_func)

  transformed_acts = {}

  for name in activation_names:
    transformed_acts[name] = _transform_activation(activation_getter(name),
                                                   params[name])
  return transformed_acts
