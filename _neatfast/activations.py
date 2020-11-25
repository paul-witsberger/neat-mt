"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""
from __future__ import division
import math
import types


def sigmoid_activation(z):
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    return math.tanh(z)


def sin_activation(z):
    return math.sin(math.pi * z)


def gauss_activation(z):
    return math.exp(-5.0 * z ** 2) if abs(z) < 1.4 else 0.0


def relu_activation(z):
    return z if z > 0.0 else 0.0


def elu_activation(z):
    return z if z > 0.0 else math.exp(z) - 1


def lelu_activation(z):
    leaky = 0.005
    return z if z > 0.0 else leaky * z


def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)


def softplus_activation(z):
    return 0.2 * math.log(1 + math.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z * 2))


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z):
    # z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


def step_activation(z):
    return 1.0 if z > 0.0 else 0.0


def multistep_activation(z):
    if z > 1.:
        return 1.0
    elif z > 0.0:
        return 2 / 3.
    elif z > -1.:
        return 1 / 3.
    else:
        return 0.

def multiramp_clamp_activation(z):
    level_1 = -1.
    level_2 = -1. / 3
    level_3 =  1. / 3
    level_4 =  1.
    if z > level_4:
        return 1.
    elif z > level_3:
        return 2. / 3 + (z - level_3) / (level_4 - level_3) / 3
    elif z > level_2:
        return 1. / 3 + (z - level_2) / (level_3 - level_2) / 3
    elif z > level_1:
        return (z - level_1) / (level_2 - level_1) / 3
    else:
        return 0.


class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        self.add('elu', elu_activation)
        self.add('lelu', lelu_activation)
        self.add('selu', selu_activation)
        self.add('softplus', softplus_activation)
        self.add('identity', identity_activation)
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)
        self.add('step', step_activation)
        self.add('multistep', multistep_activation)

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
