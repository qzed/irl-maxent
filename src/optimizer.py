import numpy as np


class Optimizer:
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        return NormalizeGrad(self, ord)


class Sga(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters += lr * grad


class ExpSga(Optimizer):
    def __init__(self, lr, normalize=True):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, x, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad * x)

        if self.normalize:
            self.parameters /= self.parameters.sum()

        # TODO: compare exp. gradient descent with the one from bziebart's thesis:
        # theta = theta * np.exp(learning_rate/(k+1) * grad)


class NormalizeGrad(Optimizer):
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        return self.opt.step(grad / np.linalg.norm(grad, self.ord), *args, **kwargs)


def linear_decay(lr0=0.2, decay_rate=0.5, decay_steps=1.0):
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr


def power_decay(lr0=0.2, decay_rate=1.0, decay_steps=1.0, power=2):
    def _lr(k):
        return lr0 / (decay_rate * np.floor(k / decay_steps) + 1.0)**power

    return _lr


def exponential_decay(lr0=0.2, decay_rate=0.5, decay_steps=1.0):
    def _lr(k):
        return lr0 * np.exp(-decay_rate * np.floor(k / decay_steps))

    return _lr


class Initializer:
    def __init__(self):
        pass

    def initialize(self, shape):
        raise NotImplementedError

    def __call__(self, shape):
        return self.initialize(shape)


class Uniform(Initializer):
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def initialize(self, shape):
        return np.random.uniform(size=shape, low=self.low, high=self.high)


class Constant(Initializer):
    def __init__(self, value=1.0, fn=None):
        super().__init__()
        self.value = value
        self.fn = fn

    def initialize(self, shape):
        if self.fn is not None:
            return np.ones(shape) * self.fn(shape)
        else:
            return np.ones(shape) * self.value
