import numpy as np
from typing import TYPE_CHECKING

# workaround for circular type-hinting
if TYPE_CHECKING:
    from model import Layer

class Optimizer:
    def update(self, lr: float, layer: 'Layer'):
        """
        Update parameters of given layer.
        """
        raise NotImplementedError("update() function must be implemented.")
    
    def reset(self, layer):
        """
        Clear all optimizer-specific accumulators for given layer.
        """
        pass

class GD(Optimizer):
    def update(self, lr: float, layer: 'Layer'):
        layer.B -= lr * layer.B_grad
        layer.W -= lr * layer.W_grad

class SGD(Optimizer):
    def __init__(self, momentum: float=0.001):
        self.momentum = momentum

    def update(self, lr: float, layer: 'Layer'):
        if not hasattr(layer, 'W_velocity'):
            layer.W_velocity = np.zeros_like(layer.W)
            layer.B_velocity = np.zeros_like(layer.B)

        # update velocities
        layer.W_velocity = self.momentum * layer.W_velocity + layer.W_grad
        layer.B_velocity = self.momentum * layer.B_velocity + layer.B_grad
        
        # update parameters
        layer.W -= lr * layer.W_velocity
        layer.B -= lr * layer.B_velocity

    def reset(self, layer: 'Layer'):
        if hasattr(layer, 'W_velocity'):
            del layer.W_velocity
        if hasattr(layer, 'B_velocity'):
            del layer.B_velocity

class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1  # global timestep counter for Adam decay

    def update(self, lr: float, layer: 'Layer'):
        """
        Perform the Adam update on current layer.

        g_t: parameter gradient at timestep t
        m_t: 1st moment at timestep t - moving average of gradients
        v_t: 2nd moment at timestep t - moving average of gradients squared (like a variance)

        m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
        v_t = β_2 * v_{t-1} + (1 - β_2) * (g_t)^2
        m_hat_t = m_t / [1 - (β_1)^t]
        v_hat_t = v_t / [1 - (β_2)^t]
        θ_{t+1} = θ_t - lr * m_hat_t / [sqrt(v_hat_t) + ε]
        """
        if not hasattr(layer, 'mW'):
            layer.mW = np.zeros_like(layer.W)
            layer.vW = np.zeros_like(layer.W)
            layer.mB = np.zeros_like(layer.B)
            layer.vB = np.zeros_like(layer.B)

        # assume most recent (averaged) gradients are in dW and dB
        layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.W_grad
        layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.W_grad**2)

        layer.mB = self.beta1 * layer.mB + (1 - self.beta1) * layer.B_grad
        layer.vB = self.beta2 * layer.vB + (1 - self.beta2) * (layer.B_grad**2)

        # bias correction
        mW_hat = layer.mW / (1 - self.beta1**self.t)
        vW_hat = layer.vW / (1 - self.beta2**self.t)
        mB_hat = layer.mB / (1 - self.beta1**self.t)
        vB_hat = layer.vB / (1 - self.beta2**self.t)

        # parameter update
        layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.B -= lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)

    def reset(self, layer: 'Layer'):
        for attr in ['mW', 'vW', 'mB', 'vB']:
            if hasattr(layer, attr):
                delattr(layer, attr)
        self.t = 1  # reset global timestep

class Lion(Optimizer):
    def __init__(self, beta: float=0.9):
        self.beta = beta

    def update(self, lr: float, layer: 'Layer'):
        """
        Perform the Lion update on current Layer.

        g = parameter gradient
        m = 1st moment

        m = βm + (1 - β)g
        θ = θ - lr * sgn(m)
        """
        if not hasattr(layer, 'mW'):
            layer.mW = np.zeros_like(layer.W)
            layer.mB = np.zeros_like(layer.B)
        
        # most recent (batch averaged) gradients are in layer.W_grad and layer.B_grad
        layer.mW = self.beta * layer.mW + (1 - self.beta) * layer.W_grad
        layer.mB = self.beta * layer.mB + (1 - self.beta) * layer.B_grad
        
        # use sign of moment to update parameters
        layer.W -= lr * np.sign(layer.mW)
        layer.B -= lr * np.sign(layer.mB)

    def reset(self, layer: 'Layer'):
        for attr in ['mW', 'mB']:
            if hasattr(layer, attr):
                delattr(layer, attr)
