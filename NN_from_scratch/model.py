from typing import List, Callable, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

class Linear:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.array(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)
        
class ELU:
    def __call__(self, x: np.ndarray, alpha=1.) -> np.ndarray:
        x = np.clip(x, -100, 100)  # prevent overflow
        return np.where(x > 0, x, alpha * (np.expm1(x)))
        
    def derivative(self, x: np.ndarray, alpha=1.) -> np.ndarray:
        elu_x = self.__call__(x)
        return np.where(x > 0, 1, elu_x + alpha)
    
    def second_deriv(self, x: np.ndarray, alpha: float = 1.) -> np.ndarray:
        x = np.clip(x, -100, 100)
        return np.where(x > 0, 0.0, alpha * np.exp(x))

    def third_deriv(self, x: np.ndarray, alpha: float = 1.) -> np.ndarray:
        # for x > 0, zero; for x <= 0 same as second deriv
        x = np.clip(x, -100, 100)
        return np.where(x > 0, 0.0, alpha * np.exp(x))
    
class Tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1.0 - tanh_x ** 2)
    
    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        tanh_x = np.tanh(x)
        return -2 * (1 - tanh_x**2) * (1 - 3 * tanh_x**2)
    
class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -100, 100)  # prevent overflow
        return 1 / (1 + np.exp(-np.array(x)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1.0 - sigmoid_x) * (1.0 - 2.0 * sigmoid_x)
    
    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self.__call__(x)
        return s * (1 - s) * (1 - 6 * s + 6 * s**2)
    
class BaseLayer:
    def __init__(self):
        self.A = None
        self.z = None
        self.previous = None
        self.next = None

    def forward(self):
        raise NotImplementedError('Method must be implemented for subclasses.')
    
    def backward(self):
        raise NotImplementedError('Method must be implemented for subclasses.')

class Layer(BaseLayer):
    def __init__(self, size: int, previous: 'Layer'=None, weight_init: str='xavier_uniform'):
        """
        Creates Layer of {size} neurons.

        Args:
            size: number of neurons in layer.
            batch_dim (optional): dimension of batches for training, set to 1 default.
            previous (optional): pointer to previous layer.
            next (optional): pointer to next layer.
        """
        super().__init__()
        self.size = size

        self.W = None
        self.B = None
        self.delta = None
        self.activation_deriv = None

        # will accumulate gradients to then alter parameters by
        self.W_grad = []
        self.B_grad = []

        # choose from weight initalisation methods
        # biases always initialised to zero
        self.weight_init = weight_init

        self.weight_initialisations = {
            'normal': self.normal,
            'uniform': self.uniform,
            'xavier_uniform': self.xavier_uniform,
            'xavier_normal': self.xavier_normal
        }

        if previous:
            self.previous = previous
            self.W, self.B = self.construct_weights_and_biases()

    @staticmethod
    def normal(out_dim: int, in_dim: int, scale: float=0.01) -> np.ndarray:
        """Weights initialised from normal distribution scaled by 0.01"""
        return np.random.randn(out_dim, in_dim) * scale
    
    @staticmethod
    def uniform(out_dim: int, in_dim: int, lo=-0.05, hi=0.05) -> np.ndarray:
        """Weights initialised from uniform distribution between lo and hi."""
        return np.random.uniform(lo, hi, size=(out_dim, in_dim))
    
    @staticmethod
    def xavier_uniform(out_dim: int, in_dim: int) -> np.ndarray:
        """Xavier-Glorot intialised from uniform distribution"""
        n = in_dim + out_dim
        return np.random.uniform(-np.sqrt(6 / n), np.sqrt(6 / n), (out_dim, in_dim))

    @staticmethod
    def xavier_normal(out_dim: int, in_dim: int) -> np.ndarray:
        """Xavier-Glorot initialised from normal distribution"""
        std = np.sqrt(2. / (in_dim + out_dim))
        return np.random.normal(loc=0.0, scale=std, size=(out_dim, in_dim))

    def assign_previous_layer(self, previous_layer: 'Layer') -> None:
        """
        Assigns current layer's previous pointer to {next_layer}.
        Assigns {previous_layer}'s previous pointer to current layer.

        Generate edges to connect all neurons in current layer to {previous_layer}.
        """
        self.previous = previous_layer
        self.previous.next = self
        self.W, self.B = self.construct_weights_and_biases()

    def construct_weights_and_biases(self) -> List[np.ndarray]:
        """
        For the l^th layer we have weight matrix W^{l} and bias vector b^{l}.
        These operate on the attentions of previous layer a^{l-1}.
            z^{l} = W^{l} * a^{l-1} + b^{l}
            a^{l} = A(z^{l}) (activation function)

        NOTE: matrix & vectors defined in column form
            (shape = rows x columns)
            dim(W^l) = len(a^l) x len(a^l-1) 
        """
        if self.previous is None:
            raise ValueError("Previous layer not assigned.")
        
        # traces back to last Layer class to fetch size attribute
        # needs to skip through activation layers
        current_layer = self.previous
        while not isinstance(current_layer, Layer):
            if current_layer.previous is None:
                raise ValueError("Previous layer chain is broken.")
            current_layer = current_layer.previous
        
        prev_size = current_layer.size
        
        # Xavier-Glorot weight initialisation
        weights_tensor = self.weight_initialisations[self.weight_init](out_dim=self.size, in_dim=prev_size)
        biases_tensor = np.zeros((self.size,))

        return weights_tensor, biases_tensor
    
    def forward(self, input_data: np.ndarray=None, store_grads: bool=True) -> None:
        """
        Take activation vector a^{l-1}j from previous layer (should be an ActivationLayer)
        Perform z^{l}_i = w^{l}_ij a^{l-1}_j + b_i
        """
        if input_data is not None:
            A_prev = input_data
        else:
            A_prev = self.previous.A
                
        # contracting W tensor with activation tensor over previous.size dimension
        # allows for extra 'b' batch dimension

        # in case there is no batch dim
        if A_prev.ndim == 1:
            A_prev = A_prev[None, :]

        z = np.einsum('ij, bj -> bi', self.W, A_prev) + self.B

        # assert z.shape == (self.batch_dim, self.size)
        self.z = z
        self.activation_deriv = self.next.derivative(self.z)

        # calculate Jacobian for autograd, don't take mean over batch dimension
        # batch dimension left for Hessian calculation for 2nd order derivatives
        if store_grads:
            self.activation_second_deriv = self.next.second_deriv(self.z)
            self.g = np.einsum('bj, jk -> bjk', self.activation_deriv, self.W)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Layer passes backwards its error vector D^{l}_j = D^{l+1}_j * [w^{l+1}_jk * A'(z^{l}_k)]

        w^{l+1}_jk are weights of first downstream Layer (skipping ActivationLayers).

        A'(z^{l}_k) is calculated by the current layer's activation block which sets the
        'self.activation_deriv' attribute to the evaluated value.
    
        Previous layer should be activation layer to which D is passed to.

        NOTE: this method also performs the Jacobian calculations for autograd

        J^{l}_jk = A'(z^{l}_j) * w^{l}_jk

        This custom setup assumes that every Layer object has a subsequent ActivationLayer.
        Input layers are thus conventionally given a Linear ActivationLayer.

        Args:
            error: the error vector of the next Layer downstream
                shape: (batch_dim, current.size)
        """
        # for debugging, allow for smooth running but flag the error
        if self.activation_deriv is None:
            self.activation_deriv = 1
            print(f"Layer with size {self.size} did not receive an activation deriv.")

        # this error catch indicates a failure in layer linkage
        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')
        
        # next layer should be Activation if we are mid-stream
        current = self.next

        # if next layer is None, we are at an output layer WITHOUT activation after it
        if current is None:
            self.delta = error
            return error

        # if we aren't at the output, move forward until you find first Layer
        while not isinstance(current, Layer):
            # if next is None, we are at an output layer WITH activation after it
            if current.next is None:
                self.delta = error * self.activation_deriv
                return self.delta

            current = current.next
        # current should be the first downstream Layer whose 'error' vector this function received

        # rather than asserting equality 'error == current.delta' which may fail under float approximations
        # assert error.shape == (self.batch_dim, current.size)

        self.delta = np.einsum('ij, bi -> bj', current.W, error) * self.activation_deriv

        # used to propagate derivatives with respect to network inputs
        # shape: (batch_size, self.size, prev_size) then average over batch dimension

        # assert self.delta.shape == (self.batch_dim, self.size)
        
        return self.delta
    
class ActivationLayer(BaseLayer):
    def __init__(self, activation_function: Callable[[np.ndarray], np.ndarray]):
        """
        Wrapper class for Activations.
        Includes default methods for derivatives as many are zero.
        """
        super().__init__()
        self.activation_function = activation_function

    def forward(self, store_grads=True):
        if self.previous:
            self.A = self.activation_function(self.previous.z)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.activation_function.derivative(x)

    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        return getattr(self.activation_function, "second_deriv", lambda x: np.zeros_like(x))(x)

    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        return getattr(self.activation_function, "third_deriv", lambda x: np.zeros_like(x))(x)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        D^{l}_j = D^{l+1}_j * [w^{l+1}_jk * A'(z^{l}_k)]

        ActivationLayer passes backwards derivative evaluated on its activations
        multiplied by fed back error vector. This is: D^{l+1}_j * A'(z^{l}_k)
        """
        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')
        
        # # we call the previous layer's "pre-activation" z
        # deriv = self.derivative(self.previous.z)
        # self.previous.activation_deriv = deriv

        return error
    
class Optimizer:
    def update(self, lr, layer):
        """
        Update parameters of given layer.
        """
        raise NotImplementedError("update() function must be implemented.")
    
    def reset(self, layer):
        """
        Clear all optimizer-specific accumulators for given layer.
        """
        raise NotImplementedError("reset() function must be implemented.")

class GD(Optimizer):
    def update(self, lr: float, layer: Layer):
        layer.B -= lr * np.mean(layer.B_grad, axis=0)
        layer.W -= lr * np.mean(layer.W_grad, axis=0)

class SGD(Optimizer):
    def __init__(self, momentum: float=0.001):
        self.momentum = momentum

    def update(self, lr: float, layer: Layer):
        if not hasattr(layer, 'W_velocity'):
            layer.W_velocity = np.zeros_like(layer.W)
            layer.B_velocity = np.zeros_like(layer.B)

        # accumulate gradients (accounts for Physics loss)
        dW = np.mean(layer.W_grad, axis=0)
        dB = np.mean(layer.B_grad, axis=0)     

        # update velocities
        layer.W_velocity = self.momentum * layer.W_velocity + dW
        layer.B_velocity = self.momentum * layer.B_velocity + dB
        
        # update parameters
        layer.W -= lr * layer.W_velocity
        layer.B -= lr * layer.B_velocity

    def reset(self, layer: Layer):
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

    def update(self, lr: float, layer: Layer):
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

        # accumulate gradients (accounts for Physics loss)
        dW = np.mean(layer.W_grad, axis=0)
        dB = np.mean(layer.B_grad, axis=0)

        # assume most recent (averaged) gradients are in dW and dB
        layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * dW
        layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (dW**2)

        layer.mB = self.beta1 * layer.mB + (1 - self.beta1) * dB
        layer.vB = self.beta2 * layer.vB + (1 - self.beta2) * (dB**2)

        # bias correction
        mW_hat = layer.mW / (1 - self.beta1**self.t)
        vW_hat = layer.vW / (1 - self.beta2**self.t)
        mB_hat = layer.mB / (1 - self.beta1**self.t)
        vB_hat = layer.vB / (1 - self.beta2**self.t)

        # parameter update
        layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.B -= lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)

    def reset(self, layer: Layer):
        for attr in ['mW', 'vW', 'mB', 'vB']:
            if hasattr(layer, attr):
                delattr(layer, attr)
        self.t = 1  # reset global timestep

class Lion(Optimizer):
    def __init__(self, beta: float=0.9):
        self.beta = beta

    def update(self, lr: float, layer: Layer):
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
        
        # accumulate gradients (accounts for Physics loss)
        dW = np.mean(layer.W_grad, axis=0)
        dB = np.mean(layer.B_grad, axis=0)

        # assume most recent (averaged) gradients are in dW and dB
        layer.mW = self.beta * layer.mW + (1 - self.beta) * dW
        layer.mB = self.beta * layer.mB + (1 - self.beta) * dB
        
        # use sign of moment to update parameters
        layer.W -= lr * np.sign(layer.mW)
        layer.B -= lr * np.sign(layer.mB)

    def reset(self, layer: Layer):
        for attr in ['mW', 'mB']:
            if hasattr(layer, attr):
                delattr(layer, attr)

class Network:
    def __init__(self, layers: List[BaseLayer], optim: Optimizer, physics_loss_weight: float=0.001, batch_training: bool=True):
        """
        Initialise network with list of BaseLayers.
        First layer automatically interpreted as Input.
        Layers linked sequentially.

        Args:
            layers: list of network layers
            physiscs_loss_weight: the 'λ' weighting for the Physics Loss
            batch_training: network supports completely dynamic batch training (default set to True)
        """
        self.layers = layers
        self.input_dim = layers[0].size
        # assumes there is an output activation (ActivationLayers do not have a size)
        self.output_dim = layers[-2].size

        self.physics_loss_weight = physics_loss_weight
        self.optim = optim

        # link the layers
        self.setup()

        self.loss_functions = {
            'mse': self.MSELoss,
            'ce': self.CELoss
        }

        # self.batch_dim = layers[0].A.shape[0]
        self.batch_training = batch_training

    def setup(self) -> None:
        """
        Function to link layers together and intialise parameters.
        Can be used to reset the network for repeated testing.

        Also resets any pre-existing optimiser accumulated measures.
        """
        for i, layer in enumerate(self.layers):
            # input layer just holds the variables, does not have weights/biases
            if i == 0:
                continue
            # reset any existing optimiser parameters
            self.optim.reset(layer)

            layer.previous = self.layers[i-1]
            self.layers[i-1].next = layer
            if isinstance(layer, Layer):
                layer.assign_previous_layer(self.layers[i-1])

    def forward(self, input_data: np.ndarray, store_grads=True) -> np.ndarray:
        """
        Args:
            input_data shape: (batch_dim, input_layer.size)
        """
        # assert input_data.shape == self.layers[0].A.shape

        self.layers[0].z = input_data

        for layer in self.layers[1:]:
            layer.forward(store_grads=store_grads)
        return self.layers[-1].A
    
    def MSELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((x - y) ** 2, axis=-1)

    def CELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(x), axis=-1)
            
    def numerical_physics_loss(self, x: np.ndarray, h=1e-5) -> List[np.ndarray]:
        """
        Physics-informed loss enforcing d²y/dx² + y = 0
        Introduces additive λ[2L_p(1 - 2/h^2)] into loss derivative wrt final output a^{L}
        """
        # compute model predictions at x, x+h, and x-h
        # y = self.forward(x)
        # y_plus = self.forward(x + h)  
        # y_minus = self.forward(x - h)

        X_all = np.concatenate([x, x + h, x - h], axis=0)  # shape: (3 * batch_dim, input_dim)
        Y_all = self.forward(X_all)                       # returns shape: (3 * batch_dim, output_dim)

        y, y_plus, y_minus = np.split(Y_all, 3)
        
        # finite difference approximation of the second derivative
        residual = (y_plus + y_minus - 2*y) / h**2 + y
        grad_y = 2 * residual * (1 - 2 / h**2)      # dL/dy(x)
        grad_y_plus = 2 * residual * (1 / h**2)     # dL/dy(x+h)
        grad_y_minus = grad_y_plus                  # dL/dy(x-h)

        # NOTE squaring to ensure non-negative physics loss
        physics_loss = np.mean(residual ** 2)
        
        return physics_loss, grad_y, grad_y_plus, grad_y_minus

    def autograd(self) -> None:
        """
        Performs the iterative calculation for H and total J:
        """
        batch_size = self.layers[0].z.shape[0]

        # shape of J and H: (batch, input_dim, input_dim)
        overall_J = np.tile(np.eye(self.input_dim)[None, :, :], (batch_size, 1, 1))
        overall_H = np.zeros_like(overall_J)

        for layer in self.layers:
            if isinstance(layer, Layer) and hasattr(layer, 'g'):
                g_l = layer.g   # shape (batch, out_dim, in_dim)
                W = layer.W     # shape (out_dim, in_dim)

                A_prime = layer.activation_deriv
                A_double_prime = layer.activation_second_deriv

                # directional_deriv = w^{l}_jm * J^{l-1}_mk ignoring batch dim
                directional_deriv = np.einsum('oi, bij -> boj', W, overall_J)
                # A''^{l}(z^{l}_j)[directional_deriv]^2
                term1 = A_double_prime[:, :, None] * directional_deriv**2

                # propagated_H = w^{l}_jm * H^{l-1}_mk ignoring batch dim
                propagated_H = np.einsum('oi, bij -> boj', W, overall_H)
                # A'(z^{l}_j)[propagated_H]
                term2 = A_prime[:, :, None] * propagated_H

                overall_H = term1 + term2
                # J^{l} = g_{layer} * J^{l-1}
                overall_J = np.einsum('boi, bij -> boj', g_l, overall_J)

        # average over batch dimension
        self.J_batch = overall_J
        self.H_batch = overall_H
        # self.J = np.mean(overall_J, axis=0)
        # self.H = np.mean(overall_H, axis=0)

    def autograd_derivs(self) -> None:
        """
        Computes dJ/da & dH/da for a: final layer activations.
        Used for Physics Loss functions.

        NOTE: mathematical derivations in accompanying pdf
        """
        final_layer = self.layers[-1]
        final_z = self.layers[-2].z

        A_prime = final_layer.derivative(final_z)       # shape (batch, out_dim)
        A_double = final_layer.second_deriv(final_z)    # shape (batch, out_dim)
        A_third = final_layer.third_deriv(final_z)      # shape (batch, out_dim)

        # expand dims to allow elementwise operations on last two axes
        A_prime_exp   = A_prime[:, :, None]    # (batch, out_dim, 1)
        A_double_exp  = A_double[:, :, None]   # (batch, out_dim, 1)
        A_third_exp   = A_third[:, :, None]    # (batch, out_dim, 1)

        # overall_J_batch = A_prime ⊙ directional_deriv
        directional_deriv = self.J_batch / A_prime_exp

        # overall_H_batch = A_double ⊙ (directional_deriv)^2 + A_prime ⊙ propagated_H
        propagated_H = (self.H_batch - A_double_exp * directional_deriv**2) / A_prime_exp

        # dJ/da^L = (A_double/(A_prime^2)) ⊙ overall_J_batch.
        dJ_da_batch = (A_double_exp / (A_prime_exp**2)) * self.J_batch

        # dH/da^L = (A_third/(A_prime^3)) ⊙ (overall_J_batch^2) + (A_double/A_prime) ⊙ propagated_H
        dH_da_batch = (A_third_exp / (A_prime_exp**3)) * (self.J_batch**2) \
                    + (A_double_exp / A_prime_exp) * propagated_H

        self.dJ_da = dJ_da_batch   # shape: (batch_dim, out_dim, input_dim)
        self.dH_da = dH_da_batch   # shape: (batch_dim, out_dim, input_dim)

    def physics_loss(self, y) -> List[np.ndarray]:
        self.autograd()
        self.autograd_derivs()

        # print(self.H_batch.shape)
        # print(self.J_batch.shape)
        # print(y.shape)
        # print(self.dH_da.shape)
        # print(self.dJ_da.shape)
        # sys.exit("Stopping the program")

        y = np.squeeze(y)
        loss = (np.squeeze(self.H_batch) + y) ** 2
        grad_loss = 2 * (np.squeeze(self.H_batch) + y) * (np.squeeze(self.dH_da) + 1)

        # print(f"grad_loss shape: {grad_loss.shape}")
        # print(f"loss shape: {loss.shape}")

        # sys.exit("stopping")
        
        return loss, grad_loss

    def backward_helper(self, error: np.ndarray) -> None:
        """
        Helper function for the network backpropagation.
        Accumulates parameter gradients.
        """
        current_error = error

        # Back-propagate errors
        for layer in self.layers[-1:0:-1]:
            current_error = layer.backward(current_error)

        # calculate parameter gradients (averaged over batch dimension, IF there is batch_training)
        batch_dim = error.shape[0] if self.batch_training else 1

        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                if layer.previous.A.ndim == 1:
                    layer.previous.A[None, :]
                    layer.delta[None, :]

                dB = np.sum(layer.delta, axis=0) / (batch_dim)
                layer.B_grad.append(dB)

                dW = np.einsum('bi, bj -> ij', layer.delta, layer.previous.A) / (batch_dim)
                layer.W_grad.append(dW)
        #         print(f"layer with size: {layer.size} given dW shape: {dW.shape}")
                
        # sys.exit('stop')
    def backward(self, data: np.ndarray, label: np.ndarray, lr: float, loss_func: str, physics_loss: bool=True, store_grads: bool=False) -> np.ndarray:
        """
        Perform one backwards pass.

        Args:
            data shape: (batch_dim, input_dim)
            label shape: (batch_dim, output_dim)
            lr: learning rate float
            loss_func: specifies which loss function to apply
            physics_loss: introduce additive term into Loss to enforce physical constraints
            store_grads: calculates jacobians for layers when performing backwards pass
        """
        # Jacobians computed in forward pass
        forward_out = self.forward(data, store_grads)

        assert forward_out.shape == label.shape

        if loss_func not in self.loss_functions:
            raise ValueError(f'Unsupported loss function: {loss_func}')
        
        loss = self.loss_functions[loss_func](forward_out, label)

        # print(f"MSE loss shape: {loss.shape}")

        if physics_loss:
            Lp, grad_loss = self.physics_loss(y=forward_out)
            grad_loss = grad_loss.reshape(-1, 1)

            # # perform "3-way backpropagation" due to finite-difference derivative
            # # then average the accumulated parameter gradients
            # Lp, grad_y, grad_y_plus, grad_y_minus = self.physics_loss(data)
            loss += self.physics_loss_weight * Lp

            # print(f"grad_loss shape: {grad_loss.shape}")
            # print(f"label shape: {label.shape}")
            # print(f"weight shape: {"it's a goddamn float" if isinstance(self.physics_loss_weight, float) else self.physics_loss_weight.shape}")
            # print(f"forward_out shape: {forward_out.shape}")
            mse_error = forward_out - label

            # print(f"mse_error shape: {mse_error.shape}")
            error = mse_error + (self.physics_loss_weight * grad_loss)

            # print(f"error shape: {error.shape}")

            # # self.backward_helper(error_y)
            # # errors for y+- evaluated using only the physics loss (we don't have a label)
            # error_y_plus = self.physics_loss_weight * grad_y_plus
            # # self.backward_helper(error_y_plus)

            # error_y_minus = self.physics_loss_weight * grad_y_minus
            # # self.backward_helper(error_y_minus)

            # concatenated_error = np.concatenate([error_y, error_y_plus, error_y_minus], axis=0)
            self.backward_helper(error)

        else:
            error = forward_out - label
            self.backward_helper(error)

        # assert loss.shape == (self.batch_dim,)

        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                self.optim.update(lr=lr, layer=layer)
                layer.W_grad = []
                layer.B_grad = []

        # if optimizer is Adam, increase global timestep
        if isinstance(self.optim, Adam):
            self.optim.t += 1

        return loss

    @staticmethod
    def adaptive_lr_schedule(current_lr: float, losses: List[float], patience: int=5, threshold: float=1e-3) -> float:
        """
        Adaptive learning rate schedule with a ReduceLROnPlateau mechanism.
        If the loss has not improved for 'patience' epochs, reduce the learning rate.
        """
        # if len(losses) < patience:
        #     return current_lr
        
        # recent_losses = losses[-patience:]
        # min_recent_loss = min(recent_losses)
        # prev_loss = losses[-2] if len(losses) > 1 else None
        # current_loss = losses[-1]

        # # if loss hasn't improved by more than 'threshold' over 'patience' epochs, decrease lr aggressively
        # if all(abs(l - min_recent_loss) < threshold for l in recent_losses):
        #     return current_lr * 0.9
        
        return current_lr * 0.999

    def learn(self, learn_data: List[Tuple[np.ndarray, np.ndarray]], lr: float, epochs: int, loss_func: str, physics_loss: bool=True, plot: bool=False, store_grads: bool=False) -> None:
        """
        Args:
            learn_data: [(input_tensor, label_tensor), (input_tensor, label_tensor), ... ]
                each input_tensor has shape: (batch_dim, input_layer.size)
            lr: initial learning rate float
            epochs: number of training epochs int
            loss_func: specifies which loss function to apply
            physics_loss: turn on the physics loss function
            plot: choose to plot loss against epochs
            store_grads: calculates jacobians for layers when performing backwards pass
        """
        self.current_lr = lr
        self.prev_loss = None

        losses = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0
            for input, label in learn_data:
                # use current adaptive learning rate in backward pass
                loss = self.backward(input, label, self.current_lr, loss_func, physics_loss, store_grads)
                epoch_loss += np.mean(loss)

            current_loss = epoch_loss / len(learn_data)
            losses.append(current_loss)

            # update learning rate adaptively at the end of each epoch
            new_lr = self.adaptive_lr_schedule(self.current_lr, losses)
            self.prev_loss = current_loss
            self.current_lr = new_lr
        
        if plot:
            plt.figure()
            plt.plot(range(epochs), losses, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.show()

    def save_parameters(self, file_path: str) -> None:
        """
        Saves model parameters (weights & biases) to a file.
        """
        if not file_path.endswith('.npz'):
            raise ValueError('Invalid file extension for saving.')

        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                params[f'layer_{i}_W'] = layer.W
                params[f'layer_{i}_B'] = layer.B

        np.savez(file_path, **params)

    def load_parameters(self, file_path: str) -> None:
        """
        Loads learned hyperparameters from specified file.
        """
        if not file_path.endswith('.npz'):
            raise ValueError('Invalid file extension for loading.')

        params = np.load(file_path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                layer.W = params[f'layer_{i}_W']
                layer.B = params[f'layer_{i}_B']
