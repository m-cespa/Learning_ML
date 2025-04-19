from typing import List, Callable, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList
from activation_functions import Tanh, Sin, Sigmoid, ELU, ReLU, Linear, Quadratic
from optimizers import Optimizer, GD, SGD, Adam, Lion

import os
os.environ["OMP_DISPLAY_ENV"] = "FALSE"

@njit(fastmath=True, parallel=False, cache=False)
def autograd_core(
    batch_size: int,
    input_dim: int,
    network_output_dim: int,
    g_list,             # list of (batch, out_dim, in_dim)
    W_list,             # list of (out_dim, in_dim)
    A1_list,            # A_prime: list of (batch, out_dim)
    A2_list,            # A_double: list of (batch, out_dim)
    A3_list,            # A_triple: list of (batch, out_dim)
    tol: float = 1e-3
):
    # Initialise
    overall_J = np.zeros((batch_size, input_dim, input_dim))
    overall_H = np.zeros((batch_size, input_dim, input_dim, input_dim))
    overall_T = np.zeros((batch_size, input_dim, input_dim, input_dim))

    # initialise overall_J to identity across batches
    for b in range(batch_size):
        for i in range(input_dim):
            overall_J[b, i, i] = 1.0

    num_layers = len(g_list)

    for l in range(num_layers):
        g_l = g_list[l]           # (batch, out_dim, in_dim)
        W = W_list[l]             # (out_dim, in_dim)
        A_prime = A1_list[l]      # (batch, out_dim)
        A_double = A2_list[l]     # (batch, out_dim)
        A_triple = A3_list[l]     # (batch, out_dim)
        out_dim, in_dim = W.shape

        # ---- Compute new_J = g_l @ overall_J ----
        new_J = np.zeros((batch_size, out_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    acc = 0.0
                    for q in range(in_dim):
                        acc += g_l[b, j, q] * overall_J[b, q, k]
                    new_J[b, j, k] = acc

        # ---- Compute w_J_prev = W @ overall_J ----
        w_J_prev = np.zeros((batch_size, out_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    acc = 0.0
                    for q in range(in_dim):
                        acc += W[j, q] * overall_J[b, q, k]
                    w_J_prev[b, j, k] = acc

        # ---- H_term1 = A'' * (w_J_prev)^2 ----
        H_term1 = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        H_term1[b, j, k, m] = A_double[b, j] * w_J_prev[b, j, m] * w_J_prev[b, j, k]

        # ---- w_H_prev = W @ overall_H ----
        w_H_prev = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        acc = 0.0
                        for q in range(in_dim):
                            acc += W[j, q] * overall_H[b, q, k, m]
                        w_H_prev[b, j, k, m] = acc

        # ---- H_term2 = A' * w_H_prev ----
        H_term2 = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        H_term2[b, j, k, m] = A_prime[b, j] * w_H_prev[b, j, k, m]

        # ---- new_H = H_term1 + H_term2 ----
        new_H = H_term1 + H_term2

        # ---- diag(w_H_prev) ----
        diag_w_H_prev = np.zeros((batch_size, out_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    diag_w_H_prev[b, j, k] = w_H_prev[b, j, k, k]

        # Now calculating T_jkm
        # ---- bracket1 = A''' * w_J^2 + A'' * diag(w_H_prev) ----
        bracket1 = np.zeros((batch_size, out_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    bracket1[b, j, k] = (
                        A_triple[b, j] * w_J_prev[b, j, k] * w_J_prev[b, j, k]
                        + A_double[b, j] * diag_w_H_prev[b, j, k]
                    )

        # ---- T_term1 = w_J_prev * bracket1 ----
        T_term1 = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        T_term1[b, j, k, m] = w_J_prev[b, j, m] * bracket1[b, j, k]

        # ---- T_term2 = 2 * A'' * w_H_prev ----
        T_term2 = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        T_term2[b, j, k, m] = 2.0 * A_double[b, j] * w_H_prev[b, j, k, m]

        # ---- T_term3 = A' * (W @ overall_T) ----
        w_T_prev = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        acc = 0.0
                        for q in range(in_dim):
                            acc += W[j, q] * overall_T[b, q, k, m]
                        w_T_prev[b, j, k, m] = acc

        T_term3 = np.zeros((batch_size, out_dim, input_dim, input_dim))
        for b in range(batch_size):
            for j in range(out_dim):
                for k in range(input_dim):
                    for m in range(input_dim):
                        T_term3[b, j, k, m] = A_prime[b, j] * w_T_prev[b, j, k, m]

        # Update recursive tensors
        overall_J = new_J
        overall_H = new_H
        overall_T = T_term1 + T_term2 + T_term3

    # ---- post-loop ----
    # clamp J to avoid division by zero
    inv_J = np.zeros_like(overall_J)
    for b in range(batch_size):
        for j in range(overall_J.shape[1]):
            for m in range(input_dim):
                Jval = overall_J[b, j, m]
                absJ = abs(Jval)
                safe = max(absJ, tol)
                inv_J[b, j, m] = np.sign(Jval) / safe

    # ---- dJ/da = H @ inv_J ----
    # shape: (batch, out_dim, in_dim, out_dim)
    dJ_da = np.zeros((batch_size, network_output_dim, input_dim, network_output_dim))

    for b in range(batch_size):
        for j in range(network_output_dim):
            for k in range(input_dim):
                for m in range(network_output_dim):
                    acc = 0.0
                    # sum over the "q" index (which runs over input_dim)
                    for q in range(input_dim):
                        acc += overall_H[b, j, k, q] * inv_J[b, m, q]
                    dJ_da[b, j, k, m] = acc

    # ---- diag_H ----
    diag_H = np.zeros((batch_size, overall_H.shape[1], input_dim))
    for b in range(batch_size):
        for j in range(overall_H.shape[1]):
            for k in range(input_dim):
                diag_H[b, j, k] = overall_H[b, j, k, k]

    # ---- dH/da = T @ inv_J ----
    dH_da = np.zeros_like(dJ_da)
    # shape: (batch, out_dim, in_dim, out_dim)
    for b in range(batch_size):
        for j in range(network_output_dim):
            for k in range(input_dim):
                for m in range(network_output_dim):
                    acc = 0.0
                    for q in range(input_dim):
                        acc += overall_T[b, j, k, q] * inv_J[b, m, q]
                    dH_da[b, j, k, m] = acc

    return overall_J, diag_H, dJ_da, dH_da, overall_H, overall_T
   
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
            size: number of neurons in layer
            previous (optional): pointer to previous layer
            weight_init: select the weight initialisation method
        """
        super().__init__()
        self.size = size

        # will accumulate gradients to then alter parameters by
        self.W_grad = 0
        self.B_grad = 0

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
        
        # weights shape: (self.size, previous.size)
        # bias shape: (self.size)
        weights_tensor = self.weight_initialisations[self.weight_init](out_dim=self.size, in_dim=prev_size)
        biases_tensor = np.zeros((self.size,))

        return weights_tensor, biases_tensor
    
    def forward(self, input_data: np.ndarray=None, store_grads: bool=True) -> None:
        """
        Take activation vector a^{l-1}j from previous layer (should be an ActivationLayer)
        Perform z^{l}_i = w^{l}_ij a^{l-1}_j + b_i
        """
        # if data isn't passed to the forward method, will fetch from previous.A
        # assumes previous layer is ActivationLayer (always the case)
        if input_data is not None:
            A_prev = input_data
        else:
            A_prev = self.previous.A
                
        # in case there is no batch dim, introduce a singleton dimension
        if A_prev.ndim == 1:
            A_prev = A_prev[None, :]

        z = A_prev @ self.W.T + self.B
        # effectively does np.einsum('ij, bj -> bi', self.W, A_prev) + self.B

        # z shape: (self.batch_dim, self.size)
        self.z = z
        self.activation_deriv = self.next.derivative(self.z)

        # calculate layer-to-layer Jacobian for autograd, leaving batch dimension
        if store_grads:
            self.activation_second_deriv = self.next.second_deriv(self.z)
            self.activation_third_deriv = self.next.third_deriv(self.z)
            self.g = self.activation_deriv[:, :, None] * self.W[None, :, :]

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

        # delta shape: (batch_dim, self.size)
        self.delta = (error @ current.W) * self.activation_deriv
        # the above effectively np.einsum('ij, bi -> bj', current.W, error) * self.activation_deriv

        return self.delta
    
class ActivationLayer(BaseLayer):
    def __init__(self, activation_function: Callable[[np.ndarray], np.ndarray]):
        """
        Wrapper class for Activations.
        Includes default methods for derivatives as many are zero.
        """
        super().__init__()
        self.activation_function = activation_function

    def forward(self, x=None, store_grads=True):
        if self.previous:
            self.A = self.activation_function(self.previous.z)
        else:
            self.A = self.activation_function(x)

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
        
        return error
        
class Network:
    def __init__(self, layers: List[BaseLayer], optim: Optimizer=Adam(), lambda1=0.0001, lambda2=0.001, batch_training=True):
        """
        Initialise network with list of BaseLayers.
        First layer automatically interpreted as Input.
        Layers linked sequentially.

        Args:
            layers: list of network layers
            lambda1: scalar weighting for Differential Equation physics loss
            lambda2: scalar weighting for Boundary Conditions physics loss
            batch_training: network supports completely dynamic batch training (default set to True)
        """
        self.layers = layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.optim = optim

        # link the layers
        self.setup()

        self.input_dim = layers[0].size
        self.output_dim = layers[-1].size

        self.loss_functions = {
            'mse': self.MSELoss,
            'ce': self.CELoss
        }
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

            # link layers together
            layer.previous = self.layers[i-1]
            self.layers[i-1].next = layer

            # initialise Layer object weights
            if isinstance(layer, Layer):
                layer.assign_previous_layer(self.layers[i-1])

            # assign size of the previous Layer object to all ActivationLayers
            if isinstance(layer, ActivationLayer):
                layer.size = layer.previous.size

    def forward(self, input_data: np.ndarray, store_grads=True) -> np.ndarray:
        """
        Args:
            input_data shape: (batch_dim, input_layer.size)
            store_grads: calculates layer Jacobians in forward pass
        """
        self.layers[0].z = input_data

        for layer in self.layers[1:]:
            layer.forward(store_grads=store_grads)
        return self.layers[-1].A
    
    def MSELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((x - y) ** 2, axis=0)

    def CELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(x), axis=-1)
    
    def numerical_jacobian_hessian(self, X: np.ndarray, h: float = 1e-5):
        batch_size, input_dim = X.shape
        eps = 1e-12
        
        # Original outputs
        u0 = self.forward(X, store_grads=False).flatten()  # shape: (batch_size,)
        
        # Initialize outputs
        jacobian = np.zeros((batch_size, input_dim))
        hessian_diag = np.zeros((batch_size, input_dim))
        
        for j in range(input_dim):
            # Create perturbations
            offsets = np.zeros_like(X)
            offsets[:, j] = h
            
            X_plus2 = X + 2*offsets
            X_plus = X + offsets
            X_minus = X - offsets
            X_minus2 = X - 2*offsets
            
            # Forward passes (batched)
            perturbed_inputs = np.concatenate([X_plus2, X_plus, X_minus, X_minus2])
            perturbed_outputs = self.forward(perturbed_inputs, store_grads=False)
            u_plus2, u_plus, u_minus, u_minus2 = np.split(perturbed_outputs, 4)
            
            # Flatten all outputs
            u0_flat = u0
            u_plus2 = u_plus2.flatten()
            u_plus = u_plus.flatten()
            u_minus = u_minus.flatten()
            u_minus2 = u_minus2.flatten()
            
            # Fourth-order finite differences
            J = (-u_plus2 + 8*u_plus - 8*u_minus + u_minus2) / (12 * h)
            H = (-u_plus2 + 16*u_plus - 30*u0_flat + 16*u_minus - u_minus2) / (12 * h**2)
            
            # Store Jacobian and Hessian
            jacobian[:, j] = J
            hessian_diag[:, j] = H
            
        return jacobian, hessian_diag
        
    def autograd_numba(self):
        batch_size = self.layers[0].z.shape[0]

        # gather into typed lists for Numba parsing
        g_list  = TypedList()
        W_list  = TypedList()
        A1_list = TypedList()
        A2_list = TypedList()
        A3_list = TypedList()
        for layer in self.layers:
            if hasattr(layer, 'g'):
                g_list.append(np.asarray(layer.g))
                W_list.append(np.asarray(layer.W))
                A1_list.append(np.asarray(layer.activation_deriv))
                A2_list.append(np.asarray(layer.activation_second_deriv))
                A3_list.append(np.asarray(layer.activation_third_deriv))

        # call external Numba core
        J, diag_H, dJ_da, dH_da, overall_H, T = autograd_core(
            batch_size, self.input_dim, self.output_dim,
            g_list, W_list,
            A1_list, A2_list, A3_list,
        )

        return J, diag_H, dJ_da, dH_da, overall_H, T
    
    def autograd(self) -> List[np.ndarray]:
        """
        Performs the iterative calculation for:
            J_{jk}^L: full Jacobian
            H_{jkm}^L: full Hessian
            dJ_{jk}^L/da_m^L: derivative of full Jacobian wrt output activations
            dH_{jk}^L/da_m^L: derivative of diagonal Hessian wrt output activations
                (only the diagonal Hessian is used in the Physics loss functions)
        """
        batch_size = self.layers[0].z.shape[0]

        delta_D = np.eye(self.input_dim)

        # build array recursively, so initialise them to their 0th layer shapes
        overall_J = np.tile(delta_D[None, :, :], (batch_size, 1, 1))
        overall_H = np.zeros((batch_size, self.input_dim, self.input_dim, self.input_dim))
        overall_T = np.zeros((batch_size, self.input_dim, self.input_dim, self.input_dim))
        # NOTE: the dJ/da array does not require recursive construction, it is computed after

        # NOTE: 0th layer won't have a 'g' (layer-to-layer Jacobian)
        for layer in self.layers:
            if isinstance(layer, Layer) and hasattr(layer, 'g'):
                g_l = layer.g   # shape (batch, out_dim, in_dim)

                # recover required layer arrays
                W = layer.W                                 # shape (out_dim, in_dim)
                A_prime = layer.activation_deriv            # shape (batch, out_dim)      
                A_double = layer.activation_second_deriv    # shape (batch, out_dim) 
                A_triple = layer.activation_third_deriv     # shape (batch, out_dim)

                # NOTE: calcualte new J
                # J^{l} = g_{l} * J^{l-1}
                new_J = np.einsum('boi, bij -> boj', g_l, overall_J)
                # new_J = g_l @ overall_J

                # 'overall_J' is currently J^{l-1} with shape (batch, previous_layer_out, D)
                w_J_prev = np.einsum('jq, bqm -> bjm', W, overall_J)
                H_term1 = np.einsum('bj, bjm, bjk -> bjkm', A_double, w_J_prev, w_J_prev)
                # 'overall_H' is currently H^{l-1} with shape (batch, previous_layer_out, D, D)
                # will reuse this w_H_prev
                w_H_prev = np.einsum('jq, bqkm -> bjkm', W, overall_H)
                H_term2 = A_prime[:, :, None, None] * w_H_prev
                new_H = H_term1 + H_term2

                # (w_jq J_qm^l-1)[A''' (w_jq J_qk^l-1)^2 + A'' w_jq H_qk^l-1]
                w_H_prev_diag = np.diagonal(w_H_prev, axis1=2, axis2=3)
                bracket1 = A_triple[:, :, None] * w_J_prev**2 + A_double[:, :, None] * w_H_prev_diag
                T_term1 = np.einsum('bjm, bjk -> bjkm', w_J_prev, bracket1)

                # 2A'' w_jq H_qkm^l-1
                T_term2 = 2 * A_double[:, :, None, None] * w_H_prev

                # A' w_jq T_qkm^l-1
                T_term3 = A_prime[:, :, None, None] * np.einsum('jq, bqkm -> bjkm', W, overall_T)

                overall_T = T_term1 + T_term2 + T_term3
                overall_H = new_H
                overall_J = new_J

        # compute dJ/da:
        # use a tolerance thresholder to avoid very large 1/J
        tol = 1e-6
        J_safe = np.sign(overall_J) * np.maximum(np.abs(overall_J), tol)
        inv_J  = 1.0 / J_safe
        dJ_da = np.einsum('bjkq, bmq -> bjkm', overall_H, inv_J)

        # we only really care about the diagonal Hessian
        diag_H = np.diagonal(overall_H, axis1=2, axis2=3)

        dH_da = np.einsum('bjkq, bmq -> bjkm', overall_T, inv_J)

        return overall_J, diag_H, dJ_da, dH_da, overall_H, overall_T

    def compute_physics_loss(self, collocation_data, boundary_data=None) -> List[np.ndarray]:
        """
        Calculates the Physics Loss as k*d^2u/dx^2 - du/dt (= 0 in theory)
        Uses the collocation_data of size N and runs a singular forward pass
        with batch size N.

        Optionally evaluates BC and IC loss by imposing:
            u(L, t) = 0
            u(0, t) = 0
            u(x, 0) = 0
        """
        collocation_data = collocation_data
        self.forward(input_data=collocation_data, store_grads=True)

        J, H, dJ_da, dH_da, _, _ = self.autograd_numba()

        # x is 0th index, t is 1st
        du_dt = np.squeeze(J[:, :, 1])
        d2u_dx2 = np.squeeze(H[:, :, 0])

        # expect dJ/da, dH/da to be shape (N_colloc, output_dim, input_dim, output_dim)
        dJ_t_da = np.squeeze(dJ_da[:, :, 1, :])
        dH_x_da = np.squeeze(dH_da[:, :, 0, :])

        residual = (1. * d2u_dx2 - du_dt).squeeze()

        # handle BC/IC terms if provided
        if boundary_data is not None:
            ics_bcs = self.forward(input_data=boundary_data, store_grads=False)
            bc_loss = 0.5 * np.mean(ics_bcs**2, axis=0)
            bc_grad = np.mean(ics_bcs, axis=0)
        else:
            bc_loss = 0.
            bc_grad = 0.

        residual_loss = 0.5 * np.mean(residual**2, axis=0)
        residual_grad = np.mean((1. * dH_x_da - dJ_t_da) * residual, axis=0)
        
        # gradient clipping for numerical stability
        max_grad_norm = 1.
        grad_norm = np.linalg.norm(residual_grad)
        if grad_norm > max_grad_norm:
            residual_grad *= max_grad_norm / grad_norm

        return residual_loss, bc_loss, residual_grad, bc_grad

    def backward_helper(self, error: np.ndarray) -> None:
        """
        Helper function for the network backpropagation.
        Accumulates parameter gradients.
        """
        current_error = error

        # back-propagate errors
        for layer in self.layers[-1:0:-1]:
            current_error = layer.backward(current_error)

        # calculate parameter gradients (averaged over batch dimension, IF there is batch_training)
        # important to recalculate this size in case a batch is differently shaped from the previous
        batch_dim = error.shape[0] if self.batch_training else 1

        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                # in case there isn't an explicit batch_dim introduce a singleton dim
                if layer.previous.A.ndim == 1:
                    layer.previous.A[None, :]
                    layer.delta[None, :]

                # average the parameter gradients over the batch dimension
                # Δ(b^l_j) = δ_j
                # Δ(w^l_ij) = δ_i a^{l-1}_j
                layer.B_grad = np.mean(layer.delta, axis=0)
                layer.W_grad = layer.delta.T @ layer.previous.A / batch_dim
                # effectively does np.einsum('bi, bj -> ij', layer.delta, layer.previous.A)

    def backward(self, data: np.ndarray, label: np.ndarray, lr: float, loss_func: str, 
                 collocation_data=None, boundary_data=None, store_grads: bool=True) -> np.ndarray:
        """
        Args:
            data shape: (batch_dim, input_dim)
            label shape: (batch_dim, output_dim)
            collocation_data shape: (N_collocation, input_dim) - used for Physics loss if supplied
            boundary_data shape: (N_boundary, input_dim) - used for Physics loss if supplied
            lr: learning rate float
            loss_func: specifies which loss function to apply
            store_grads: calculates jacobians for layers when performing backwards pass
        """
        # if collocation data available, get physics loss, otherwise use zeros
        if collocation_data is not None:
            residual_loss, bc_loss, residual_grad, bc_grad = self.compute_physics_loss(collocation_data, boundary_data)
        else:
            residual_loss, bc_loss, residual_grad, bc_grad = 0., 0., 0., 0.

        # NOTE: forward pass is performed AFTER physics loss as the physics loss
        # leaves network activations changed (no longer correspond to the training data)
        forward_out = self.forward(data, store_grads)
        N_batch = forward_out.shape[0] if self.batch_training else 1

        assert forward_out.shape == label.shape

        if loss_func not in self.loss_functions:
            raise ValueError(f'Unsupported loss function: {loss_func}')
        
        loss = self.loss_functions[loss_func](forward_out, label)
        mse_error = (forward_out - label) / N_batch

        # add physics loss term to total loss and error to propagate
        # apply lambda weighting to physics losos terms
        loss += self.lambda1 * residual_loss + self.lambda2 * bc_loss
        error = mse_error + self.lambda1 * residual_grad + self.lambda2 * bc_grad

        # perform backpropagation
        self.backward_helper(error)

        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                # perform the optimiser update on each layer
                self.optim.update(lr=lr, layer=layer)
                # reset parameter gradients for next epoch
                layer.W_grad = 0
                layer.B_grad = 0

        # if optimizer is Adam, increase global timestep
        if isinstance(self.optim, Adam):
            self.optim.t += 1

        # return total loss & unweighted residual
        return loss, residual_loss
    
    def evaluate_relative_error(self, num_points: int = 100) -> float:
        T = 1 / ((np.pi / 10.) ** 2)

        x_eval = np.linspace(0, 10., num_points)
        t_eval = np.linspace(0, T, num_points)
        X, T_ = np.meshgrid(x_eval, t_eval)
        eval_points = np.stack([X.flatten(), T_.flatten()], axis=1)

        # ground truth
        u_true = np.sin(np.pi * eval_points[:, 0] / 10.) * np.exp((np.pi / 10.)**2 * -eval_points[:, 1])

        # network solution
        u_pred = self.forward(eval_points).squeeze()

        # relative L2 error
        relative_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        return relative_error

    @staticmethod
    def adaptive_lr_schedule(current_lr: float, decay: float=0.999) -> float:
        return current_lr * decay

    def learn(self, learn_data: List[Tuple[np.ndarray, np.ndarray]], lr: float, epochs: int, loss_func: str,
            collocation_data=None, boundary_data=None, store_grads: bool=False, decay: float=0.999,
            test_collocation_data=None, track_diagnostics: bool=False):
        """
        Trains model on provided dataset.

        Args:
            learn_data: List of (input_tensor, label_tensor) tuples.
            lr: Initial learning rate.
            epochs: Number of epochs to train.
            loss_func: Loss function identifier string.
            collocation_data: (N, input_dim) for physics loss (optional).
            boundary_data: (N_boundary, input_dim) for physics loss (optional).
            store_grads: Whether to store gradients during backpropagation.
            decay: Learning rate decay factor applied after each epoch.
            test_collocation_data: Used to compute residuals on unseen collocation data.
            track_diagnostics: If False, only returns total loss history.

        Returns:
            Dict with training history. Keys vary based on `track_diagnostics`.
        """
        N = len(learn_data)
        total_losses = []

        # optional tracking
        residual_losses = [] if track_diagnostics else None
        validation_residuals = [] if track_diagnostics else None
        relative_errors = [] if track_diagnostics else None

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            random.shuffle(learn_data)
            epoch_total_loss = 0
            epoch_residual_loss = 0

            for input, label in learn_data:
                loss, residual_loss = self.backward(input, label, lr, loss_func, collocation_data, boundary_data, store_grads)
                epoch_total_loss += loss
                epoch_residual_loss += residual_loss

            current_total_loss = epoch_total_loss / N
            total_losses.append(current_total_loss)

            if track_diagnostics:
                residual_losses.append(epoch_residual_loss / N)

                if test_collocation_data is not None:
                    val_residual, _, _, _ = self.compute_physics_loss(test_collocation_data)
                    validation_residuals.append(val_residual * 2)
                else:
                    validation_residuals.append(None)

                relative_domain_error = self.evaluate_relative_error()
                relative_errors.append(relative_domain_error)

            lr = self.adaptive_lr_schedule(lr, decay)

        # return minimal or full diagnostics
        history = {"total_losses": total_losses}
        if track_diagnostics:
            history.update({
                "residual_losses": residual_losses,
                "validation_residuals": validation_residuals,
                "relative_errors": relative_errors
            })
        return history

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

        params = np.load(file_path, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                layer.W = params[f'layer_{i}_W']
                layer.B = params[f'layer_{i}_B']    
