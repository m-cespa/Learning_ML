from typing import List, Callable
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.array(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes derivative of ReLU element-wise.
        d(ReLU(z))/dz = 1 if z > 0, otherwise 0.
        """
        return (x > 0).astype(np.float32)
    
class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.array(x)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes derivative of Sigmoid.
        d(Sigmoid(z))/dz = Sigmoid(z)*(1-Sigmoid(z))
        """
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)
    
class BaseLayer:
    def __init__(self):
        self.A = None
        self.previous = None
        self.next = None

    def forward(self):
        raise NotImplementedError('Method must be implemented for subclasses.')
    
    def backward(self):
        raise NotImplementedError('Method must be implemented for subclasses.')
    
class Layer(BaseLayer):
    def __init__(self, size: int, previous: 'Layer'=None):
        """
        Creates Layer of {size} neurons.

        Args:
            size: number of neurons in layer.
            previous (optional): pointer to previous layer.
            next (optional): pointer to next layer.
        """
        super().__init__()
        self.size = size
        self.W = None
        self.B = None
        self.delta = None
        self.activation_deriv = None
        if previous:
            self.previous = previous
            self.W, self.B = self.construct_weights_and_biases()

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
        For the l^th layer we have weight matrix W^l and bias vector b^l.
        These operate on the attentions of previous layer a^(l-1).
            a^l = a^(l-1) * W^(l).T + b^(l)

            dim(W^l) = len(a^l) x len(a^l-1)
        """
        if self.previous is None:
            raise ValueError("Previous layer not assigned.")
        
        # traces back to last Layer class to fetch size attribute
        current_layer = self.previous
        while not isinstance(current_layer, Layer):
            if current_layer.previous is None:
                raise ValueError("Previous layer chain is broken.")
            current_layer = current_layer.previous
        
        prev_size = current_layer.size
        
        weights_matrix = np.random.uniform(-1, 1, (self.size, prev_size))
        biases_vector = np.zeros(self.size)
        
        return weights_matrix, biases_vector
    
    def forward(self) -> None:
        """
        Take activation vector A from previous layer.
        Perform A_current = A * W.T + B
        """
        if self.previous is None:
            raise ValueError("Previous layer not assigned.")
        
        # edge case for 1D matrices
        if len(self.W.shape) == 1:
            z = np.matmul(self.previous.A, self.W.reshape(-1,1)) + self.B
            self.A = z
            return
        
        assert self.previous.A.shape[0] == self.W.shape[1]

        z = np.matmul(self.previous.A, self.W.T) + self.B
        self.A = z

    def backward(self, error) -> np.ndarray:
        """
        Layer passes backwards D * W.T updated error vector.
        W is the weights of the first downstream Layer (skipping ActivationLayers).
        Previous layer should be activation layer which will rescale the saved error vector.
        """
        if self.activation_deriv is None:
            self.activation_deriv = 1

        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')

        current = self.next
        while not isinstance(current, Layer):
            if current.next is None:
                self.delta = error
                return self.delta

            current = current.next

        # edge case for 1D matrices
        if len(current.W.shape) == 1:
            self.delta = np.matmul(error, current.W[np.newaxis]) * self.activation_deriv
            return self.delta

        self.delta = np.matmul(error, current.W) * self.activation_deriv
        
        return self.delta

class ActivationLayer(BaseLayer):
    def __init__(self, activation_function: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.activation_function = activation_function

    def forward(self):
        if self.previous:
            self.A = self.activation_function(self.previous.A)

    def backward(self, error: np.ndarray) -> List[np.ndarray]:
        """
        ActivationLayer passes backwards derivative evaluated on its activations
        multiplied by fed back error vector.

        If the ActivationLayer's next pointer is a Layer (true except output layer)
        the error vector of the next layer is updated accordingly.
        """
        deriv = self.activation_function.derivative(self.A)

        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')
        
        self.previous.activation_deriv = deriv

        return error

class Network:
    def __init__(self, layers: List[BaseLayer]):
        """
        Initialise network with list of BaseLayers.
        First layer automatically interpreted as Input.
        Layers linked sequentially.
        """
        self.layers = layers
        for i in range(1, len(self.layers)):
            self.layers[i].previous = self.layers[i-1]
            self.layers[i-1].next = self.layers[i]
            if isinstance(self.layers[i], Layer):
                self.layers[i].assign_previous_layer(self.layers[i-1])

        self.loss_functions = {
            'mse': self.MSELoss,
            'ce': self.CELoss
        }

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Set InputLayer activations to arg: input_data.
        Perform forward pass.
        """
        self.layers[0].A = input_data
        for layer in self.layers[1:]:
            layer.forward()
        return self.layers[-1].A
    
    def MSELoss(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * np.sum((x - y) ** 2)
    
    def CELoss(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(x))
    
    def backward(self, data: np.ndarray, labels: np.ndarray, lr: float, loss_func: str) -> float:
        forward_out = self.forward(data)

        if loss_func not in self.loss_functions:
            raise ValueError(f'Unsupported loss function: {loss_func}')
        
        loss = self.loss_functions[loss_func](forward_out, labels)

        error = forward_out - labels
        current_error = error

        # back-propagate errors
        for layer in self.layers[-1:0:-1]:
            current_error = layer.backward(current_error)

        # perform gradient descent
        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                layer.B -= lr * layer.delta
                layer.W -= lr * np.outer(layer.delta, layer.previous.A)

        return loss
    
    def learn(self, learn_data: np.ndarray, lr: float, epochs: int, loss_func: str) -> None:
        """
        Learning data is [[inputs], [labels]] to perform unsupervised learning.
        Backpropagation performed with specified learning rate {lr}.
        Process iterated for {epochs}
        """
        losses = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0
            np.random.shuffle(learn_data)

            for input, label in learn_data:
                input = np.array(input)
                label = np.array(label)
                loss = self.backward(input, label, lr, loss_func)
                epoch_loss += loss

            losses.append(epoch_loss / len(learn_data))

        plt.figure()
        plt.plot(range(epochs), losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    def save_parameters(self, file_path: str) -> None:
        """
        Saves hyperparameters to a file.
        """
        if not file_path.endswith('.npz'):
            raise ValueError('Invalid file extension for saving.')
        
        params = {
            f'layer_{i}_W': layer.W for i, layer in enumerate(self.layers) if isinstance(layer, Layer)
        }
        params.update({
            f'layer_{i}_B': layer.B for i, layer in enumerate(self.layers) if isinstance(layer, Layer)
        })
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
