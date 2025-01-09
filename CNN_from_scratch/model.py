from typing import List, Callable, Tuple
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

class Conv2D(BaseLayer):
    def __init__(self, batch_dim: int, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0):
        """
        Creates a Convolutional Kernel.

        Args:
            in_channels: number of input (colour) channels
            out_channels: number of channels output is compressed down to
            kernel_size: size of convolutional kernel square
            stride: step size of kernel over image matrix
            padding: adds extra border of 0s to original image
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = np.random.uniform(-1, 1, (batch_dim, out_channels, in_channels, kernel_size, kernel_size))
        self.B = np.zeros(batch_dim, out_channels)

class Layer(BaseLayer):
    def __init__(self, size: int, batch_dim: int=1, channel_dim: int=1, previous: 'Layer'=None):
        """
        Creates Layer of {size} neurons.

        Args:
            size: number of neurons in layer.
            batch_dim (optional): dimension of batches for training, set to 1 default.
            channel_dim (optional): dimension of colour channels (3 for RGB), set to 1 default.
            previous (optional): pointer to previous layer.
            next (optional): pointer to next layer.
        """
        super().__init__()
        self.size = size
        self.batch_dim = batch_dim
        self.channel_dim = channel_dim

        self.A = np.zeros((batch_dim, channel_dim, size))
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
        
        weights_tensor = np.random.uniform(-1, 1, (self.batch_dim, self.channel_dim, self.size, prev_size))
        biases_tensor = np.zeros((self.batch_dim, self.channel_dim, self.size))
        
        return weights_tensor, biases_tensor
    
    def forward(self) -> None:
        """
        Take activation vector A from previous layer.
        Perform A_current = A * W.T + B
        """
        if self.previous is None:
            raise ValueError("Previous layer not assigned.")
                
        # assert self.previous.A.shape == (self.batch_dim, self.channel_dim, self.previous.size)
        # assert self.W.shape == (self.batch_dim, self.channel_dim, self.size, self.previous.size)
        # assert self.B.shape == (self.batch_dim, self.channel_dim, self.size)

        # contracting W tensor with activation tensor over previous.size dimension
        z = np.einsum('bcj, bcij -> bci', self.previous.A, self.W) + self.B
        assert z.shape == (self.batch_dim, self.channel_dim, self.size)
        self.A = z

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Layer passes backwards D * W.T updated error vector.
        W is the weights of the first (current) downstream Layer (skipping ActivationLayers).
        Previous layer should be activation layer which will rescale the saved error vector.

        Args:
            error shape: (batch_dim, channel_dim, current.size)
        """
        if self.activation_deriv is None:
            self.activation_deriv = 1

        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')

        # next layer should be Activation
        current = self.next

        # move forward until you find first Layer
        while not isinstance(current, Layer):
            # if the next layer is None, we are at the output layer
            # set the error tensor of self to error tensor of output layer
            if current.next is None:
                self.delta = error
                return self.delta

            current = current.next
        # current should be the first downstream Layer

        assert error.shape == (self.batch_dim, self.channel_dim, current.size)

        # contracting W tensor with error tensor current.size dimendion
        # we expect each error vector (per batch per channel) to be the size of the current layer
        self.delta = np.einsum('bci, bcij -> bcj', error, current.W) * self.activation_deriv

        assert self.delta.shape == (self.batch_dim, self.channel_dim, self.size)
        
        return self.delta

class ActivationLayer(BaseLayer):
    def __init__(self, activation_function: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.activation_function = activation_function

    def forward(self):
        if self.previous:
            self.A = self.activation_function(self.previous.A)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        ActivationLayer passes backwards derivative evaluated on its activations
        multiplied by fed back error vector.

        If the ActivationLayer's next pointer is a Layer (true except output layer)
        the error vector of the next layer is updated accordingly.
        """
        deriv = self.activation_function.derivative(self.A)

        if self.previous is None:
            raise ValueError('No previous layer to pass error to.')
        
        # previous layer to activation is always a Layer object
        # set the deriv attribute to the correct value
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

        self.batch_dim = layers[0].A.shape[0]
        self.channel_dim = layers[0].A.shape[1]

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Args:
            input_data shape: (batch_dim, channel_dim, input_layer.size)
        """
        assert input_data.shape == self.layers[0].A.shape

        self.layers[0].A = input_data
        for layer in self.layers[1:]:
            layer.forward()
        return self.layers[-1].A
    
    def MSELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((x - y) ** 2, axis=-1)
    
    def CELoss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(x), axis=-1)
    
    def backward(self, data_tensor: np.ndarray, label_tensor: np.ndarray, lr: float, loss_func: str) -> np.ndarray:
        """
        Perform one backwards pass.

        Args:
            data_tensor shape: (batch_dim, channel_dim, input_dim)
            label_tensor shape: (batch_dim, channel_dim, output_dim)
            lr: learning rate float
            loss_func: specifies which loss function to apply
        """
        forward_out = self.forward(data_tensor)

        assert forward_out.shape == label_tensor.shape

        if loss_func not in self.loss_functions:
            raise ValueError(f'Unsupported loss function: {loss_func}')
        
        loss = self.loss_functions[loss_func](forward_out, label_tensor)

        assert loss.shape == (self.batch_dim, self.channel_dim)

        error = forward_out - label_tensor
        current_error = error

        # back-propagate errors
        for layer in self.layers[-1:0:-1]:
            current_error = layer.backward(current_error)

        # perform gradient descent
        for layer in self.layers[1:]:
            if isinstance(layer, Layer):
                # delta.shape: (batch_dim, channel_dim, layer.size)
                # previous.A.shape: (batch_dim, channel_dim, previous.size)
                layer.B -= lr * layer.delta
                layer.W -= lr * np.einsum('bci, bcj -> bcij', layer.delta, layer.previous.A)

        return loss
    
    def learn(self, learn_data: List[Tuple[np.ndarray, np.ndarray]], lr: float, epochs: int, loss_func: str) -> None:
        """
        Args:
            learn_data: [(input_tensor, label_tensor), (input_tensor, label_tensor) ... ]
                each input_tensor has shape: (batch_dim, channel_dim, input_layer.size)
            lr: learning rate float
            epochs: number of training epochs int
            loss_func: specifies which loss function to apply
        """
        losses = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0

            for input, label in learn_data:
                loss = self.backward(input, label, lr, loss_func)
                
                # expect loss object to have shape (batch_dim, channel_dim, output_dim)
                epoch_loss += np.mean(loss)

            losses.append(epoch_loss / len(learn_data))

        plt.figure()
        plt.plot(range(epochs), losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
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
