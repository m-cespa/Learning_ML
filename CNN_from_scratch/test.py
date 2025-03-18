from model import Layer, ActivationLayer, Network, ReLU, Sigmoid
import numpy as np
from typing import List, Callable, Tuple

def generate_xor_data(num_pairs, batch_dim, channel_dim, input_size, output_size):
    """
    Generate XOR data as (label_tensor, input_tensor) tuples.
    Each tensor has batch and channel dimensions.
    """
    learn_data = []
    xor_pairs = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    for i in range(num_pairs):
        input_data, output_data = xor_pairs[i % 4]

        # Create input tensor with shape (batch_dim, channel_dim, input_size)
        input_tensor = np.tile(input_data, (batch_dim, channel_dim, 1)).reshape(batch_dim, channel_dim, input_size)

        # Create label tensor with shape (batch_dim, channel_dim, output_size)
        label_tensor = np.tile(output_data, (batch_dim, channel_dim, 1)).reshape(batch_dim, channel_dim, output_size)

        learn_data.append((input_tensor, label_tensor))
    
    return learn_data

def generate_trig_data(num_pairs: int, batch_dim: int, input_size: int, output_size: int, 
                         channel_dim: int, domain: Tuple[Tuple[float, float], ...]=None,
                         func: Callable[[np.ndarray], np.ndarray]=None) -> list:
    """
    Generate training data for a trigonometric function.
    
    The generated data consists of (input_tensor, label_tensor) tuples, each of shape 
    (batch_dim, channel_dim, dimension). By default, if no function is provided:
    
        - For input_size == 1: f(x) = sin(x1)
        - For input_size == 2: f(x) = sin(x1)*sin(x2)
    
    Parameters:
        num_pairs (int): Number of training pairs to generate.
        batch_dim (int): Batch dimension size.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        channel_dim (int): Channel dimension size.
        domain (tuple): Tuple of input_size tuples, each specifying (min, max) for that input dimension.
                        Defaults to ((0, 2*pi), ...).
        func (callable): Function that takes an input array of shape (input_size,) and returns an array 
                         of shape (output_size,). If None, a default is used.
                         
    Returns:
        list: List of (input_tensor, label_tensor) tuples.
    """
    # Default domain: (0, 2*pi) for each input dimension
    if domain is None:
        domain = tuple((0, 2*np.pi) for _ in range(input_size))
    
    # Default function based on input_size
    if func is None:
        if input_size == 1:
            func = lambda x: np.array([np.sin(x[0])])
        elif input_size == 2:
            func = lambda x: np.array([np.sin(x[0]) * np.sin(x[1])])
        else:
            raise ValueError("No default function defined for input_size > 2. Please provide a function.")
    
    learn_data = []
    for i in range(num_pairs):
        # Generate random inputs for each dimension according to the domain
        input_data = np.array([np.random.uniform(domain[j][0], domain[j][1]) for j in range(input_size)])
        # Compute the output using the provided function
        output_data = func(input_data)
        
        # Create input tensor with shape (batch_dim, channel_dim, input_size)
        input_tensor = np.tile(input_data, (batch_dim, channel_dim, 1))
        # Create label tensor with shape (batch_dim, channel_dim, output_size)
        label_tensor = np.tile(output_data, (batch_dim, channel_dim, 1))
        
        learn_data.append((input_tensor, label_tensor))
    
    return learn_data


if __name__ == "__main__":
    # Define the input, output sizes, batch, and channel dimensions
    batch_dim = 4
    channel_dim = 1
    input_size = 1
    output_size = 1

    # Define the layers with batch and channel dimension support
    input_layer = Layer(size=input_size, batch_dim=batch_dim, channel_dim=channel_dim)
    activation1 = ActivationLayer(Sigmoid())
    hidden_layer_1 = Layer(size=32, batch_dim=batch_dim, channel_dim=channel_dim)
    activation2 = ActivationLayer(Sigmoid())
    hidden_layer_2 = Layer(size=16, batch_dim=batch_dim, channel_dim=channel_dim)
    activation3 = ActivationLayer(Sigmoid())
    hidden_layer_3 = Layer(size=16, batch_dim=batch_dim, channel_dim=channel_dim)
    activation4 = ActivationLayer(Sigmoid())
    hidden_layer_4 = Layer(size=8, batch_dim=batch_dim, channel_dim=channel_dim)
    activation5 = ActivationLayer(Sigmoid())
    output_layer = Layer(size=output_size, batch_dim=batch_dim, channel_dim=channel_dim)
    activation6 = ActivationLayer(Sigmoid())  # Sigmoid for binary classification

    # Link layers sequentially
    network = Network([input_layer,
                       activation1,
                       hidden_layer_1,
                       activation2,
                       hidden_layer_2,
                       activation3,
                       hidden_layer_3,
                       activation4,
                       hidden_layer_4,
                       activation5,
                       output_layer,
                       activation6])

    # Generate XOR training data with batch and channel dimensions
    learn_data = generate_trig_data(
        num_pairs=100,
        batch_dim=batch_dim,
        channel_dim=channel_dim,
        input_size=input_size,
        output_size=output_size
    )

    # Print shapes of the first label and input tensors to verify correctness
    print(f"Shape of label tensor: {learn_data[0][1].shape}")  # Should be (batch_dim, channel_dim, output_size)
    print(f"Shape of input tensor: {learn_data[0][0].shape}")  # Should be (batch_dim, channel_dim, input_size)

    print(network.layers[0].A.shape)

    # Train the network with the training data
    network.learn(learn_data=learn_data, lr=0.1, epochs=200, loss_func='mse')
    print(network.current_lr)

    network.interactive_inference()

    # Optionally print weights to check
    # print(f'\nInput layer W:\n{input_layer.W}\nInput layer B:\n{input_layer.B}')
    # print(f'\nFirst Hidden layer W:\n{hidden_layer_1.W}\nFirst Hidden layer B:\n{hidden_layer_1.B}')
