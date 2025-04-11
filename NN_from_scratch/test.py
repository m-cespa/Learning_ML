from model import Layer, ActivationLayer, Network, ReLU, Sigmoid, ELU, Linear, Tanh, Adam, SGD, GD, Lion
import numpy as np
from typing import List, Callable, Tuple
import matplotlib.pyplot as plt
import time
import csv

def generate_xor_data(num_pairs, batch_dim, input_size, output_size):
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

        # Create input tensor with shape (batch_dim, input_size)
        input_tensor = np.tile(input_data, (batch_dim, 1)).reshape(batch_dim, input_size)

        # Create label tensor with shape (batch_dim, output_size)
        label_tensor = np.tile(output_data, (batch_dim, 1)).reshape(batch_dim, output_size)

        learn_data.append((input_tensor, label_tensor))
    
    return learn_data

def generate_trig_data(num_pairs: int, batch_dim: int, input_size: int, 
                       func: Callable[[np.ndarray], np.ndarray]=None) -> list:
    """
    Generate training data for a trigonometric function.
    Returns:
        list: List of (input_tensor, label_tensor) tuples.
    """
    domain = tuple((0, 2*np.pi) for _ in range(input_size))

    if func is None:
        if input_size == 1:
            func = lambda x: np.array([np.sin(x[0])])
        elif input_size == 2:
            func = lambda x: np.array([np.sin(x[0]) * np.sin(x[1])])
        else:
            raise ValueError("No default function defined for input_size > 2. Please provide a function.")

    learn_data = []
    for _ in range(num_pairs):
        input_tensor = np.zeros((batch_dim, input_size))
        label_tensor = np.zeros((batch_dim, 1))  # Assuming 1D output

        for i in range(batch_dim):
            x_i = np.array([np.random.uniform(*domain[j]) for j in range(input_size)])
            y_i = func(x_i)
            input_tensor[i] = x_i
            label_tensor[i] = y_i

        learn_data.append((input_tensor, label_tensor))

    return learn_data



if __name__ == "__main__":
    # Define the input, output sizes, batch dimensions
    batch_dim = 10
    input_size = 1
    output_size = 1

    # Define the layers with batch dimension support
    input_layer = Layer(size=input_size)
    activation1 = ActivationLayer(Linear())
    hidden_layer_1 = Layer(size=4)
    activation2 = ActivationLayer(ELU())
    hidden_layer_3 = Layer(size=8)
    activation4 = ActivationLayer(ELU())
    hidden_layer_4 = Layer(size=8)
    activation5 = ActivationLayer(ELU())
    output_layer = Layer(size=output_size)
    output_activation = ActivationLayer(Tanh())

    # Link layers sequentially
    network = Network(layers=[input_layer,
                       activation1,
                       hidden_layer_1,
                       activation2,
                       hidden_layer_3,
                       activation4,
                       hidden_layer_4,
                       activation5,
                       output_layer,
                       output_activation],
                       physics_loss_weight=0.001,
                       optim=Adam())

    # Generate XOR training data with batch and channel dimensions
    learn_data = generate_trig_data(
        num_pairs=200,
        batch_dim=batch_dim,
        input_size=input_size,
    )

    # Print shapes of the first label and input tensors to verify correctness
    print(f"Shape of label tensor: {learn_data[0][1].shape}")  # Should be (batch_dim, channel_dim, output_size)
    print(f"Shape of input tensor: {learn_data[0][0].shape}")  # Should be (batch_dim, channel_dim, input_size)

    # print(network.layers[0].A.shape)
    # test datum to probe Jacobian and Hessian
    # learn_data.append((np.array([[1]]), np.array([[0.84147]])))
    
    network.learn(learn_data=learn_data, lr=0.001, epochs=50, loss_func='mse', physics_loss=True, plot=True, store_grads=True)

    for layer in network.layers:
        if isinstance(layer, Layer):
            if hasattr(layer, 'g'):
                print(f"{layer} has g: {layer.g.shape}")

    # print(network.layers[0].z.shape)
    # network.autograd()
    # network.autograd_derivs()

    # print(network.J)
    # print(network.H)
    # print(network.dJ_da)
    # print(network.dH_da)

    # print(f"\n{network.complex_step_derivative(1)}")
    # print(f"\n{network.forward(np.array([1]))}")
    # print(f"\n{network.numerical_first_derivative(np.array([1]))}")
    # print(f"\n{network.numerical_second_derivative(np.atleast_2d(1))}")


    # times = []
    # for _ in range(5):
    #     start = time.time()
    #     network.learn(learn_data=learn_data, lr=0.001, epochs=10, loss_func='mse', physics_loss=False)
    #     dt = time.time() - start

    #     times.append(dt)
    #     network.setup()

    # # with open('10_epochs.csv', mode='w', newline='') as file:
    # #     writer = csv.writer(file)
    # #     writer.writerow(['Run', 'non_batched_time (s)'])  # Header
    # #     for i, t in enumerate(times, 1):
    # #         writer.writerow([i, t])

    # with open('10_epochs.csv', mode='r', newline='') as file:
    #     reader = list(csv.reader(file))
    #     header = reader[0]
    #     rows = reader[1:]

    # # Your new batched times (same number of runs)

    # # Add new column to header
    # header.append('non_physics_time (s)')

    # # Add new column to each row
    # for i, row in enumerate(rows):
    #     row.append(str(times[i]))

    # # Write the updated data back
    # with open('10_epochs.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)
    #     writer.writerows(rows)
    
    # print(network.current_lr)

    # test_inputs = np.linspace(0, 2*np.pi, 100)
    # model_outputs = [network.forward(np.atleast_2d(input)).squeeze(0) for input in test_inputs]

    # plt.figure()
    # plt.plot(test_inputs, np.sin(test_inputs), label="True")
    # plt.plot(test_inputs, model_outputs, label="Model")
    # plt.legend()
    # plt.show()

    # network.interactive_inference()

    # Optionally print weights to check
    # print(f'\nInput layer W:\n{input_layer.W}\nInput layer B:\n{input_layer.B}')
    # print(f'\nFirst Hidden layer W:\n{hidden_layer_1.W}\nFirst Hidden layer B:\n{hidden_layer_1.B}')
