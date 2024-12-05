from model import Layer, ActivationLayer, Network, ReLU, Sigmoid
import numpy as np

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

if __name__ == "__main__":
    # Define the input, output sizes, batch, and channel dimensions
    batch_dim = 4
    channel_dim = 2
    input_size = 2
    output_size = 1

    # Define the layers with batch and channel dimension support
    input_layer = Layer(size=input_size, batch_dim=batch_dim, channel_dim=channel_dim)
    activation1 = ActivationLayer(ReLU())
    hidden_layer_1 = Layer(size=32, batch_dim=batch_dim, channel_dim=channel_dim)
    activation2 = ActivationLayer(ReLU())
    hidden_layer_2 = Layer(size=16, batch_dim=batch_dim, channel_dim=channel_dim)
    activation3 = ActivationLayer(ReLU())
    hidden_layer_3 = Layer(size=8, batch_dim=batch_dim, channel_dim=channel_dim)
    activation4 = ActivationLayer(ReLU())
    output_layer = Layer(size=output_size, batch_dim=batch_dim, channel_dim=channel_dim)
    activation5 = ActivationLayer(Sigmoid())  # Sigmoid for binary classification

    # Link layers sequentially
    network = Network([input_layer,
                       activation1,
                       hidden_layer_1,
                       activation2,
                       hidden_layer_2,
                       activation3,
                       hidden_layer_3,
                       activation4,
                       output_layer,
                       activation5])

    # Generate XOR training data with batch and channel dimensions
    learn_data = generate_xor_data(
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
    network.learn(learn_data=learn_data, lr=0.008, epochs=50, loss_func='ce')

    # Optionally print weights to check
    print(f'\nInput layer W:\n{input_layer.W}\nInput layer B:\n{input_layer.B}')
    print(f'\nFirst Hidden layer W:\n{hidden_layer_1.W}\nFirst Hidden layer B:\n{hidden_layer_1.B}')
