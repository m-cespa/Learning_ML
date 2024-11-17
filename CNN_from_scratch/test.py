from model import Layer, ActivationLayer, Network, ReLU, Sigmoid
import numpy as np

def generate_xor_data(num_pairs):
    train_data = []
    xor_pairs = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    for i in range(num_pairs):
        input_data, output_data = xor_pairs[i % 4]
        train_data.append([np.array(input_data), np.array(output_data)])
    
    return train_data

if __name__ == "__main__":
    input_layer = Layer(size=2)
    activation1 = ActivationLayer(Sigmoid())
    hidden_layer_1 = Layer(size=32)
    activation2 = ActivationLayer(ReLU())
    hidden_layer_2 = Layer(size=16)
    activation3 = ActivationLayer(ReLU())
    hidden_layer_3 = Layer(size=8)
    activation4 = ActivationLayer(Sigmoid())
    output_layer = Layer(size=1)
    activation5 = ActivationLayer(ReLU())

    # layers linked sequentially
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

    # XOR training data for trial attempt
    train_data = generate_xor_data(100)

    test_data = np.array([0,0])

    # print(network.forward(test_data))

    network.learn(learn_data=train_data, lr=0.008, epochs=50, loss_func='ce')

    # print(f'\nInput layer W:\n{input_layer.W}\nInput layer B:\n{input_layer.B}')
    # print(f'\nFirst Hidden layer W:\n{hidden_layer_1.W}\nFirst Hidden layer B:\n{hidden_layer_1.B}')
