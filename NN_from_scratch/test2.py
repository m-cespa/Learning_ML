import numpy as np
from model import Layer, Network, ActivationLayer, Linear, Quadratic, Adam, ReLU, ELU, Tanh, Sin
from typing import List, Tuple, Optional, Callable

def generate_test_data(
    total_samples: int,
    batch_dim: int,
    input_size: int,
    func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    domain: Optional[List[Tuple[float, float]]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    num_batches = total_samples // batch_dim

    # set domain to default if not provided
    if domain is None:
        domain = [(0, 10) for _ in range(input_size)]
    elif len(domain) != input_size:
        raise ValueError("Length of domain must match input_size")

    # set default function if none provided
    if func is None:
        if input_size == 1:
            func = lambda x: np.sin(x[0])
        elif input_size == 2:
            func = lambda x: np.sin(x[0]) * np.cos(x[1])
        else:
            raise ValueError("No default function defined for input_size > 2. Please provide a function.")

    batches = []
    for _ in range(num_batches):
        input_tensor = np.zeros((batch_dim, input_size))
        label_tensor = np.zeros((batch_dim, 1))
        for i in range(batch_dim):
            x_i = np.array([np.random.uniform(low, high) for (low, high) in domain])
            y_i = func(x_i)
            input_tensor[i] = x_i
            label_tensor[i] = y_i
        batches.append((input_tensor, label_tensor))

    return batches


batch_dim = 5
input_size = 1
output_size = 1

# Define the layers with batch dimension support
input_layer = Layer(size=input_size)
input_activation = ActivationLayer(Linear())

output_layer = Layer(size=output_size)
output_activation = ActivationLayer(Sin())

# Link layers sequentially
network = Network(layers=[input_layer,
                    input_activation,
                    output_layer,
                    output_activation],
                    lambda1=0.0001,
                    lambda2=0.0001,
                    optim=Adam())

learn_data = generate_test_data(
    total_samples=1000,
    batch_dim=batch_dim,
    domain=[(0, 2*np.pi)],
    input_size=1,
    func= lambda x: np.sin(x[0])
)

# network.learn(learn_data, lr=0.001, epochs=100, loss_func='mse', store_grads=False, plot=True, decay=0.99)

network.layers[-2].W = np.array([[1.]])
network.layers[-2].B = np.array([0.])

network.forward(np.array([[np.pi/2]]), store_grads=True)

J, diagH, dJ, dH, full_H, full_T = network.autograd_numba()

print(J)
print(diagH)
print(dJ)
print(dH)

def numerical_jacobian_hessian(net, X: np.ndarray, h: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size, input_dim = X.shape
    eps = 1e-12
    
    # Original outputs
    u0 = net.forward(X, store_grads=False).flatten()  # shape: (batch_size,)
    
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
        perturbed_outputs = net.forward(perturbed_inputs, store_grads=False)
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

print(numerical_jacobian_hessian(network, X=np.array([[np.pi/2]])))
 
