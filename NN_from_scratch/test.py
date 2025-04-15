from model import Layer, ActivationLayer, Network, ReLU, Sigmoid, ELU, Linear, Tanh, Adam, SGD, GD, Lion, Quadratic
import numpy as np
from typing import List, Callable, Tuple, Optional
import matplotlib.pyplot as plt
import time
import csv

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

def generate_heat_eq_data(total_samples: int, batch_dim: int, k: float, L: float):
    num_batches = total_samples // batch_dim  # e.g. 200/5 = 40 mini-batches.
    T = 2 / (k * (np.pi / L)**2)  # Define T: the maximum time for training. Adjust as needed.
    
    # Training domain: for each input dimension we define a range.
    # x in [0, L] and t in [0, T]
    domain = [(0, L), (0, T)]
    
    # Define the target function for the heat equation.
    # Note: x is a 1D array with two elements: [x, t]
    func = lambda x: 6 * np.sin(np.pi * x[0] / L) * np.exp(-x[1] * (k * np.pi / L)**2)
    
    batches = []
    for _ in range(num_batches):
        # Create empty tensors for a mini-batch.
        input_tensor = np.zeros((batch_dim, 2))
        label_tensor = np.zeros((batch_dim, 1))
        for i in range(batch_dim):
            # Generate one sample by sampling uniformly from the domain intervals.
            x_i = np.array([np.random.uniform(low, high) for (low, high) in domain])
            y_i = func(x_i)
            input_tensor[i] = x_i
            label_tensor[i] = y_i  # y_i is a scalar, stored as (1,)
        batches.append((input_tensor, label_tensor))
    
    return batches

def generate_collocation_data(N: int, k: float, L: float):
    T = 2 / (k * (np.pi / L)**2)  # Max time
    
    # Compute minimal grid resolution to get at least N points
    grid_size = int(np.ceil(np.sqrt(N)))
    while grid_size**2 < N:
        grid_size += 1

    # Build denser grid
    x_vals = np.linspace(0, L, grid_size)
    t_vals = np.linspace(0, T, grid_size)
    X, T_ = np.meshgrid(x_vals, t_vals)
    grid_points = np.stack([X.flatten(), T_.flatten()], axis=1)

    # Randomly sample exactly N points without replacement
    indices = np.random.choice(len(grid_points), size=N, replace=False)
    collocation_inputs = grid_points[indices]

    return [k, L, collocation_inputs]


def generate_boundary_data(N: int, k: float, L: float):
    T = 2 / (k * (np.pi / L)**2)  # Maximum time

    # u(x,0): x in [0, L], t = 0.
    # u(0,t): t in [0, T], x = 0.
    # u(L,t): t in [0, T], x = L.

    # We determine a grid size per boundary so that we have at least N points in total.
    n_boundary = int(np.ceil(N / 3))  # points per boundary (roughly)
    
    # Boundary 1: u(x, 0)
    x1 = np.linspace(0, L, n_boundary)
    t1 = np.zeros(n_boundary)
    pts1 = np.stack([x1, t1], axis=1)
    
    # Boundary 2: u(0, t)
    t2 = np.linspace(0, T, n_boundary)
    x2 = np.zeros(n_boundary)
    pts2 = np.stack([x2, t2], axis=1)
    
    # Boundary 3: u(L, t)
    t3 = np.linspace(0, T, n_boundary)
    x3 = np.full(n_boundary, L)
    pts3 = np.stack([x3, t3], axis=1)
    
    # Combine all boundary points
    all_pts = np.concatenate([pts1, pts2, pts3], axis=0)
    
    # If we have more points than needed, randomly sample exactly N points (without replacement)
    if all_pts.shape[0] > N:
        indices = np.random.choice(all_pts.shape[0], size=N, replace=False)
        boundary_inputs = all_pts[indices]
    else:
        boundary_inputs = all_pts  # if exactly N or fewer (unlikely), return all
    
    return boundary_inputs


if __name__ == "__main__":
    # Define the input, output sizes, batch dimensions
    batch_dim = 5
    input_size = 2
    output_size = 1

    # Define the layers with batch dimension support
    input_layer = Layer(size=input_size)
    input_activation = ActivationLayer(Linear())

    output_layer = Layer(size=output_size)
    output_activation = ActivationLayer(Tanh())

    # Link layers sequentially
    network = Network(layers=[input_layer,
                       input_activation,
                       Layer(size=4),
                       ActivationLayer(ELU()),
                       Layer(size=8),
                       ActivationLayer(ELU()),
                       Layer(size=16),
                       ActivationLayer(ELU()),
                       Layer(size=8),
                       ActivationLayer(ELU()),
                       output_layer,
                       output_activation],
                       physics_loss_weight=0.001,
                       optim=Adam())

    # Generate XOR training data with batch and channel dimensions
    # learn_data = generate_heat_eq_data(
    #     total_samples=200,
    #     batch_dim=batch_dim,
    #     k=1.,
    #     L=5.
    # )

    learn_data = generate_test_data(
        total_samples=500,
        batch_dim=batch_dim,
        domain=[(0, 2*np.pi), (0, 2*np.pi)],
        input_size=2,
        func= lambda x: np.sin(x[0]) * np.cos(x[1])
    )

    collocation_data = generate_collocation_data(N=200, k=1., L=5.)
    boundary_data = generate_boundary_data(N=200, k=1., L=5.)

    # Print shapes of the first label and input tensors to verify correctness
    print(f"Shape of label tensor: {learn_data[0][1].shape}")  # Should be (batch_dim, output_size)
    print(f"Shape of input tensor: {learn_data[0][0].shape}")  # Should be (batch_dim, input_size)
    
    network.learn(learn_data=learn_data, lr=0.001, epochs=200, loss_func='mse', 
                  collocation_data=None, boundary_data=boundary_data, plot=True, store_grads=True)

    # # Parameters
    # k = 1.0
    # L = 5.0
    # # Define T such that t in [0, T] covers the intended training time.
    # T = 2 / ((np.pi / L)**2)

    # # Define the analytic solution:
    # def analytic_u(x, t):
    #     # x: numpy array, t: scalar
    #     return 6 * np.sin(np.pi * x / L) * np.exp(-t * (k * np.pi / L)**2)

    # # Generate a dense grid of x values covering [0, 2L] for visualization
    # x_vals = np.linspace(0, L, 200)

    # # Choose time instances.
    # # 4 times uniformly in [0, T] and 2 times uniformly in [T, T+5]
    # times_inside = np.linspace(0, T, 3)
    # times_outside = np.linspace(T+3, T+3, 1)
    # all_times = np.concatenate((times_inside, times_outside))

    # # Create the figure and axis (one single plot)
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()

    # # Define a set of colors using a colormap for clarity.
    # colors = plt.cm.viridis(np.linspace(0, 1, len(all_times)))

    # # Loop over each time instance, plot both analytic and network curves.
    # for idx, t in enumerate(all_times):
    #     # For each t, create a batch input with shape (200,2): each row is [x, t]
    #     X = np.vstack((x_vals, np.full_like(x_vals, t))).T  # shape (200, 2)
        
    #     # Get network prediction:
    #     pred = network.forward(input_data=X, store_grads=False)  # expected shape (200,1)
    #     pred = np.squeeze(pred)  # shape (200,)
        
    #     # Compute ground truth from the analytic solution
    #     gt = analytic_u(x_vals, t)
        
    #     # Plot analytic solution as a solid line, network prediction as dashed.
    #     # Label only one representative label per time.
    #     ax.plot(x_vals, gt, color=colors[idx], linestyle='-', linewidth=2,
    #             label=f"Analytic, t={t:.2f}")
    #     ax.plot(x_vals, pred, color=colors[idx], linestyle='--', linewidth=2,
    #             label=f"Network, t={t:.2f}")

    # # Improve the plot
    # ax.set_title(r"Network vs Analytic Solution$", fontsize=16)
    # ax.set_xlabel(r"$x$", fontsize=14)
    # ax.set_ylabel(r"$u(x,t)$", fontsize=14)
    # ax.legend(fontsize=10, loc='upper right', ncol=2)
    # ax.grid(True)

    # plt.tight_layout()
    # plt.show()

