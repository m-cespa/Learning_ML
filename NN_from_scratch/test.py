from model import Layer, ActivationLayer, Network, ReLU, Sigmoid, ELU, Linear, Tanh, Adam, SGD, GD, Lion, Quadratic
import numpy as np
from typing import List, Callable, Tuple, Optional
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams
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
    T = 1 / (k * (np.pi / L)**2)  # Define T: the maximum time for training. Adjust as needed.
    
    # Training domain: for each input dimension we define a range.
    # x in [0, L] and t in [0, T]
    domain = [(0, L), (0, T)]
    
    # Define the target function for the heat equation.
    # Note: x is a 1D array with two elements: [x, t]
    func = lambda x: np.sin(np.pi * x[0] / L) * np.exp(-x[1] * (k * np.pi / L)**2)
    
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
    T = 1 / (k * (np.pi / L)**2)  # Max time
    
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

    return collocation_inputs

def generate_boundary_data(N: int, k: float, L: float):
    T = 1 / (k * (np.pi / L)**2)  # Maximum time

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

def split_collocation_data(N_train, N_val, k, L):
    N_total = N_train + N_val
    all_points = generate_collocation_data(N_total, k, L)
    np.random.shuffle(all_points)

    collocation_train = all_points[:N_train]
    collocation_val = all_points[N_train:]

    return collocation_train, collocation_val

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
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       output_layer,
                       output_activation],
                       lambda1=0.01,
                       lambda2=0.001,
                       optim=Adam())

    # Generate XOR training data with batch and channel dimensions

    # learn_data = generate_test_data(
    #     total_samples=500,
    #     batch_dim=batch_dim,
    #     domain=[(0, 2*np.pi), (0, 2*np.pi)],
    #     input_size=2,
    #     func= lambda x: np.sin(x[0]) * np.cos(x[1])
    # )

    # --- Define the toy heat equation parameters and ground truth function ---
    def heat_eq_true(x, t, k=1., L=10.):
        """
        Analytical solution of the heat equation for the given toy problem:
        u(x,t) = 6 * sin(pi*x/L) * exp(-t*(k*pi/L)**2)
        """
        return np.sin(k * np.pi * x / L) * np.exp(-t * k * (np.pi / L)**2)

    # --- Generate training and evaluation data ---
    # Assume these functions already exist from your provided code.
    learn_data = generate_heat_eq_data(total_samples=500, batch_dim=batch_dim, k=1., L=10.)

    collocation_train, collocation_val = split_collocation_data(N_train=200, N_val=200, k=1., L=10.)
    boundary_data = generate_boundary_data(N=200, k=1., L=10.)

    history = network.learn(learn_data, lr=0.001, epochs=100, loss_func='mse',
                            collocation_data=collocation_train, boundary_data=boundary_data, store_grads=True,
                            test_collocation_data=collocation_val, track_diagnostics=True)
    
    def plot_training_history(history: dict, figsize=(12, 8)):
        """
        Plots the training history metrics from the learn() function output.

        Args:
            history (dict): Dictionary containing keys:
                - "total_loss": list of total losses per epoch
                - "physics_loss": list of physics losses per epoch
                - "residuals": list of mean residuals per epoch
                - "relative_errors": list of relative domain errors per epoch
            figsize (tuple): Size of the figure canvas
        """
        rcParams["mathtext.default"] = "regular"  # Enables nice math font

        metrics = {
            "total_losses": r"Total Loss",
            "residual_losses": r"Residual Loss",
            "validation_residuals": r"Mean PDE Residual",
            "relative_errors": r"Relative $\ell_2$ Error"
        }

        epochs = range(1, len(history["total_losses"]) + 1)
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()

        for i, (key, label) in enumerate(metrics.items()):
            if key in history:
                axs[i].plot(epochs, history[key], label=label, color="tab:blue")
                axs[i].set_title(label, fontsize=12)
                axs[i].set_xlabel("Epoch", fontsize=10)
                axs[i].set_ylabel(label, fontsize=10)
                axs[i].tick_params(axis="both", which="major", labelsize=9)
                axs[i].grid(True)

        plt.tight_layout()
        plt.show()

    plot_training_history(history)


