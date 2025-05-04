from model import Layer, ActivationLayer, Network, ReLU, Sigmoid, ELU, Linear, Tanh, Adam, SGD, GD, Lion, Quadratic
import numpy as np
from numpy import pi, sin, cos, exp
from typing import List, Callable, Tuple, Optional
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams
import time
import csv
import pickle

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

def generate_heat_eq_data(total_samples: int, batch_dim: int, k: float, L: float, A: float):
    num_batches = total_samples // batch_dim  # e.g. 200/5 = 40 mini-batches.
    T = 1 / (k * (pi / L)**2)  # Define T: the maximum time for training. Adjust as needed.
    
    # Training domain: for each input dimension we define a range.
    # x in [0, L] and t in [0, T]
    domain = [(0, L), (0, T)]
    
    # Define the target function for the heat equation.
    # Note: x is a 1D array with two elements: [x, t]
    func = lambda x: 1. + A * cos(pi * x[0] / L) * exp(-k * x[1] * (pi / L)**2)
    
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
    T = 1 / (k * (pi / L)**2)  # Max time
    
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

def generate_boundary_data(N: int, k: float, L: float, A: float):
    T = 1 / (k * (pi / L)**2)  # Final time for the simulation window
    n_each = N // 2  # Split equally between IC and BC

    # Initial condition: u(x, 0) = 1 + A * cos(pi * x / L)
    x_ic = np.linspace(0, L, n_each)
    t_ic = np.zeros_like(x_ic)
    inputs_ic = np.stack([x_ic, t_ic], axis=1)
    targets_ic = (1. + A * cos(pi * x_ic / L)).reshape(-1, 1)

    # Boundary condition: u(L/2, t) = 1
    t_bc = np.linspace(0, T, n_each)
    x_bc = np.full_like(t_bc, L / 2)
    inputs_bc = np.stack([x_bc, t_bc], axis=1)
    targets_bc = np.ones_like(t_bc).reshape(-1, 1)

    # Concatenate inputs and targets
    boundary_inputs = np.concatenate([inputs_ic, inputs_bc], axis=0)
    boundary_targets = np.concatenate([targets_ic, targets_bc], axis=0)

    return (boundary_inputs, boundary_targets)

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
    output_activation = ActivationLayer(Linear())

    # Link layers sequentially
    network = Network(layers=[input_layer,
                       input_activation,
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       Layer(size=20),
                       ActivationLayer(ELU()),
                       Layer(size=20),
                       ActivationLayer(ELU()),
                       Layer(size=10),
                       ActivationLayer(ELU()),
                       output_layer,
                       output_activation],
                       lambda1=0.5,
                       lambda2=0.5,
                       optim=Adam())

    # learn_data = generate_test_data(
    #     total_samples=500,
    #     batch_dim=batch_dim,
    #     domain=[(0, 2*np.pi), (0, 2*np.pi)],
    #     input_size=2,
    #     func= lambda x: np.sin(x[0]) * np.cos(x[1])
    # )

    # --- Define the toy heat equation parameters and ground truth function ---
    def heat_eq_true(x, t, k=0.1, L=10., A=0.5):
        """
        Analytical solution of the heat equation for the given toy problem:
        u(x,t) = 6 * sin(pi*x/L) * exp(-t*(k*pi/L)**2)
        """
        return 1. + A * cos(pi * x / L) * exp(-k * t * (pi / L)**2)

    # --- Generate training and evaluation data ---
    # Assume these functions already exist from your provided code.
    learn_data = generate_heat_eq_data(total_samples=500, batch_dim=batch_dim, k=0.5, L=10., A=0.5)

    collocation_train, collocation_val = split_collocation_data(N_train=400, N_val=200, k=0.5, L=10.)
    boundary_data = generate_boundary_data(N=400, k=0.5, L=10., A=0.5)
    
    history_pinn = network.learn(learn_data, lr=0.008, epochs=150, loss_func='mse',
                            collocation_data=collocation_train, boundary_data=boundary_data, store_grads=True,
                            test_collocation_data=collocation_val, track_diagnostics=True, plot=True)
    
    network.save_parameters(file_path='runge_kutta_comparison.npz')

    network.load_parameters(file_path='runge_kutta_comparison.npz')
    
    x_inputs = np.linspace(0, 10, 100)
    T = 1 / (0.5 * (pi / 10.) ** 2)
    t_inputs = np.array([0, T, 2*T])
    X, T_ = np.meshgrid(x_inputs, t_inputs)
    eval_points = np.stack([X.flatten(), T_.flatten()], axis=1)

    model_out = network.forward(eval_points, store_grads=False).squeeze()
    true_out = heat_eq_true(eval_points[:, 0], eval_points[:, 1], k=0.5, L=10., A=0.5)

    # Reshape to (3, 100) = (n_times, n_x)
    model_out = model_out.reshape(len(t_inputs), len(x_inputs))
    true_out = true_out.reshape(len(t_inputs), len(x_inputs))

    plt.figure(figsize=(10, 6))
    for i, t in enumerate(t_inputs):
        plt.plot(x_inputs, model_out[i], label=f"Model t={t:.2f}")
        plt.plot(x_inputs, true_out[i], '--', label=f"True t={t:.2f}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Model vs True Solution at Different Times")
    plt.legend()
    plt.show()

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
            "total_losses": r"Total Training Loss",
            "residual_losses": r"Residual Training Loss on Training Set",
            "validation_residuals": r"Mean PDE Residual on Validation Set",
            "relative_errors": r"Relative $\ell_2$ Error on Validation Set"
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

        plt.show()

    plot_training_history(history_pinn)
    


