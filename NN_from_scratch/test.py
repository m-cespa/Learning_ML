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
    output_activation = ActivationLayer(Linear())

    # Link layers sequentially
    network = Network(layers=[input_layer,
                       input_activation,
                       Layer(size=8),
                       ActivationLayer(ELU()),
                       Layer(size=16),
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

    # learn_data = generate_test_data(
    #     total_samples=1000,
    #     batch_dim=batch_dim,
    #     domain=[(0, 10), (0, 10)],
    #     input_size=2,
    #     func= lambda x: 3*x[0] + 4*x[1]
    # )

    # --- Define the toy heat equation parameters and ground truth function ---
    def heat_eq_true(x, t, k=1., L=5.):
        """
        Analytical solution of the heat equation for the given toy problem:
        u(x,t) = 6 * sin(pi*x/L) * exp(-t*(k*pi/L)**2)
        """
        return 6 * np.sin(np.pi * x / L) * np.exp(-t * (k * np.pi / L)**2)

    # --- Generate training and evaluation data ---
    # Assume these functions already exist from your provided code.
    learn_data = generate_heat_eq_data(total_samples=500, batch_dim=5, k=1., L=5.)
    collocation_data = generate_collocation_data(N=200, k=1., L=5.)
    boundary_data = generate_boundary_data(N=100, k=1., L=5.)

    # Maximum training time T from your data generation:
    T = 2 / (1. * (np.pi / 5.)**2)
    L_val = 5.
    # Create x grid for evaluation (200 points along the interval [0, L])
    x_vals = np.linspace(0, L_val, 200)

    # --- Setup CSV Logging ---
    csv_filename = "pinn_vs_nn_results.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['model_type', 'epochs', 'runtime_sec', 'error_alpha1', 'error_alpha1.5', 'error_alpha2']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # --- Define a helper to evaluate model accuracy ---
        def evaluate_model(model, alpha, k=1., L=5.):
            """
            Evaluate the trained model at t = alpha * T over x in [0, L].
            Returns the mean squared error against the true solution.
            """
            t_val = alpha * T
            # Create evaluation input: each row is [x, t]
            X_eval = np.vstack((x_vals, np.full_like(x_vals, t_val))).T  # shape: (200,2)
            # Get the prediction; assuming model.forward supports batching as described.
            pred = model.forward(input_data=X_eval, store_grads=False)
            pred = np.squeeze(pred)  # shape: (200,)
            # Compute true values
            true_val = heat_eq_true(x_vals, t_val, k=k, L=L)
            # Return mean squared error
            mse = np.mean((pred - true_val)**2)
            return mse

        # --- Loop over experiment configurations ---
        # We define the list of epoch values.
        epoch_list = list(range(10, 101, 10))
        # Two configurations: one for the PINN and one for regular NN.
        for model_type, colloc_data in [('PINN', collocation_data), ('NN', None)]:
            print(f"Starting experiments for model type: {model_type}")
            for epochs in epoch_list:
                # Reset or reinitialize your network here if needed.
                # For example: network.reset_weights()  or construct a new model instance.
                #
                # Start timing training
                t0 = time.time()
                
                # Train the model: use colloc_data for PINN, or None for a regular NN.
                network.learn(
                    learn_data=learn_data,
                    lr=0.001,
                    epochs=epochs,
                    loss_func='mse',
                    collocation_data=colloc_data,
                    boundary_data=boundary_data,
                    plot=False,
                    store_grads=True
                )
                
                runtime_sec = time.time() - t0
                
                # Evaluate on t = alpha * T for alpha=1, 1.5, 2.
                error_alpha1   = evaluate_model(network, alpha=1)
                error_alpha15  = evaluate_model(network, alpha=1.5)
                error_alpha2   = evaluate_model(network, alpha=2)
                
                # Log the results:
                writer.writerow({
                    'model_type': model_type,
                    'epochs': epochs,
                    'runtime_sec': runtime_sec,
                    'error_alpha1': error_alpha1,
                    'error_alpha1.5': error_alpha15,
                    'error_alpha2': error_alpha2
                })
                
                print(f"[{model_type}] Epochs: {epochs}, Runtime: {runtime_sec:.2f} sec, "
                    f"Errors: (α=1: {error_alpha1:.4f}, α=1.5: {error_alpha15:.4f}, α=2: {error_alpha2:.4f})")
                
                # Optionally: Save model state or reset network weights before next experiment, if needed.


    # Parameters
    k = 1.0
    L = 5.0
    # Define T such that t in [0, T] covers the intended training time.
    T = 2 / ((np.pi / L)**2)

    # Define the analytic solution:
    def analytic_u(x, t):
        # x: numpy array, t: scalar
        return 6 * np.sin(np.pi * x / L) * np.exp(-t * (k * np.pi / L)**2)

    # Generate a dense grid of x values covering [0, 2L] for visualization
    x_vals = np.linspace(0, L, 200)

    # Choose time instances.
    # 4 times uniformly in [0, T] and 2 times uniformly in [T, T+5]
    times_inside = np.linspace(0, T, 3)
    times_outside = np.linspace(1.5*T, 1.5*T, 1)
    all_times = np.concatenate((times_inside, times_outside))

    # Create the figure and axis (one single plot)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Define a set of colors using a colormap for clarity.
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_times)))

    # Loop over each time instance, plot both analytic and network curves.
    for idx, t in enumerate(all_times):
        # For each t, create a batch input with shape (200,2): each row is [x, t]
        X = np.vstack((x_vals, np.full_like(x_vals, t))).T  # shape (200, 2)
        
        # Get network prediction:
        pred = network.forward(input_data=X, store_grads=False)  # expected shape (200,1)
        pred = np.squeeze(pred)  # shape (200,)
        
        # Compute ground truth from the analytic solution
        gt = analytic_u(x_vals, t)
        
        # Plot analytic solution as a solid line, network prediction as dashed.
        # Label only one representative label per time.
        ax.plot(x_vals, gt, color=colors[idx], linestyle='-', linewidth=2,
                label=f"Analytic, t={t:.2f}")
        ax.plot(x_vals, pred, color=colors[idx], linestyle='--', linewidth=2,
                label=f"Network, t={t:.2f}")

    # Improve the plot
    ax.set_title(r"Network vs Analytic Solution$", fontsize=16)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$u(x,t)$", fontsize=14)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    def compute_errors_over_domain(net, N: int = 100, h: float = 1e-5):
        x_vals = np.linspace(0, 10, N)
        y_vals = np.linspace(0, 10, N)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        error_J     = np.zeros((N, N))
        error_H     = np.zeros((N, N))
        error_dJ_da = np.zeros((N, N))
        error_dH_da = np.zeros((N, N))
        
        # Arrays to store all analytical and numerical values for averaging.
        analytical_J     = []
        analytical_H     = []
        analytical_dJ_da = []
        analytical_dH_da = []
        
        numerical_J     = []
        numerical_H     = []
        numerical_dJ_da = []
        numerical_dH_da = []
                
        for i in range(N):
            for j in range(N):
                inp = np.array([[X[i, j], Y[i, j]]], dtype=np.float64)
                
                # Forward pass and autograd.
                _ = net.forward(inp, store_grads=True)
                net.autograd()        # Updates net.J_batch, net.H_batch.
                net.autograd_derivs() # Updates net.dJ_da, net.dH_da.
                
                # Autograd results.
                autograd_J     = np.squeeze(net.J_batch)
                autograd_H     = np.squeeze(net.H_batch)
                autograd_dJ_da = np.squeeze(net.dJ_da)
                autograd_dH_da = np.squeeze(net.dH_da)
                
                # Numerical results.
                numJ, numH, num_dJ_da, num_dH_da = net.numerical_jacobian_hessian(inp, h=h)
                numJ     = np.squeeze(numJ)
                numH     = np.squeeze(numH)
                num_dJ_da = np.squeeze(num_dJ_da)
                num_dH_da = np.squeeze(num_dH_da)
                
                # Store for averaging.
                analytical_J.append(autograd_J)
                analytical_H.append(autograd_H)
                analytical_dJ_da.append(autograd_dJ_da)
                analytical_dH_da.append(autograd_dH_da)
                
                numerical_J.append(numJ)
                numerical_H.append(numH)
                numerical_dJ_da.append(num_dJ_da)
                numerical_dH_da.append(num_dH_da)
                
                # Compute errors.
                error_J[i, j]     = np.linalg.norm(autograd_J - numJ)
                error_H[i, j]     = np.linalg.norm(autograd_H - numH)
                error_dJ_da[i, j] = np.linalg.norm(autograd_dJ_da - num_dJ_da)
                error_dH_da[i, j] = np.linalg.norm(autograd_dH_da - num_dH_da)
        
        # Compute means.
        mean_analytical = {
            'J': np.mean(analytical_J, axis=0),
            'H': np.mean(analytical_H, axis=0),
            'dJ_da': np.mean(analytical_dJ_da, axis=0),
            'dH_da': np.mean(analytical_dH_da, axis=0),
        }
        
        mean_numerical = {
            'J': np.mean(numerical_J, axis=0),
            'H': np.mean(numerical_H, axis=0),
            'dJ_da': np.mean(numerical_dJ_da, axis=0),
            'dH_da': np.mean(numerical_dH_da, axis=0),
        }
                
        # Plotting
        plt.figure(figsize=(16, 12))

        # Jacobian Error
        plt.subplot(2, 2, 1)
        plt.contourf(X, Y, error_J, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title("L2 Error of Jacobian Estimates (J)")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.text(0.05, 0.9, 
                f"Analytical J: {mean_analytical['J']}\nNumerical J: {mean_numerical['J']}", 
                transform=plt.gca().transAxes, fontsize=9, color='white', 
                bbox=dict(facecolor='black', alpha=0.6))

        # Hessian Error
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, error_H, levels=50, cmap='magma')
        plt.colorbar()
        plt.title("L2 Error of Diagonal Hessian Estimates (H)")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.text(0.05, 0.9, 
                f"Analytical H: {mean_analytical['H']}\nNumerical H: {mean_numerical['H']}", 
                transform=plt.gca().transAxes, fontsize=9, color='white', 
                bbox=dict(facecolor='black', alpha=0.6))

        # dJ/da Error
        plt.subplot(2, 2, 3)
        plt.contourf(X, Y, error_dJ_da, levels=50, cmap='cividis')
        plt.colorbar()
        plt.title("L2 Error of dJ/da Estimates")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.text(0.05, 0.9, 
                f"Analytical dJ/da: {mean_analytical['dJ_da']}\nNumerical dJ/da: {mean_numerical['dJ_da']}", 
                transform=plt.gca().transAxes, fontsize=9, color='white', 
                bbox=dict(facecolor='black', alpha=0.6))

        # dH/da Error
        plt.subplot(2, 2, 4)
        plt.contourf(X, Y, error_dH_da, levels=50, cmap='plasma')
        plt.colorbar()
        plt.title("L2 Error of dH/da Estimates")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.text(0.05, 0.9, 
                f"Analytical dH/da: {mean_analytical['dH_da']}\nNumerical dH/da: {mean_numerical['dH_da']}", 
                transform=plt.gca().transAxes, fontsize=9, color='white', 
                bbox=dict(facecolor='black', alpha=0.6))

        plt.tight_layout()
        plt.show()

