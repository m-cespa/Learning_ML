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


def generate_trig_data(total_samples: int, batch_dim: int, input_size: int,
                         func: Callable[[np.ndarray], np.ndarray] = None
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    num_batches = total_samples // batch_dim  # e.g. 200/5 = 40 mini-batches.
    # Manually defined training domain: [0, 2*pi] for each input dimension.
    domain = [(0, 2 * np.pi) for _ in range(input_size)]
    
    # Set default function if none provided.
    if func is None:
        if input_size == 1:
            func = lambda x: np.exp(-x[0])*np.sin(3*x[0])
        elif input_size == 2:
            func = lambda x: 6*np.sin(np.pi*x[0]/5.)*np.exp(-x[1]*(np.pi/5.)**2)
        else:
            raise ValueError("No default function defined for input_size > 2. Please provide a function.")
    
    batches = []
    for _ in range(num_batches):
        # Create empty tensors for the batch.
        input_tensor = np.zeros((batch_dim, input_size))
        label_tensor = np.zeros((batch_dim, 1))  # 1D output per sample.
        for i in range(batch_dim):
            # Generate one sample by drawing uniformly from each domain interval.
            x_i = np.array([np.random.uniform(low, high) for (low, high) in domain])
            y_i = func(x_i)  # Compute label from provided function.
            input_tensor[i] = x_i
            label_tensor[i] = y_i  # Even if y_i is scalar, shape (1,) becomes (1,).
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
    batch_dim = 8
    input_size = 2
    output_size = 1

    # Define the layers with batch dimension support
    input_layer = Layer(size=input_size)
    input_activation = ActivationLayer(Linear())
    hidden_layer_1 = Layer(size=4)
    activation1 = ActivationLayer(ELU())
    hidden_layer_2 = Layer(size=16)
    activation2 = ActivationLayer(ELU())
    hidden_layer_3 = Layer(size=32)
    activation3 = ActivationLayer(ELU())
    hidden_layer_4 = Layer(size=16)
    activation4 = ActivationLayer(ELU())
    hidden_layer_5 = Layer(size=8)
    activation5 = ActivationLayer(ELU())
    output_layer = Layer(size=output_size)
    output_activation = ActivationLayer(Linear())

    # Link layers sequentially
    network = Network(layers=[input_layer,
                       input_activation,
                       hidden_layer_1,
                       activation1,
                       hidden_layer_2,
                       activation2,
                       hidden_layer_3,
                       activation3,
                       hidden_layer_4,
                       activation4,
                       hidden_layer_5,
                       activation5,
                       output_layer,
                       output_activation],
                       physics_loss_weight=0.001,
                       optim=Adam())

    # Generate XOR training data with batch and channel dimensions
    learn_data = generate_heat_eq_data(
        total_samples=500,
        batch_dim=batch_dim,
        k=1.,
        L=5.
    )

    collocation_data = generate_collocation_data(N=200, k=1., L=5.)
    boundary_data = generate_boundary_data(N=200, k=1., L=5.)

    # Print shapes of the first label and input tensors to verify correctness
    print(f"Shape of label tensor: {learn_data[0][1].shape}")  # Should be (batch_dim, output_size)
    print(f"Shape of input tensor: {learn_data[0][0].shape}")  # Should be (batch_dim, input_size)
    
    network.learn(learn_data=learn_data, lr=0.001, epochs=20, loss_func='mse', 
                  collocation_data=collocation_data, boundary_data=boundary_data, plot=True, store_grads=True)
    
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
    times_outside = np.linspace(T+5, T+5, 1)
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

    # # Create a grid of points in the domain [0, 2π] x [0, 2π]
    # num_points = 100
    # x1 = np.linspace(0, 2*np.pi, num_points)
    # x2 = np.linspace(0, 2*np.pi, num_points)
    # X1, X2 = np.meshgrid(x1, x2)

    # # Reshape grid points into a (N, 2) array for network input
    # grid_points = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])

    # # Get network predictions, assume network.forward returns predictions of shape (N, 1)
    # predictions = network.forward(grid_points)   # predicted shape (N, 1)
    # predictions = predictions.reshape(X1.shape)   # reshape to (num_points, num_points)

    # # Ground truth: sin(x1)*cos(x2)
    # ground_truth = np.sin(X1) * np.cos(X2)

    # # Compute absolute error
    # error = np.abs(predictions - ground_truth)

    # # Plot the ground truth, network prediction, and absolute error in a three-panel figure.
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # # Ground Truth Plot
    # im0 = axes[0].imshow(ground_truth, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='viridis')
    # axes[0].set_title(r"Ground Truth: $u(x_1, x_2) = \sin(x1)\cos(x2)$")
    # axes[0].set_xlabel(r"$x_1$")
    # axes[0].set_ylabel(r"$x_2$")
    # fig.colorbar(im0, ax=axes[0])

    # # Network Prediction Plot
    # im1 = axes[1].imshow(predictions, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='viridis')
    # axes[1].set_title(r"Model: $\hat{u}(x_1, x_2)$")
    # axes[1].set_xlabel(r"$x_1$")
    # axes[1].set_ylabel(r"$x_2$")
    # fig.colorbar(im1, ax=axes[1])

    # # Error Plot
    # im2 = axes[2].imshow(error, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='magma')
    # axes[2].set_title(r"$|\hat{u}(x_1, x_2) - u(x_1, x_2)|$")
    # axes[2].set_xlabel(r"$x_1$")
    # axes[2].set_ylabel(r"$x_2$")
    # fig.colorbar(im2, ax=axes[2])

    # plt.tight_layout()
    # plt.savefig('sin_cos_heatmap.png', dpi=300)
    # plt.show()


    # for layer in network.layers:
    #     if isinstance(layer, Layer):
    #         if hasattr(layer, 'g'):
    #             print(f"{layer} has g: {layer.g.shape}")

    # print(network.layers[0].z.shape)
    # grad_test_data = np.tile([[0., 1.]], (1, 1))
    # print(network.forward(grad_test_data, store_grads=True))


    # numerical_J, numerical_H = network.numerical_jacobian_hessian(X=grad_test_data)
    # print(network.dJ_da)
    # print(network.dH_da)


    # def compute_errors_over_domain(net, N: int = 50, h: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     # Create a uniform grid in x and y over [0, 2π]
    #     x_vals = np.linspace(0, 2*np.pi, N)
    #     y_vals = np.linspace(0, 2*np.pi, N)
    #     X, Y = np.meshgrid(x_vals, y_vals)
        
    #     error_J = np.zeros((N, N))
    #     error_H = np.zeros((N, N))
        
    #     # Loop over each grid point. Our network expects inputs of shape (1,2).
    #     for i in range(N):
    #         for j in range(N):
    #             inp = np.array([[X[i, j], Y[i, j]]], dtype=np.float64)   # shape (1,2)
                
    #             # Run forward pass to set network internal states.
    #             _ = net.forward(inp, store_grads=True)
    #             # Compute autograd estimates (which update network.J_batch and network.H_batch)
    #             net.autograd()      # computes self.J_batch (e.g., shape (1, 1, 2))
    #             # (Optionally, one might call autograd_derivs() for sensitivity but here we compare J and H directly.)
                
    #             # Squeeze autograd results to shape (2,)
    #             autograd_J = np.squeeze(net.J_batch)  # expected to be (1,2) --> (2,)
    #             autograd_H = np.squeeze(net.H_batch)  # expected to be (1,2) --> (2,)
                
    #             # Compute numerical estimates on the same point.
    #             numJ, numH = net.numerical_jacobian_hessian(inp, h=h)
    #             numJ = np.squeeze(numJ)  # shape (2,)
    #             numH = np.squeeze(numH)  # shape (2,)
                
    #             # L2 norm of the difference (for each, over the two input dimensions)
    #             error_J[i, j] = np.linalg.norm(autograd_J - numJ)
    #             error_H[i, j] = np.linalg.norm(autograd_H - numH)
                
    #     return X, Y, error_J, error_H

    # # Example usage:
    # # Assume 'network' is already instantiated and trained (or at least set up) to approximate u(x,y) = sin(x)cos(y).
    # X, Y, error_J, error_H = compute_errors_over_domain(network, N=50, h=1e-5)

    # # Plot the L2 error for the Jacobian and Hessian.
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.contourf(X, Y, error_J, levels=50, cmap='viridis')
    # plt.colorbar()
    # plt.title(r"L2 Error of Jacobian Estimates")
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")

    # plt.subplot(1, 2, 2)
    # plt.contourf(X, Y, error_H, levels=50, cmap='magma')
    # plt.colorbar()
    # plt.title(r"L2 Error of Diagonal Hessian Estimates")
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")

    # plt.tight_layout()
    # plt.savefig('numerical_vs_autograd.png', dpi=300)
    # plt.show()
