"""
Visualize the loss landscape for SPICE-SINDy training.
Uses the loss-landscapes library to create 2D visualizations of the loss surface.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import loss_landscapes
import loss_landscapes.metrics

from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset, split_data_along_timedim
from spice.resources.spice_utils import SpiceDataset
from spice.precoded import BufferWorkingMemoryRNN, BUFFER_WORKING_MEMORY_CONFIG


def compute_loss_for_landscape(model, inputs, targets, sindy_weight=0.1):
    """
    Compute total loss (cross-entropy + SINDy regularization) for loss landscape.

    Args:
        model: The RNN model
        inputs: Input data [batch, time, features]
        targets: Target labels [batch, time, n_actions]
        sindy_weight: Weight for SINDy regularization

    Returns:
        Total loss value
    """
    model.eval()

    with torch.no_grad():
        xs = inputs.to(model.device)
        ys = targets.to(model.device)

        # Initialize model state
        model.set_initial_state(batch_size=len(xs))

        # Forward pass
        mask = xs[..., :1] > -1
        model_output = model(xs, model.get_state(), batch_first=True)
        ys_pred = model_output[0]
        sindy_loss = model_output[2] if len(model_output) > 2 else 0

        ys_pred = ys_pred * mask
        ys_targets = ys * mask

        # Cross-entropy loss
        loss_ce = torch.nn.functional.cross_entropy(
            ys_pred.reshape(-1, model._n_actions),
            torch.argmax(ys_targets.reshape(-1, model._n_actions), dim=1),
            reduction='mean'
        )

        # Add SINDy regularization
        if sindy_weight > 0 and isinstance(sindy_loss, torch.Tensor):
            sindy_loss_masked = sindy_loss * mask[..., 0]
            mask_nonzero = (sindy_loss_masked != 0).float()
            num_nonzero = mask_nonzero.sum()
            sindy_loss_mean = (sindy_loss_masked * mask_nonzero).sum() / (num_nonzero + 1e-8)
            total_loss = loss_ce + sindy_weight * sindy_loss_mean
        else:
            total_loss = loss_ce

    return total_loss.item()


class LossLandscapeWrapper(torch.nn.Module):
    """Wrapper to make SPICE model compatible with loss-landscapes library."""

    def __init__(self, model, inputs, targets, sindy_weight):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.sindy_weight = sindy_weight

    def forward(self, x=None, *args, **kwargs):
        """Compute loss (required by loss-landscapes)."""
        # Ignore x, args, kwargs - we use stored inputs/targets
        return compute_loss_for_landscape(
            self.model, self.inputs, self.targets, self.sindy_weight
        )

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()


def visualize_loss_landscape_2d(
    model,
    dataset,
    sindy_weight=0.1,
    distance=0.5,
    steps=40,
    save_path="loss_landscape.png"
):
    """
    Create a 2D loss landscape visualization using random directions.

    Args:
        model: Trained SPICE model
        dataset: SpiceDataset with inputs and targets
        sindy_weight: Weight for SINDy regularization
        distance: How far to move in each direction
        steps: Number of steps in each direction
        save_path: Path to save the figure
    """
    print("Creating loss landscape visualization...")
    print(f"Distance: {distance}, Steps: {steps}")

    # Move model to CPU to avoid device mismatch issues with loss-landscapes
    device_original = model.device
    model.cpu()
    model.device = torch.device('cpu')  # Update device attribute

    # Prepare data on CPU
    inputs = dataset.xs[:32].cpu()  # Use subset for faster computation
    targets = dataset.ys[:32].cpu()

    # Create wrapper
    wrapper = LossLandscapeWrapper(model, inputs, targets, sindy_weight)

    # Define metric (simple callable that returns loss)
    def metric_fn(model_wrapper):
        return model_wrapper.forward(inputs)

    # Generate random directions for perturbation
    print("Generating random directions...")

    # Compute loss landscape
    print("Computing loss landscape (this may take a few minutes)...")
    loss_data_fin = loss_landscapes.random_plane(
        wrapper,
        metric_fn,
        distance=distance,
        steps=steps,
        normalization='filter',
        deepcopy_model=True
    )

    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(12, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(121)
    X = np.linspace(-distance, distance, steps)
    Y = np.linspace(-distance, distance, steps)
    X, Y = np.meshgrid(X, Y)

    levels = np.linspace(loss_data_fin.min(), loss_data_fin.max(), 20)
    contour = ax1.contourf(X, Y, loss_data_fin, levels=levels, cmap='viridis')
    ax1.contour(X, Y, loss_data_fin, levels=levels, colors='black', alpha=0.2, linewidths=0.5)

    # Mark the center (current model position)
    ax1.plot(0, 0, 'r*', markersize=20, label='Current Model')

    ax1.set_xlabel('Direction 1', fontsize=12)
    ax1.set_ylabel('Direction 2', fontsize=12)
    ax1.set_title('Loss Landscape (2D Contour)', fontsize=14)
    ax1.legend()
    plt.colorbar(contour, ax=ax1, label='Loss')

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, loss_data_fin, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax2.plot([0], [0], [loss_data_fin[steps//2, steps//2]], 'r*', markersize=15)

    ax2.set_xlabel('Direction 1', fontsize=10)
    ax2.set_ylabel('Direction 2', fontsize=10)
    ax2.set_zlabel('Loss', fontsize=10)
    ax2.set_title('Loss Landscape (3D Surface)', fontsize=14)
    ax2.view_init(elev=30, azim=45)
    plt.colorbar(surf, ax=ax2, shrink=0.5, label='Loss')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss landscape to {save_path}")

    # Restore original device
    model.to(device_original)
    model.device = device_original

    return fig, loss_data_fin


def visualize_loss_landscape_1d(
    model,
    dataset,
    sindy_weight=0.1,
    distance=1.0,
    steps=50,
    save_path="loss_landscape_1d.png"
):
    """
    Create 1D loss landscape along a random direction.

    Args:
        model: Trained SPICE model
        dataset: SpiceDataset with inputs and targets
        sindy_weight: Weight for SINDy regularization
        distance: How far to move in the direction
        steps: Number of steps
        save_path: Path to save the figure
    """
    print("\nCreating 1D loss landscape visualization...")

    # Move model to CPU to avoid device mismatch issues
    device_original = model.device
    model.cpu()
    model.device = torch.device('cpu')  # Update device attribute

    # Prepare data on CPU
    inputs = dataset.xs[:32].cpu()
    targets = dataset.ys[:32].cpu()

    # Create wrapper
    wrapper = LossLandscapeWrapper(model, inputs, targets, sindy_weight)

    # Define metric (simple callable that returns loss)
    def metric_fn(model_wrapper):
        return model_wrapper.forward(inputs)

    # Compute 1D loss landscape
    print("Computing 1D loss landscape...")
    loss_data = loss_landscapes.linear_interpolation(
        wrapper,
        wrapper,  # Use same model as start and end (centered at current position)
        metric_fn,
        steps=steps,
        deepcopy_model=True
    )

    # Adjust to center at 0
    x_coords = np.linspace(-distance, distance, steps)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_coords, loss_data, linewidth=2, color='blue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Current Model')
    ax.scatter([0], [loss_data[steps//2]], color='red', s=100, zorder=5)

    ax.set_xlabel('Distance along random direction', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('1D Loss Landscape', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 1D loss landscape to {save_path}")

    # Restore original device
    model.to(device_original)
    model.device = device_original

    return fig, loss_data


if __name__ == '__main__':
    # Configuration
    data_path = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    model_path = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"

    print("Loading dataset...")
    dataset = convert_dataset(file=data_path)[0]
    dataset_train, dataset_test = split_data_along_timedim(dataset, 0.8)

    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())

    print(f"Dataset: {n_participants} participants, {n_actions} actions")

    # Initialize estimator and load trained model
    print("\nInitializing SPICE estimator...")
    estimator = SpiceEstimator(
        rnn_class=BufferWorkingMemoryRNN,
        spice_config=BUFFER_WORKING_MEMORY_CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        sindy_weight=10,
        spice_library_polynomial_degree=2,
    )

    print(f"Loading model from {model_path}...")
    estimator.load_spice(model_path)

    # Create loss landscape visualizations
    sindy_weight = 10

    # 2D landscape
    fig_2d, loss_data_2d = visualize_loss_landscape_2d(
        estimator.rnn_model,
        dataset_train,
        sindy_weight=sindy_weight,
        distance=0.5,
        steps=20,  # Reduced for faster computation (20x20=400 evaluations)
        save_path="weinhardt2025/analysis/loss_landscape_2d.png"
    )

    # 1D landscape
    fig_1d, loss_data_1d = visualize_loss_landscape_1d(
        estimator.rnn_model,
        dataset_train,
        sindy_weight=sindy_weight,
        distance=1.0,
        steps=30,  # Reduced for faster computation
        save_path="weinhardt2025/analysis/loss_landscape_1d.png"
    )

    print("\nDone! Loss landscape visualizations created.")
    print("Statistics:")
    print(f"  2D landscape - Min loss: {loss_data_2d.min():.6f}, Max loss: {loss_data_2d.max():.6f}")
    print(f"  1D landscape - Min loss: {loss_data_1d.min():.6f}, Max loss: {loss_data_1d.max():.6f}")

    # Show plots
    plt.show()
