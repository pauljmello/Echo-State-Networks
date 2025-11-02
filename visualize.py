from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_results_dir():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_prediction_vs_target(pred, target, error, cfg, results_dir):
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    n_plot = min(500, len(pred_np))
    time = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time, target_np[:n_plot], 'b-', label='Target', alpha=0.8, linewidth=1.5)
    ax.plot(time, pred_np[:n_plot], 'r--', label='Prediction', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'ESN Prediction vs Target (NRMSE={error:.6f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(results_dir / 'prediction_vs_target.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'prediction_vs_target.png'}")


def save_prediction_error(pred, target, results_dir):
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    errors = pred_np - target_np

    n_plot = min(500, len(errors))
    time = np.arange(n_plot)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Error over time
    ax1.plot(time, errors[:n_plot], 'purple', alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(time, 0, errors[:n_plot], alpha=0.3, color='purple')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Prediction Error', fontsize=12)
    ax1.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    abs_errors = np.abs(errors[:n_plot])
    ax2.plot(time, abs_errors, 'orange', alpha=0.7, linewidth=1)
    ax2.fill_between(time, 0, abs_errors, alpha=0.3, color='orange')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Absolute Prediction Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    mae = np.mean(abs_errors)
    ax2.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'Mean Absolute Error: {mae:.4f}')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / 'prediction_error.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'prediction_error.png'}")


def save_scatter_plot(pred, target, results_dir):
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    if torch.std(pred) > 1e-6 and torch.std(target) > 1e-6:
        correlation = torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1].item()
    else:
        correlation = 0.0

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(target_np, pred_np, alpha=0.4, s=20, c='steelblue', edgecolors='none')

    lim = [min(target_np.min(), pred_np.min()), max(target_np.max(), pred_np.max())]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect Prediction', alpha=0.8)

    z = np.polyfit(target_np, pred_np, 1)
    p = np.poly1d(z)
    ax.plot(lim, p(lim), 'g-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

    ax.set_xlabel('Target Value', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'Prediction vs Target Scatter (r={correlation:.4f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # R^2
    r_squared = correlation ** 2
    ax.text(0.05, 0.95, f'R^2 = {r_squared:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(results_dir / 'scatter_plot.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'scatter_plot.png'}")


def save_reservoir_states(esn, test_in, results_dir):
    states, _ = esn.run(test_in[:500], washout=0)
    states_np = states.cpu().numpy().T

    n_neurons_plot = min(50, esn.n_res)
    indices = np.linspace(0, esn.n_res - 1, n_neurons_plot, dtype=int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Heatmap of neuron activations
    im1 = ax1.imshow(states_np[indices, :], aspect='auto', cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Neuron Index', fontsize=12)
    ax1.set_title('Reservoir State Activations Over Time', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Activation')

    # First 10 individual neuron trajectories
    time = np.arange(states_np.shape[1])
    for i in range(min(10, n_neurons_plot)):
        ax2.plot(time, states_np[indices[i], :], alpha=0.6, linewidth=1, label=f'Neuron {indices[i]}')

    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Activation', fontsize=12)
    ax2.set_title('Individual Neuron Trajectories', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(results_dir / 'reservoir_states.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'reservoir_states.png'}")


def save_reservoir_weights(esn, cfg, results_dir):
    W = esn.W.cpu().numpy()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # Sparsity visualization of full weight matrix
    n_show = min(200, esn.n_res)
    W_show = W[:n_show, :n_show]

    im1 = ax1.imshow(W_show, aspect='auto', cmap='RdBu_r', vmin=-cfg['rho'], vmax=cfg['rho'], interpolation='nearest')
    ax1.set_xlabel('Neuron Index', fontsize=11)
    ax1.set_ylabel('Neuron Index', fontsize=11)
    ax1.set_title(f'Reservoir Weight Matrix (First {n_show} x {n_show})', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Weight Value')

    # binary sparsity pattern
    sparsity_pattern = (np.abs(W_show) > 1e-10).astype(float)
    ax2.imshow(sparsity_pattern, aspect='auto', cmap='binary', interpolation='nearest')
    ax2.set_xlabel('Neuron Index', fontsize=11)
    ax2.set_ylabel('Neuron Index', fontsize=11)
    density = np.sum(sparsity_pattern) / sparsity_pattern.size
    ax2.set_title(f'Sparsity Pattern (Density={density:.3f})', fontsize=12, fontweight='bold')

    # Weight distribution
    weights_nonzero = W[W != 0].flatten()
    ax3.hist(weights_nonzero, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Weight Value', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Non-Zero Weight Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Spectral properties
    eigenvalues = np.linalg.eigvals(W)
    ax4.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.5, s=20, c='steelblue')

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = cfg['rho'] * np.cos(theta)
    circle_y = cfg['rho'] * np.sin(theta)
    ax4.plot(circle_x, circle_y, 'r--', linewidth=2, alpha=0.7, label=f'Target ρ={cfg["rho"]}')

    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.set_xlabel('Real Part', fontsize=11)
    ax4.set_ylabel('Imaginary Part', fontsize=11)
    ax4.set_title('Eigenvalue Spectrum', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')

    # actual spectral radius
    actual_rho = np.max(np.abs(eigenvalues))
    ax4.text(0.05, 0.95, f'Actual ρ={actual_rho:.4f}', transform=ax4.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(results_dir / 'reservoir_weights.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'reservoir_weights.png'}")


def generate_all_visualizations(esn, pred, target, error, test_in, cfg):
    results_dir = create_results_dir()
    print(f"\nSaving visualizations to: {results_dir.absolute()}\n")

    save_prediction_vs_target(pred, target, error, cfg, results_dir)
    save_prediction_error(pred, target, results_dir)
    save_scatter_plot(pred, target, results_dir)
    save_reservoir_states(esn, test_in, results_dir)
    save_reservoir_weights(esn, cfg, results_dir)

    print(f"\nVisualizations saved to: {results_dir.absolute()}")
