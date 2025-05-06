"""
Basic example demonstrating TREBLE functionality.

This example:
1. Generates synthetic correlated random walk data
2. Runs the iterative window search process
3. Analyzes the metrics to find optimal window size
4. Creates a behavioral embedding using UMAP
5. Visualizes the resulting behavior space

Note: This uses synthetically generated data. For real analysis,
you would load your own tracked animal trajectory data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import treble
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treble import (
    get_velocity_windows,
    iterative_umap,
    bin_umap,
    run_procrustes,
    calculate_recurrence,
    plot_results,
    plot_variance,
    plot_recurrence,
    plot_vector_field,
    plot_umap_features,
    plot_umap_pdf
)


def generate_correlated_random_walk(
    n_steps: int = 5000,
    correlation: float = 0.8,
    turn_bias: float = 0.0,
    speed_mean: float = 1.0,
    speed_std: float = 0.2,
    angular_std: float = 0.5
) -> pd.DataFrame:
    """Generate a correlated random walk trajectory."""
    # Initialize arrays
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    
    # Generate correlated turning angles
    turn_angles = np.random.normal(turn_bias, angular_std, n_steps)
    for i in range(1, n_steps):
        turn_angles[i] = correlation * turn_angles[i-1] + (1 - correlation) * turn_angles[i]
    
    # Generate speeds
    speeds = np.random.normal(speed_mean, speed_std, n_steps)
    speeds = np.maximum(0.1, speeds)  # Ensure minimum speed
    
    # Initialize direction
    theta[0] = np.random.uniform(0, 2 * np.pi)
    
    # Generate trajectory
    for i in range(1, n_steps):
        theta[i] = theta[i-1] + turn_angles[i]
        x[i] = x[i-1] + speeds[i] * np.cos(theta[i])
        y[i] = y[i-1] + speeds[i] * np.sin(theta[i])
    
    # Calculate velocities
    translational_velocity = np.concatenate([[0], np.sqrt(np.diff(x)**2 + np.diff(y)**2)])
    angular_velocity = np.concatenate([[0], np.diff(theta)])
    
    # Normalize angular velocity to [-pi, pi]
    angular_velocity = np.mod(angular_velocity + np.pi, 2 * np.pi) - np.pi
    
    # Create dataframe
    df = pd.DataFrame({
        'time': np.arange(n_steps),
        'x': x,
        'y': y,
        'theta': theta,
        'translational_velocity': translational_velocity,
        'angular_velocity': angular_velocity
    })
    
    return df


def main():
    # Generate synthetic data (10 correlated random walks)
    print("Generating synthetic data...")
    n_trajectories = 10
    velocity_data = [
        generate_correlated_random_walk(
            n_steps=5000, 
            correlation=0.8,
            turn_bias=np.random.uniform(-0.1, 0.1),
            speed_mean=np.random.uniform(0.8, 1.2),
            speed_std=0.2,
            angular_std=np.random.uniform(0.4, 0.6)
        )
        for _ in range(n_trajectories)
    ]
    
    # Plot example trajectory
    example_traj = velocity_data[0]
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(example_traj['x'], example_traj['y'])
    plt.title('Example Trajectory')
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(example_traj['time'], example_traj['translational_velocity'], label='Speed')
    plt.plot(example_traj['time'], example_traj['angular_velocity'], label='Angular velocity')
    plt.title('Velocity Components')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Run iterative window search
    print("Running iterative window search...")
    window_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    iterative_results = {}
    
    # Use a smaller subset for faster computation
    data_subset = [df.iloc[:1000].copy() for df in velocity_data]
    
    for size in window_sizes:
        print(f"Testing window size {size}...")
        iterative_results[str(size)] = iterative_umap(
            data_subset,
            velocity_windows=True,
            window_size=size,
            plot=False,  # Set to True to see UMAP plots for each window size
            n_neighbors=15,
            min_dist=0.1
        )
    
    # Calculate Procrustes and Euclidean distances
    print("Calculating distance metrics...")
    procrustes_distances = {}
    euclidean_distances = {}
    
    for size in window_sizes:
        size_str = str(size)
        distances = run_procrustes(iterative_results[size_str]["umaps"])
        procrustes_distances[size_str] = distances["procrustes"]
        euclidean_distances[size_str] = distances["euclidean_distances"]
    
    # Plot distance metrics
    print("Plotting distance metrics...")
    plot_results(
        procrustes_distances,
        ylim=(0, 8),
        ylab="Procrustes RMSD",
        xlab="Window size (frames)",
        title="Procrustes Distance by Window Size"
    )
    
    plot_results(
        euclidean_distances,
        ylim=(0, 500),
        ylab="Mean Euclidean distance",
        xlab="Window size (frames)",
        title="Euclidean Distance by Window Size"
    )
    
    # Plot distance metrics as variance
    plot_variance(
        procrustes_distances,
        ylim=(0, 0.2),
        ylab="Procrustes distance (coef. var.)",
        xlab="Window size (frames)",
        title="Procrustes Distance Coefficient of Variation"
    )
    
    plot_variance(
        euclidean_distances,
        ylim=(0, 0.2),
        ylab="Euclidean distance (coef. var.)",
        xlab="Window size (frames)",
        title="Euclidean Distance Coefficient of Variation"
    )
    
    # Calculate recurrence
    print("Calculating recurrence metrics...")
    recurrence_results = {}
    
    for size in window_sizes:
        size_str = str(size)
        print(f"Processing window size {size}...")
        recurrence_results[size_str] = calculate_recurrence(
            iterative_results[size_str]["umaps"],
            n_bins=16
        )
    
    # Plot recurrence
    print("Plotting recurrence distributions...")
    plot_recurrence(recurrence_results)
    
    # Calculate mean recurrence times
    mean_recurrence = {}
    for size in window_sizes:
        size_str = str(size)
        means = []
        for rep_result in recurrence_results[size_str]:
            if rep_result["recurrences"]:  # Check if there are any recurrences
                means.append(np.mean(rep_result["recurrences"]))
            else:
                means.append(np.nan)
        mean_recurrence[size_str] = [m for m in means if not np.isnan(m)]
    
    # Plot mean recurrence times
    plot_results(
        mean_recurrence,
        ylim=(0, 200),
        ylab="Mean recurrence time",
        xlab="Window size (frames)",
        title="Mean Recurrence Time by Window Size"
    )
    
    plot_variance(
        mean_recurrence,
        ylim=(0, 1),
        ylab="Coefficient of variation",
        xlab="Window size (frames)",
        title="Recurrence Time Coefficient of Variation"
    )
    
    # Based on the metrics, choose an optimal window size (e.g., 15)
    optimal_window_size = 15
    print(f"Selected optimal window size: {optimal_window_size}")
    
    # Create behavior space with optimal window size
    print("Creating behavior space with optimal window size...")
    windows = []
    
    for i, vel in enumerate(velocity_data):
        windows.append(
            get_velocity_windows(
                vel, 
                window_size=optimal_window_size, 
                name=f'replicate_{i}'
            )
        )
    
    # Combine windows into a single dataframe
    combined_windows = pd.concat(windows, axis=1)
    
    # Run UMAP on combined windows
    print("Running UMAP on combined windows...")
    from umap import UMAP
    
    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    
    # Transpose for UMAP input format
    umap_result = umap_model.fit_transform(combined_windows.T.values)
    
    # Create dataframe with UMAP coordinates
    layout = pd.DataFrame({
        'x': umap_result[:, 0],
        'y': umap_result[:, 1]
    })
    
    # Add metadata
    layout['individual'] = [f"replicate_{int(i/len(windows[0].columns))}" 
                           for i in range(len(layout))]
    
    # Bin the UMAP space
    binned_layout, _ = bin_umap(layout, n_bins=32)
    
    # Visualize the behavior space
    print("Visualizing behavior space...")
    
    # Plot as points and lines
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(layout['x'], layout['y'], s=5, alpha=0.5, c='gray')
    plt.title('UMAP Embedding (Points)')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.plot(layout['x'], layout['y'], linewidth=0.5, alpha=0.5, c='gray')
    plt.title('UMAP Embedding (Trajectory)')
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Plot as vector field
    print("Generating vector field visualization...")
    plot_vector_field(layout)
    plot_vector_field(layout, color_by_theta=True)
    
    # Bin and plot as vector field
    plot_vector_field(
        layout,
        bin_umap=True,
        n_bins=100
    )
    plot_vector_field(
        layout, 
        color_by_theta=True,
        bin_umap=True,
        n_bins=100
    )
    
    # Plot with features colored
    print("Generating feature visualization...")
    plot_umap_features(
        layout, 
        combined_windows,
        feature_names=['Translational velocity', 'Angular velocity'],
        colors=['darkgreen', 'darkmagenta'],
        n_features=2
    )
    
    # Plot as points
    plot_umap_features(
        layout, 
        combined_windows,
        feature_names=['Translational velocity', 'Angular velocity'],
        colors=['darkgreen', 'darkmagenta'],
        n_features=2,
        plot_points=True
    )
    
    # Plot as probability density function
    print("Generating density visualization...")
    plot_umap_pdf(layout, h=2)
    
    # Plot individual PDFs
    individuals = layout['individual'].unique()
    
    plt.figure(figsize=(15, 10))
    for i, ind in enumerate(individuals):
        plt.subplot(2, 5, i + 1)
        ind_layout = layout[layout['individual'] == ind]
        plot_umap_pdf(ind_layout, h=2)
        plt.title(ind)
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()