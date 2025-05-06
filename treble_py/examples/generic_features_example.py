"""
Example demonstrating TREBLE with generic features.

This example:
1. Generates synthetic data with multiple features
2. Runs the iterative window search using the generic feature interface
3. Analyzes the metrics to find optimal window size
4. Creates a behavioral embedding using UMAP
5. Visualizes the resulting behavior space

This shows how TREBLE can be used with any kind of sequential feature data,
not just velocity components.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import treble
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treble import (
    get_feature_windows,
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


def generate_oscillation_data(
    n_steps: int = 5000,
    n_oscillators: int = 3,
    baseline_frequencies: Optional[List[float]] = None,
    frequency_drift: float = 0.01,
    amplitude_drift: float = 0.02,
    noise_level: float = 0.1,
    correlation: float = 0.3
) -> pd.DataFrame:
    """
    Generate synthetic oscillation data with multiple coupled oscillators.
    
    Args:
        n_steps: Number of time steps
        n_oscillators: Number of oscillators
        baseline_frequencies: Base frequencies for each oscillator (if None, random values used)
        frequency_drift: Amount of random drift in frequencies
        amplitude_drift: Amount of random drift in amplitudes
        noise_level: Level of noise to add
        correlation: Correlation between oscillators
        
    Returns:
        DataFrame with time and oscillator values
    """
    # Initialize time
    time = np.arange(n_steps)
    
    # Generate random frequencies if not provided
    if baseline_frequencies is None:
        baseline_frequencies = np.random.uniform(0.01, 0.1, n_oscillators)
    
    # Initialize array for oscillator values
    data = np.zeros((n_steps, n_oscillators))
    
    # Generate oscillations with drift and correlation
    frequencies = np.zeros((n_steps, n_oscillators))
    amplitudes = np.zeros((n_steps, n_oscillators))
    phases = np.zeros((n_steps, n_oscillators))
    
    # Initialize
    frequencies[0] = baseline_frequencies
    amplitudes[0] = np.ones(n_oscillators)
    phases[0] = np.random.uniform(0, 2*np.pi, n_oscillators)
    
    # Generate first oscillator
    for t in range(1, n_steps):
        # Update frequency with drift
        frequencies[t, 0] = frequencies[t-1, 0] + np.random.normal(0, frequency_drift)
        # Ensure frequency stays positive and reasonable
        frequencies[t, 0] = max(0.001, min(0.2, frequencies[t, 0]))
        
        # Update amplitude with drift
        amplitudes[t, 0] = amplitudes[t-1, 0] + np.random.normal(0, amplitude_drift)
        # Ensure amplitude stays positive and reasonable
        amplitudes[t, 0] = max(0.2, min(2.0, amplitudes[t, 0]))
        
        # Update phase
        phases[t, 0] = phases[t-1, 0] + 2 * np.pi * frequencies[t, 0]
        
        # Calculate oscillator value
        data[t, 0] = amplitudes[t, 0] * np.sin(phases[t, 0])
    
    # Generate other oscillators with correlation to the first
    for i in range(1, n_oscillators):
        for t in range(1, n_steps):
            # Frequency influenced by previous oscillator
            frequencies[t, i] = (
                baseline_frequencies[i] * (1 - correlation) + 
                frequencies[t, i-1] * correlation + 
                np.random.normal(0, frequency_drift)
            )
            frequencies[t, i] = max(0.001, min(0.2, frequencies[t, i]))
            
            # Amplitude influenced by previous oscillator
            amplitudes[t, i] = (
                amplitudes[t-1, i] * (1 - correlation) + 
                amplitudes[t, i-1] * correlation + 
                np.random.normal(0, amplitude_drift)
            )
            amplitudes[t, i] = max(0.2, min(2.0, amplitudes[t, i]))
            
            # Update phase
            phases[t, i] = phases[t-1, i] + 2 * np.pi * frequencies[t, i]
            
            # Calculate oscillator value
            data[t, i] = amplitudes[t, i] * np.sin(phases[t, i] + i * np.pi / n_oscillators)
    
    # Add noise
    data += np.random.normal(0, noise_level, data.shape)
    
    # Create 2D coordinates based on first two oscillators
    x = np.cumsum(data[:, 0])
    y = np.cumsum(data[:, 1])
    
    # Create DataFrame
    columns = ['time', 'x', 'y'] + [f'oscillator_{i+1}' for i in range(n_oscillators)]
    df = pd.DataFrame(
        np.column_stack([time, x, y, data]),
        columns=columns
    )
    
    return df


def generate_eeg_like_data(
    n_steps: int = 5000,
    n_channels: int = 8,
    sampling_rate: float = 100.0,
    alpha_power: float = 1.0,
    beta_power: float = 0.5,
    delta_power: float = 2.0,
    theta_power: float = 1.5,
    gamma_power: float = 0.3,
    noise_level: float = 0.5,
    correlation: float = 0.4
) -> pd.DataFrame:
    """
    Generate synthetic EEG-like data with multiple frequency bands and correlations.
    
    Args:
        n_steps: Number of time steps
        n_channels: Number of EEG channels
        sampling_rate: Sample rate in Hz
        alpha_power: Power of alpha waves (8-12 Hz)
        beta_power: Power of beta waves (12-30 Hz)
        delta_power: Power of delta waves (1-4 Hz)
        theta_power: Power of theta waves (4-8 Hz)
        gamma_power: Power of gamma waves (30-100 Hz)
        noise_level: Level of noise to add
        correlation: Correlation between channels
        
    Returns:
        DataFrame with time and EEG channel values
    """
    # Initialize time
    time = np.arange(n_steps) / sampling_rate
    
    # Define frequency bands
    delta_freq = np.random.uniform(1, 4)
    theta_freq = np.random.uniform(4, 8)
    alpha_freq = np.random.uniform(8, 12)
    beta_freq = np.random.uniform(12, 30)
    gamma_freq = np.random.uniform(30, 50)  # limiting to 50 Hz for simplicity
    
    # Create frequency components
    delta_wave = delta_power * np.sin(2 * np.pi * delta_freq * time)
    theta_wave = theta_power * np.sin(2 * np.pi * theta_freq * time)
    alpha_wave = alpha_power * np.sin(2 * np.pi * alpha_freq * time)
    beta_wave = beta_power * np.sin(2 * np.pi * beta_freq * time)
    gamma_wave = gamma_power * np.sin(2 * np.pi * gamma_freq * time)
    
    # Initialize array for channel values
    data = np.zeros((n_steps, n_channels))
    
    # Generate base signal (same for all channels, but with phase shifts)
    base_signal = delta_wave + theta_wave + alpha_wave + beta_wave + gamma_wave
    
    # Generate channel data with correlation and random phase shifts
    for i in range(n_channels):
        # Create phase-shifted copies of each frequency component
        phase_shift = np.random.uniform(0, 2*np.pi, 5)  # one for each frequency band
        
        channel_delta = delta_power * np.sin(2 * np.pi * delta_freq * time + phase_shift[0])
        channel_theta = theta_power * np.sin(2 * np.pi * theta_freq * time + phase_shift[1])
        channel_alpha = alpha_power * np.sin(2 * np.pi * alpha_freq * time + phase_shift[2])
        channel_beta = beta_power * np.sin(2 * np.pi * beta_freq * time + phase_shift[3])
        channel_gamma = gamma_power * np.sin(2 * np.pi * gamma_freq * time + phase_shift[4])
        
        # Combine with correlation factor
        channel_signal = (
            correlation * base_signal + 
            (1 - correlation) * (channel_delta + channel_theta + channel_alpha + channel_beta + channel_gamma)
        )
        
        # Add random noise
        channel_signal += np.random.normal(0, noise_level, n_steps)
        
        # Store in data array
        data[:, i] = channel_signal
    
    # Create 2D "position" by integrating the first two channels
    # (purely for visualization purposes)
    x = np.cumsum(data[:, 0]) * 0.01
    y = np.cumsum(data[:, 1]) * 0.01
    
    # Create DataFrame
    columns = ['time', 'x', 'y'] + [f'channel_{i+1}' for i in range(n_channels)]
    df = pd.DataFrame(
        np.column_stack([time, x, y, data]),
        columns=columns
    )
    
    return df


def main():
    # Generate synthetic data (10 samples of oscillation data)
    print("Generating synthetic data...")
    n_samples = 10
    np.random.seed(42)  # for reproducibility
    
    # Generate two different types of data
    oscillation_data = [
        generate_oscillation_data(
            n_steps=5000,
            n_oscillators=5,
            frequency_drift=np.random.uniform(0.005, 0.015),
            amplitude_drift=np.random.uniform(0.01, 0.03),
            noise_level=np.random.uniform(0.05, 0.15),
            correlation=np.random.uniform(0.2, 0.4)
        )
        for _ in range(5)
    ]
    
    eeg_data = [
        generate_eeg_like_data(
            n_steps=5000,
            n_channels=8,
            alpha_power=np.random.uniform(0.8, 1.2),
            beta_power=np.random.uniform(0.4, 0.6),
            delta_power=np.random.uniform(1.8, 2.2),
            theta_power=np.random.uniform(1.3, 1.7),
            noise_level=np.random.uniform(0.4, 0.6)
        )
        for _ in range(5)
    ]
    
    # Select data type to use for this example
    data_type = "oscillation"  # or "eeg"
    
    if data_type == "oscillation":
        feature_data = oscillation_data
        feature_indices = [3, 4, 5, 6, 7]  # oscillator columns
        feature_names = [f"Oscillator {i+1}" for i in range(5)]
    else:
        feature_data = eeg_data
        feature_indices = list(range(3, 11))  # channel columns
        feature_names = [f"Channel {i+1}" for i in range(8)]
    
    # Plot example data
    example_data = feature_data[0]
    plt.figure(figsize=(12, 8))
    
    # Plot 2D trajectory
    plt.subplot(2, 1, 1)
    plt.plot(example_data['x'], example_data['y'])
    plt.title('Example Trajectory')
    plt.axis('equal')
    
    # Plot features
    plt.subplot(2, 1, 2)
    for i, idx in enumerate(feature_indices):
        plt.plot(
            example_data['time'], 
            example_data.iloc[:, idx], 
            label=feature_names[i]
        )
    plt.title('Feature Time Series')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Run iterative window search with generic feature windows
    print("Running iterative window search...")
    window_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    iterative_results = {}
    
    # Use a smaller subset for faster computation
    data_subset = [df.iloc[:1000].copy() for df in feature_data]
    
    for size in window_sizes:
        print(f"Testing window size {size}...")
        # Use the generic window method with feature indices
        iterative_results[str(size)] = iterative_umap(
            data_subset,
            window_method='generic',
            window_size=size,
            step_size=1,
            feature_indices=feature_indices,
            time_column='time',
            coordinate_indices=(1, 2),  # x, y columns
            plot=False,
            normalize_features=feature_indices,  # normalize all features
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
    
    # Choose a subset of window sizes to calculate recurrence for (to save time)
    recurrence_window_sizes = [1, 10, 20, 30, 40, 50]
    
    for size in recurrence_window_sizes:
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
    for size in recurrence_window_sizes:
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
    
    # Based on the metrics, choose an optimal window size (e.g., 20)
    optimal_window_size = 20
    print(f"Selected optimal window size: {optimal_window_size}")
    
    # Create behavior space with optimal window size
    print("Creating behavior space with optimal window size...")
    windows = []
    
    for i, data in enumerate(feature_data):
        result = get_feature_windows(
            data, 
            window_size=optimal_window_size,
            step_size=1,
            feature_indices=feature_indices,
            time_column='time',
            normalize_features=feature_indices,
            name=f'replicate_{i}'
        )
        windows.append(result)
    
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