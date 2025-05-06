"""Visualization tools for TREBLE."""

from typing import List, Dict, Union, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import gaussian_kde
from itertools import combinations


def plot_results(
    res_list: Dict[str, List[float]],
    ylim: Tuple[float, float] = (0, 10),
    ylab: Optional[str] = None,
    xlab: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot results of iterative tests.
    
    Args:
        res_list: Dictionary with window sizes as keys and lists of values as values
        ylim: Y-axis limits
        ylab: Y-axis label
        xlab: X-axis label
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    means = {k: np.mean(v) for k, v in res_list.items()}
    
    # Sort by key (window size)
    sorted_keys = sorted(means.keys(), key=lambda x: float(x) if isinstance(x, str) else x)
    sorted_means = [means[k] for k in sorted_keys]
    
    # Set up x-axis positions
    x_pos = np.arange(len(sorted_means))
    
    # Plot individual points for each window size
    for i, k in enumerate(sorted_keys):
        y_values = res_list[k]
        x_jittered = np.random.normal(i, 0.1, len(y_values))
        plt.scatter(x_jittered, y_values, alpha=0.5, color='gray', s=30)
    
    # Plot means
    plt.plot(x_pos, sorted_means, 'o-', color='darkgray', markersize=8, 
             markerfacecolor='black', markeredgecolor='black')
    
    # Customize plot
    plt.xticks(x_pos, [str(k) for k in sorted_keys], rotation=45)
    plt.ylim(ylim)
    
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_variance(
    res_list: Dict[str, List[float]],
    ylim: Tuple[float, float] = (0, 1),
    ylab: Optional[str] = None,
    xlab: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    return_values: bool = False
) -> Optional[Dict[str, float]]:
    """
    Plot results of iterative tests as normalized variance.
    
    Args:
        res_list: Dictionary with window sizes as keys and lists of values as values
        ylim: Y-axis limits
        ylab: Y-axis label
        xlab: X-axis label
        title: Plot title
        figsize: Figure size
        return_values: Whether to return variance values
        
    Returns:
        Dictionary with window sizes as keys and variance values as values (if return_values=True)
    """
    plt.figure(figsize=figsize)
    
    # Calculate coefficient of variation (normalized variance)
    variance = {k: np.std(v) / np.mean(v) if np.mean(v) != 0 else 0 for k, v in res_list.items()}
    
    # Sort by key (window size)
    sorted_keys = sorted(variance.keys(), key=lambda x: float(x) if isinstance(x, str) else x)
    sorted_variance = [variance[k] for k in sorted_keys]
    
    # Set up x-axis positions
    x_pos = np.arange(len(sorted_variance))
    
    # Plot variance
    plt.plot(x_pos, sorted_variance, 'o-', color='darkgray', markersize=8, 
             markerfacecolor='black', markeredgecolor='black')
    
    # Customize plot
    plt.xticks(x_pos, [str(k) for k in sorted_keys], rotation=45)
    plt.ylim(ylim)
    
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    if return_values:
        return variance
    return None


def plot_recurrence(
    recurrences: Dict[str, List[Dict[str, Any]]],
    figsize: Tuple[int, int] = (10, 12)
) -> None:
    """
    Plot recurrence results.
    
    Args:
        recurrences: Dictionary with window sizes as keys and recurrence results as values
        figsize: Figure size
    """
    # Sort window sizes
    window_sizes = sorted(recurrences.keys(), key=lambda x: int(x))
    
    # Create figure with one row per window size
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=figsize)
    if len(window_sizes) == 1:
        axes = [axes]  # Make it iterable for a single subplot
    
    # Create a custom colormap (yellow to orange to red)
    cmap = LinearSegmentedColormap.from_list(
        'YlOrRd', ['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', 
                   '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    )
    
    # Plot each window size
    for i, window_size in enumerate(window_sizes):
        # Get the recurrence data for this window size
        recurrence_data = recurrences[window_size]
        
        # Create matrix of proportion recurrent in bins
        recurrence_matrix = np.column_stack(
            [r["proportion_recurrent_in_bins"] for r in recurrence_data]
        )
        
        # Plot heatmap
        im = axes[i].imshow(
            recurrence_matrix.T,  # Transpose for correct orientation
            aspect='auto',
            interpolation='nearest',
            cmap=cmap,
            vmin=0,
            vmax=recurrence_matrix.max()
        )
        
        axes[i].set_title(f'{window_size} frames', fontsize=12)
        axes[i].set_yticks([])
        
        if i == len(window_sizes) - 1:
            # Add x-axis labels to the bottom subplot
            max_time = 200
            tick_positions = np.linspace(0, recurrence_matrix.shape[0] - 1, 9)
            tick_labels = [str(int(t * max_time / 8)) for t in range(9)]
            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_labels)
            axes[i].set_xlabel('Time')
        else:
            axes[i].set_xticks([])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Proportion Recurrent')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_vector_field(
    layout: pd.DataFrame,
    bin_umap: bool = False,
    n_bins: int = 32,
    color_by_theta: bool = False,
    arrow_color: str = 'gray',
    figsize: Tuple[int, int] = (10, 8),
    return_layout: bool = False
) -> Optional[pd.DataFrame]:
    """
    Plot UMAP layout as a vector field.
    
    Args:
        layout: DataFrame with UMAP coordinates
        bin_umap: Whether to bin the UMAP space
        n_bins: Number of bins for binning
        color_by_theta: Whether to color arrows by angle
        arrow_color: Color for arrows (if not coloring by theta)
        figsize: Figure size
        return_layout: Whether to return the layout DataFrame
        
    Returns:
        Binned layout DataFrame (if return_layout=True)
    """
    from .umap_tools import bin_umap as bin_umap_func
    from .analysis import calculate_euclidean_distance
    
    # Create a copy to avoid modifying the original
    layout = layout.copy()
    
    # Bin UMAP if requested
    if bin_umap:
        layout, _ = bin_umap_func(layout, n_bins=n_bins)
    
    # Calculate directional derivatives
    layout['dx'] = np.concatenate([[0], np.diff(layout['x'])])
    layout['dy'] = np.concatenate([[0], np.diff(layout['y'])])
    
    # Group by bin and calculate mean derivatives
    if 'xy_new' not in layout.columns:
        # If not binned already, create a simple xy identifier
        layout['xy'] = [f"{i}" for i in range(len(layout))]
        bins = layout.groupby('xy')
    else:
        bins = layout.groupby('xy_new')
    
    # Calculate mean dx and dy for each bin
    dx_mean = bins['dx'].mean()
    dy_mean = bins['dy'].mean()
    
    # Create a dataframe for the vector field
    if 'xy_new' in layout.columns:
        # For binned data, use bin coordinates
        df = pd.DataFrame({
            'x': [float(xy.split('_')[0]) for xy in dx_mean.index],
            'y': [float(xy.split('_')[1]) for xy in dx_mean.index],
            'dx': dx_mean.values,
            'dy': dy_mean.values
        })
    else:
        # For unbinned data, use actual coordinates
        bin_centers = bins.apply(lambda g: pd.Series({
            'x': g['x'].mean(),
            'y': g['y'].mean()
        }))
        df = pd.DataFrame({
            'x': bin_centers['x'],
            'y': bin_centers['y'],
            'dx': dx_mean.values,
            'dy': dy_mean.values
        })
    
    # Calculate theta (angle) and distance for each vector
    df['theta'] = np.degrees(np.arctan2(df['dy'], df['dx']))
    df['dist'] = [calculate_euclidean_distance([0, 0], [dx, dy]) for dx, dy in zip(df['dx'], df['dy'])]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set up colors
    if color_by_theta:
        # Create a colormap for theta
        cmap = cm.hsv
        norm = Normalize(vmin=-180, vmax=180)
        colors = cmap(norm(df['theta']))
    else:
        colors = [arrow_color] * len(df)
    
    # Draw the vector field
    plt.quiver(
        df['x'], df['y'], 
        df['dx'], df['dy'],
        color=colors,
        scale=30,  # Adjust scale as needed
        width=0.003,  # Adjust width as needed
        headwidth=4,
        headlength=5
    )
    
    # Customize plot
    plt.axis('equal')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    if color_by_theta:
        # Add a colorbar for theta
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label='Angle (degrees)')
        cbar.set_ticks([-180, -90, 0, 90, 180])
    
    plt.tight_layout()
    plt.show()
    
    if return_layout:
        return layout
    return None


def plot_umap_features(
    layout: pd.DataFrame,
    windows: pd.DataFrame,
    bin_umap: bool = False,
    n_bins: int = 32,
    n_features: int = None,
    feature_names: Optional[List[str]] = None,
    colors: List[str] = None,
    plot_points: bool = False,
    figsize: Tuple[int, int] = (15, 5),
    return_layout: bool = False
) -> Optional[pd.DataFrame]:
    """
    Plot UMAP layout with features colored.
    
    Args:
        layout: DataFrame with UMAP coordinates
        windows: DataFrame with windows as columns
        bin_umap: Whether to bin the UMAP space
        n_bins: Number of bins for binning
        n_features: Number of features to plot
        feature_names: Names of features to plot
        colors: Colors for features
        plot_points: Whether to plot individual points or bins
        figsize: Figure size
        return_layout: Whether to return the layout DataFrame
        
    Returns:
        Layout DataFrame with feature values added (if return_layout=True)
    """
    from .umap_tools import bin_umap as bin_umap_func
    
    # Create a copy to avoid modifying the original
    layout = layout.copy()
    
    # Bin UMAP if requested
    if bin_umap:
        layout, _ = bin_umap_func(layout, n_bins=n_bins)
    
    # Set default colors if not provided
    if colors is None:
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', 
                  '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']
    
    # Ensure we have enough colors
    if n_features and len(colors) < n_features:
        colors = colors * (n_features // len(colors) + 1)
    
    # Ensure n_features is set
    if n_features is None:
        if feature_names:
            n_features = len(feature_names)
        else:
            n_features = 2  # Default to 2 features
    
    # Create figure with one subplot per feature
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    if n_features == 1:
        axes = [axes]  # Make it iterable for a single subplot
    
    # Split windows dataframe into features
    window_rows = windows.shape[0]
    feature_rows = window_rows // n_features
    
    # Create a list of dataframes, one per feature
    feature_dfs = []
    for i in range(n_features):
        start_idx = i * feature_rows
        end_idx = (i + 1) * feature_rows if i < n_features - 1 else window_rows
        feature_dfs.append(windows.iloc[start_idx:end_idx])
    
    # Process each feature
    for i, feat_df in enumerate(feature_dfs):
        # Calculate mean feature value per window
        m = feat_df.mean(axis=0)
        
        # Add to layout
        if feature_names and i < len(feature_names):
            layout[feature_names[i]] = m.values
        else:
            layout[f'feature_{i+1}'] = m.values
        
        # Create colormap for this feature
        color = colors[i % len(colors)]
        cmap = LinearSegmentedColormap.from_list('custom', ['lightgray', color])
        
        if plot_points:
            # Plot individual points
            feature_values = np.abs(m.values)  # Take absolute value
            vmax = feature_values.max()
            
            # Normalize feature values for colormapping
            norm = Normalize(vmin=0, vmax=vmax)
            
            scatter = axes[i].scatter(
                layout['x'], 
                layout['y'],
                c=feature_values,
                cmap=cmap,
                s=10,
                alpha=0.7
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[i], label='Value')
            
        else:
            # Plot bins
            if 'xy_new' in layout.columns:
                # Group by bin
                binned_values = layout.groupby('xy_new')[layout.columns[-1]].mean()
                
                # Extract bin coordinates
                x_coords = [float(xy.split('_')[0]) for xy in binned_values.index]
                y_coords = [float(xy.split('_')[1]) for xy in binned_values.index]
                
                # Take absolute value of feature values
                feature_values = np.abs(binned_values.values)
                vmax = feature_values.max()
                
                # Normalize feature values for colormapping
                norm = Normalize(vmin=0, vmax=vmax)
                
                scatter = axes[i].scatter(
                    x_coords, 
                    y_coords,
                    c=feature_values,
                    cmap=cmap,
                    s=50,
                    alpha=0.7
                )
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[i], label='Value')
            else:
                # If not binned, just show a warning
                axes[i].text(0.5, 0.5, 'Binning required\nfor bin-based visualization',
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[i].transAxes)
        
        # Set subplot title
        if feature_names and i < len(feature_names):
            axes[i].set_title(feature_names[i])
        else:
            axes[i].set_title(f'Feature {i+1}')
        
        # Remove ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    if return_layout:
        return layout
    return None


def plot_umap_pdf(
    layout: pd.DataFrame,
    h: float = 1.0,
    n: int = 100,
    colors: Any = 'plasma',
    figsize: Tuple[int, int] = (8, 8),
    return_pdf: bool = False
) -> Optional[Dict[str, np.ndarray]]:
    """
    Plot UMAP layout as a probability density function.
    
    Args:
        layout: DataFrame with UMAP coordinates
        h: Bandwidth parameter for KDE
        n: Number of grid points in each dimension
        colors: Colormap for the plot
        figsize: Figure size
        return_pdf: Whether to return the PDF values
        
    Returns:
        Dictionary with PDF values (if return_pdf=True)
    """
    # Extract x and y coordinates
    x = layout['x'].values
    y = layout['y'].values
    
    # Calculate 2D KDE
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a bit of padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Create meshgrid
    xx, yy = np.mgrid[x_min:x_max:complex(0, n), y_min:y_max:complex(0, n)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Calculate KDE
    kernel = gaussian_kde(np.vstack([x, y]), bw_method=h)
    pdf = np.reshape(kernel(positions), xx.shape)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot the PDF
    plt.imshow(
        pdf.T,  # Transpose for correct orientation
        origin='lower',
        aspect='auto',
        extent=[x_min, x_max, y_min, y_max],
        cmap=colors
    )
    
    # Customize plot
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Density')
    plt.title('UMAP Probability Density')
    
    plt.tight_layout()
    plt.show()
    
    if return_pdf:
        return {
            'x': xx[:, 0],
            'y': yy[0, :],
            'z': pdf
        }
    return None