"""UMAP-related tools for TREBLE."""

from typing import List, Dict, Union, Tuple, Optional, Any, Sequence
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import warnings


def bin_umap(
    layout: pd.DataFrame,
    n_bins: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Bin a UMAP space into a n x n grid.
    
    Args:
        layout: DataFrame with x and y columns representing UMAP coordinates
        n_bins: Number of bins for each dimension
        
    Returns:
        Tuple of (binned layout DataFrame, new coordinates as strings)
    """
    # Create a copy to avoid modifying the original
    layout = layout.copy()
    
    # Split x into n bins
    x_min, x_max = layout['x'].min(), layout['x'].max()
    x_bins = np.linspace(x_min, x_max, n_bins+1)
    
    # Assign each point to a bin for x
    xnew = []
    for x_val in layout['x']:
        bin_idx = np.abs(x_bins - x_val).argmin()
        xnew.append(str(bin_idx + 1))  # Use 1-based indexing like in R
    
    layout['xnew'] = xnew
    
    # Split y into n bins
    y_min, y_max = layout['y'].min(), layout['y'].max()
    y_bins = np.linspace(y_min, y_max, n_bins+1)
    
    # Assign each point to a bin for y
    ynew = []
    for y_val in layout['y']:
        bin_idx = np.abs(y_bins - y_val).argmin()
        ynew.append(str(bin_idx + 1))  # Use 1-based indexing like in R
    
    layout['ynew'] = ynew
    
    # Combine x and y bins
    xy_new = [f"{x}_{y}" for x, y in zip(xnew, ynew)]
    layout['xy_new'] = xy_new
    
    # Get unique coordinates and sort them
    unique_coords = sorted(set(xy_new), key=lambda c: (int(c.split('_')[0]), int(c.split('_')[1])))
    
    # Assign numeric coordinates
    coord_map = {coord: str(i+1) for i, coord in enumerate(unique_coords)}
    layout['coords'] = [coord_map[xy] for xy in xy_new]
    
    return layout, xy_new


def iterative_umap(
    features: List[Union[pd.DataFrame, np.ndarray]],
    window_method: str = 'standard',
    verbose: bool = False,
    plot: bool = False,
    step_size: int = 1,
    window_size: int = 30,
    n_bins: int = 32,
    run_umap: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Iteratively run UMAP on windows of a desired size.
    
    Args:
        features: List of DataFrames or numpy arrays containing features
        window_method: Method to use for windowing ('standard', 'velocity', or 'generic')
        verbose: Whether to print progress
        plot: Whether to plot results
        step_size: Step size between windows
        window_size: Size of windows to extract
        n_bins: Number of bins for binning the UMAP space
        run_umap: Whether to run UMAP (or just extract windows)
        n_neighbors: UMAP parameter for number of neighbors
        min_dist: UMAP parameter for minimum distance
        metric: UMAP parameter for distance metric
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments to pass to windowing function
            For window_method='standard': passed to get_windows
            For window_method='velocity': passed to get_velocity_windows
            For window_method='generic': passed to get_feature_windows
        
    Returns:
        Dictionary with features, windows, and optionally UMAP results
    """
    print("Getting windows")
    
    # Get windows
    windows = []
    
    from .windows import get_velocity_windows, get_windows, get_feature_windows
    
    for i, feature_data in enumerate(features):
        if window_method == 'velocity':
            windows.append(
                get_velocity_windows(
                    feature_data,
                    window_size=window_size,
                    step_size=step_size,
                    **kwargs
                )
            )
        elif window_method == 'generic':
            windows.append(
                get_feature_windows(
                    feature_data,
                    window_size=window_size,
                    step_size=step_size,
                    **kwargs
                )
            )
        else:  # default to standard
            windows.append(
                get_windows(
                    feature_data,
                    window_size=window_size,
                    step_size=step_size,
                    name=kwargs.get('name', None)
                )
            )
    
    if not run_umap:
        return {"features": features, "windows": windows}
    
    print("Running UMAP")
    # Run UMAP
    umaps = []
    for i, window in enumerate(windows):
        print(f"UMAP {i+1} out of {len(windows)}")
        
        # If returning coordinate windows, we only want the first element (feature windows)
        if isinstance(window, tuple):
            window = window[0]
        
        # Transpose to match the R implementation's data format
        window_data = window.values.T
        
        # Handle potential warnings from UMAP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                umap_result = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state,
                    verbose=True
                ).fit_transform(window_data)
            else:
                umap_result = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state
                ).fit_transform(window_data)
        
        # Create DataFrame with UMAP coordinates
        umap_df = pd.DataFrame({
            'x': umap_result[:, 0],
            'y': umap_result[:, 1]
        })
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot points
            ax1.scatter(
                umap_result[:, 0],
                umap_result[:, 1],
                c='grey', 
                alpha=0.5,
                s=5
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Points (Window {i+1})")
            
            # Plot lines
            ax2.plot(
                umap_result[:, 0],
                umap_result[:, 1],
                c='grey',
                alpha=0.5,
                linewidth=1
            )
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f"Trajectory (Window {i+1})")
            
            plt.tight_layout()
            plt.show()
        
        umaps.append(umap_df)
    
    # Bin the UMAP space
    binned_umaps = []
    for umap_layout in umaps:
        binned_layout, _ = bin_umap(umap_layout, n_bins=n_bins)
        binned_umaps.append(binned_layout)
    
    return {
        "features": features,
        "windows": windows,
        "umaps": binned_umaps
    }