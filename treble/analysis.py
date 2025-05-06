"""Analysis tools for TREBLE."""

from typing import List, Dict, Union, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean
from itertools import combinations
import matplotlib.pyplot as plt


def calculate_euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1: First point
        x2: Second point
        
    Returns:
        Euclidean distance
    """
    return euclidean(x1, x2)


def run_procrustes(
    umaps: List[pd.DataFrame],
    run_protest: bool = False
) -> Dict[str, Union[List[float], List[Any]]]:
    """
    Calculate distance between UMAP layouts using Procrustes and Euclidean distances.
    
    Args:
        umaps: List of DataFrames containing UMAP coordinates with 'x' and 'y' columns
        run_protest: Whether to run protest (not implemented in this version)
        
    Returns:
        Dictionary with procrustes and euclidean distances
    """
    if run_protest:
        raise NotImplementedError("Protest analysis is not implemented in this version")
    
    # Get all combinations of UMAPs to compare
    umap_combinations = list(combinations(range(len(umaps)), 2))
    
    # Run Procrustes analysis
    pr_res = []
    dists = []
    
    for i, j in umap_combinations:
        # Extract coordinates
        coords1 = umaps[i][['x', 'y']].values
        coords2 = umaps[j][['x', 'y']].values
        
        # Run Procrustes
        _, _, disparity = procrustes(coords1, coords2)
        pr_res.append(disparity)
        
        # Calculate Euclidean distance
        dists.append(calculate_euclidean_distance(coords1.flatten(), coords2.flatten()))
    
    return {
        "procrustes": pr_res,
        "euclidean_distances": dists
    }


def calculate_recurrence(
    umaps: List[pd.DataFrame],
    filter_outliers: bool = False,
    n_bins: int = 16,
    threshold: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Calculate the amount and timing of recurrence in a behavior space.
    
    Args:
        umaps: List of DataFrames containing UMAP coordinates
        filter_outliers: Whether to filter outliers
        n_bins: Number of bins for binning
        threshold: Threshold for recurrence (quantile)
        
    Returns:
        List of dictionaries with recurrence information
    """
    from .umap_tools import bin_umap
    
    results = []
    
    for h, u in enumerate(umaps):
        print(f"{h+1} out of {len(umaps)}")
        
        # Filter outliers if requested
        if filter_outliers:
            u = u.copy()
            u.loc[u['x'] > 30, 'x'] = 30
            u.loc[u['x'] < -30, 'x'] = -30
            u.loc[u['y'] > 30, 'y'] = 30
            u.loc[u['y'] < -30, 'y'] = -30
        
        # Bin the UMAP space
        l, _ = bin_umap(u, n_bins=n_bins)
        
        # Get unique positions
        pos = sorted(set(l['xy_new']))
        
        # Calculate distances
        dists = {}
        for i, p in enumerate(pos):
            # Get bin coordinates
            x_bin = int(p.split('_')[0])
            y_bin = int(p.split('_')[1])
            
            # Calculate distances to all points
            z = []
            for _, row in l.iterrows():
                dist = calculate_euclidean_distance(
                    [x_bin, y_bin], 
                    [int(row['xnew']), int(row['ynew'])]
                )
                z.append(dist)
            
            dists[p] = z
        
        # Calculate distance threshold
        all_dists = [dist for dist_list in dists.values() for dist in dist_list]
        thresh = np.quantile(all_dists, threshold)
        
        # Extract recurrences using threshold
        recs = {}
        for pos_name, distances in dists.items():
            rs = [i for i, d in enumerate(distances) if d < thresh]
            ds = [rs[i+1] - rs[i] for i in range(len(rs)-1) if rs[i+1] - rs[i] > thresh]
            recs[pos_name] = ds
        
        # Calculate histogram
        all_recs = [r for rec_list in recs.values() for r in rec_list]
        if not all_recs:  # If no recurrences found
            hist_counts = np.zeros(200)
            hist_edges = np.arange(201)
        else:
            hist_counts, hist_edges = np.histogram(
                all_recs, 
                bins=np.arange(1, max(all_recs, default=200) + 2)
            )
            # Pad with zeros if needed
            if len(hist_counts) < 200:
                hist_counts = np.pad(hist_counts, (0, 200 - len(hist_counts)), 'constant')
        
        # Calculate proportion recurrent in bins
        prop_recurrent = []
        for i in range(1, 201):
            z = sum(1 for rec_list in recs.values() if i in rec_list)
            prop_recurrent.append(z / len(recs) if recs else 0)
        
        # Calculate total proportion recurrent
        total_recurrent = sum(1 for rec_list in recs.values() if rec_list) / len(recs) if recs else 0
        
        # Store results
        result = {
            "distances": dists,
            "recurrences": all_recs,
            "histogram": {
                "counts": hist_counts,
                "edges": hist_edges
            },
            "proportion_recurrent_in_bins": prop_recurrent,
            "total_proportion_recurrent": total_recurrent
        }
        
        results.append(result)
    
    # Create a heatmap of recurrence
    plt.figure(figsize=(10, 6))
    recurrence_matrix = np.column_stack(
        [r["histogram"]["counts"][:200] / max(r["histogram"]["counts"][:200], default=1) 
         for r in results]
    )
    
    plt.imshow(
        recurrence_matrix, 
        aspect='auto', 
        interpolation='nearest',
        cmap='plasma'
    )
    
    plt.xlabel('Time')
    plt.ylabel('Replicate')
    plt.title('Recurrence Distribution')
    plt.colorbar(label='Normalized Count')
    plt.tight_layout()
    plt.show()
    
    return results