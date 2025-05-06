"""TREBLE_py: Time REsolved BehavioraL Embedding in Python."""

from .windows import get_windows, get_velocity_windows, get_feature_windows
from .umap_tools import iterative_umap, bin_umap
from .analysis import (
    run_procrustes,
    calculate_recurrence,
    calculate_euclidean_distance
)
from .visualization import (
    plot_results,
    plot_variance,
    plot_recurrence,
    plot_vector_field,
    plot_umap_features,
    plot_umap_pdf
)

__all__ = [
    "get_windows",
    "get_velocity_windows",
    "get_feature_windows",
    "iterative_umap",
    "bin_umap",
    "run_procrustes",
    "calculate_recurrence",
    "calculate_euclidean_distance",
    "plot_results",
    "plot_variance",
    "plot_recurrence", 
    "plot_vector_field",
    "plot_umap_features",
    "plot_umap_pdf"
]