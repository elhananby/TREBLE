# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview of TREBLE

TREBLE (Time REsolved BehavioraL Embedding) is a framework for analyzing time-resolved behavioral data. It's designed to identify and visualize behavioral patterns in movement data, particularly focused on animal locomotion. The framework uses dimensionality reduction techniques (primarily UMAP) to create low-dimensional representations of behavioral sequences.

## Repository Structure

This repository now contains both Python and R implementations:

- **Python Implementation (Root Directory)**:
  - `treble/`: Core Python modules implementing TREBLE
  - `examples/`: Example Python scripts demonstrating usage
  - `tests/`: Python unit tests
  - `pyproject.toml`: Python package configuration

- **R Implementation (R_implementation/)**:
  - `00_data/`: Contains sample data files and generated visualizations
  - `01_Rfiles/`: Core R functions and code implementation
  - `02_tutorials/`: Documentation and walkthroughs

## Key Functionality

TREBLE's core functionality includes:

1. **Windowing techniques**: Extracting time windows from behavioral data with configurable window sizes and step sizes
2. **Dimensionality reduction**: Using UMAP to create 2D embeddings of behavioral trajectories
3. **Iterative window search**: Finding optimal window sizes for capturing behavioral patterns
4. **Visualization tools**: Vector fields, feature maps, and density distributions 
5. **Statistical analysis**: Tools to compare distributions and calculate recurrence metrics

## Common Operations

### Python Implementation

```python
import numpy as np
import pandas as pd
from treble import (
    get_feature_windows,
    iterative_umap,
    bin_umap,
    plot_vector_field
)

# Load your feature data
# feature_data can be any time series with arbitrary features

# Specify which columns to use as features
feature_indices = [2, 3, 4, 5]  # Example: columns to use as features
time_column = 0  # Column index for time
coordinate_indices = (6, 7)  # Optional: columns to use as coordinates for visualization

# Run iterative window search with generic features
window_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
iterative_results = {}

for size in window_sizes:
    iterative_results[str(size)] = iterative_umap(
        feature_data,
        window_method='generic',
        window_size=size,
        feature_indices=feature_indices,
        time_column=time_column,
        normalize_features=[3, 4]  # Optional: features to normalize
    )

# Create behavior space with optimal window size
windows = []
for i, data in enumerate(feature_data):
    windows.append(
        get_feature_windows(
            data,
            window_size=15,
            feature_indices=feature_indices,
            time_column=time_column,
            normalize_features=[3, 4],
            name=f'replicate_{i}'
        )
    )
```

### R Implementation

The typical workflow for TREBLE analysis in R:

```R
# Load R packages
library(umap)
library(scales)
library(MASS)
library(RColorBrewer)
library(colorRamps)
library(vegan)

# Source TREBLE functions
source("R_implementation/01_Rfiles/TREBLE_walkthrough_functions_051920.R")

# Load sample data
vel = readRDS("R_implementation/00_data/sample_correlated_random_walk_velocities_5ksteps.RDS")

# Run iterative window search
toSweep = c(1, seq(5, 50, 5))
iterative_windows = list()
for(i in 1:length(toSweep)){
  iterative_windows[[as.character(toSweep[i])]] = iterative_umap(
    lapply(vel, function(x) x[1:1000,]),
    velocity_windows = TRUE,
    window_size = toSweep[i])
}

# Calculate metrics to determine optimal window size
iterative_windows_pr = lapply(iterative_windows, function(y) run_procrustes(y$umaps)$procrustes)
iterative_windows_dist = lapply(iterative_windows, function(y) run_procrustes(y$umaps)$euclidean_distances)

# Visualize results to select optimal window size
plot_results(iterative_windows_pr, ylim = c(0, 8), ylab = "Procrustes RMSD", xlab = "Window size (frames)")
plot_variance(iterative_windows_pr, ylim = c(0, 0.2), ylab = "Procrustes distance (coef. var.)", xlab = "Window size (frames)")

# Create behavior space with optimal window size (15 in this example)
windows = list()
for(i in 1:length(vel)){
  windows[[as.character(i)]] = get_velocity_windows(
    vel[[i]], window_size = 15, name = paste('replicate_', i, sep = ''))
}

# Run UMAP on windows
windows_combined = do.call(cbind, windows)
u = umap(t(windows_combined), verbose = TRUE)
```

## Notes for Developers

- The Python implementation provides a more modern and flexible approach with support for generic feature data
- The R implementation is the original implementation and focuses on velocity data
- When working with large datasets, be mindful of memory usage, especially in the windowing functions
- The Python implementation extends the original concepts with the `get_feature_windows` function for working with arbitrary features
- Check function docstrings for detailed parameter documentation
- Tests for the Python implementation are in the tests/ directory
- The R implementation paths have been updated to reflect the new repository structure