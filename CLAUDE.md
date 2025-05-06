# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview of TREBLE

TREBLE (Time REsolved BehavioraL Embedding) is an R-based framework for analyzing time-resolved behavioral data. It's designed to identify and visualize behavioral patterns in movement data, particularly focused on animal locomotion. The framework uses dimensionality reduction techniques (primarily UMAP) to create low-dimensional representations of behavioral sequences.

## Repository Structure

- `00_data/`: Contains sample data files and generated visualizations
  - Sample datasets like correlated random walk velocities
  - UMAP coordinate visualization images

- `01_Rfiles/`: Core R functions and code implementation
  - `TREBLE_walkthrough_functions_051920.R`: Core functions for the TREBLE methodology
  - `TREBLE_walkthrough_code_052020.R`: Example code applying the functions

- `02_tutorials/`: Documentation and walkthroughs
  - `TREBLE_iterative_windows_walkthrough.md/.html`: Tutorial explaining how to use TREBLE

## Key Functionality

TREBLE's core functionality includes:

1. **Windowing techniques**: Extracting time windows from behavioral data with configurable window sizes and step sizes
2. **Dimensionality reduction**: Using UMAP to create 2D embeddings of behavioral trajectories
3. **Iterative window search**: Finding optimal window sizes for capturing behavioral patterns
4. **Visualization tools**: Vector fields, feature maps, and density distributions 
5. **Statistical analysis**: Tools to compare distributions and calculate recurrence metrics

## Common Operations

### Running TREBLE Analysis

The typical workflow for TREBLE analysis:

1. Load velocity/movement data
2. Run iterative window search to find optimal window size
3. Generate UMAP embeddings for the chosen window size
4. Analyze and visualize the embeddings

```R
# Load R packages
library(umap)
library(scales)
library(MASS)
library(RColorBrewer)
library(colorRamps)
library(vegan)

# Source TREBLE functions
source("01_Rfiles/TREBLE_walkthrough_functions_051920.R")

# Load sample data
vel = readRDS("00_data/sample_correlated_random_walk_velocities_5ksteps.RDS")

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

# Calculate recurrence metrics
recurrence = list()
for(i in 1:length(iterative_windows)){
  recurrence[[as.character(names(iterative_windows)[i])]] = calculate_recurrence(
    iterative_windows[[i]]$umaps, n_bins = 16)
}

# Create behavior space with optimal window size (15 in this example)
windows = list()
for(i in 1:length(vel)){
  windows[[as.character(i)]] = get_velocity_windows(
    vel[[i]], window_size = 15, name = paste('replicate_', i, sep = ''))
}

# Run UMAP on windows
windows_combined = do.call(cbind, windows)
u = umap(t(windows_combined), verbose = TRUE)

# Visualize the behavior space
layout = data.frame(
  x = u$layout[,1],
  y = u$layout[,2],
  individual = paste(
    unlist(lapply(strsplit(colnames(windows_combined), "_"), function(v){v[3]})),
    unlist(lapply(strsplit(colnames(windows_combined), "_"), function(v){v[4]})),
    sep= ''),
  time = unlist(lapply(strsplit(colnames(windows_combined), "_"), function(v){v[1]}))
)

# Bin the coordinates
layout = bin_umap(layout, n_bins = 32)$layout

# Visualize the space with various plotting techniques
plot_vector_field(layout)
plot_vector_field(layout, color_by_theta = TRUE)
plot_umap_features(
  layout, windows_combined, 
  feature_names = c('Translational velocity', 'Angular velocity'),
  colors = c('darkgreen', 'darkmagenta'),
  n_features = 2
)
```

## Notes for Developers

- The repository appears to be a research codebase focused on method demonstration rather than an installable package
- Core functionality is contained in the function definitions in `TREBLE_walkthrough_functions_051920.R`
- When working with this code, check the function documentation in the R files for parameter usage
- Analyses are computationally intensive, especially with large datasets
- The `treble` library is referenced in the walkthrough but isn't included in this repository