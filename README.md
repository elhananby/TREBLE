# TREBLE: Time REsolved BehavioraL Embedding

A framework for analyzing time-resolved behavioral data, available in both Python and R implementations.

## Overview

TREBLE is designed to identify and visualize behavioral patterns in time series data, including but not limited to animal locomotion. The framework uses dimensionality reduction techniques (primarily UMAP) to create low-dimensional representations of behavioral sequences.

This repository contains:
1. A modern Python implementation with extended features
2. The original R implementation (in the `R_implementation` directory)

## Python Implementation

This Python implementation provides a generalized version of the original R-based TREBLE framework, making it accessible to Python users and extending its capabilities to work with any type of sequential feature data.

### Python Features

- **Flexible data handling**: Works with any time series data, not just specific velocity components
- **Generalized windowing**: Extract windows from any type of feature data with customizable normalization and symmetrization
- **Dimensionality reduction**: Use UMAP to create 2D embeddings of behavioral trajectories
- **Iterative window search**: Find optimal window sizes for capturing behavioral patterns
- **Visualization tools**: Create vector fields, feature maps, and density distributions
- **Statistical analysis**: Compare distributions and calculate recurrence metrics

### Installation

#### Using UV (recommended)

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

#### Using pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Quick Start (Python)

#### Velocity Data (Original TREBLE approach)

```python
import numpy as np
import pandas as pd
from treble import (
    get_velocity_windows,
    iterative_umap,
    bin_umap,
    plot_vector_field
)

# Load your velocity data
# velocity_data should be a list of dataframes with columns:
# time, x, y, translational_velocity, angular_velocity

# Run iterative window search
window_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
iterative_results = {}

for size in window_sizes:
    iterative_results[str(size)] = iterative_umap(
        velocity_data, 
        window_method='velocity',
        window_size=size
    )

# Create behavior space with optimal window size
windows = []
for i, vel in enumerate(velocity_data):
    windows.append(
        get_velocity_windows(
            vel, window_size=15, name=f'replicate_{i}'
        )
    )

# Run UMAP and visualize (see examples for details)
```

#### Generic Feature Data (New Approach)

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
# e.g., EEG data, physiological measurements, sensor readings, etc.

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

# Run UMAP and visualize (see examples for details)
```

### Python Examples

See the `examples` directory for complete usage examples:

```bash
# Run the basic example with velocity data
python examples/basic_example.py

# Run the example with generic feature data
python examples/generic_features_example.py
```

### Python Documentation

The API documentation is available in the docstrings of each function. The main modules are:

- `treble.windows`: Functions for extracting windows from time-series data
- `treble.umap_tools`: UMAP-related tools for dimensionality reduction
- `treble.analysis`: Analysis functions for calculating metrics and statistics
- `treble.visualization`: Visualization tools for plotting results

#### Key Functions

- `get_windows`: Extract basic windows from any feature data
- `get_velocity_windows`: Extract velocity windows (backward compatibility with original TREBLE)
- `get_feature_windows`: Generic windowing function with customizable feature selection and processing
- `iterative_umap`: Run UMAP on windows of different sizes to find optimal window size
- `bin_umap`: Bin UMAP coordinates into a grid
- `run_procrustes`: Calculate distance between UMAP layouts
- `calculate_recurrence`: Calculate recurrence in behavior space
- Various plotting functions for visualizing results

## R Implementation

The original R implementation is located in the `R_implementation` directory with the following structure:

- `00_data/`: Contains sample data files and generated visualizations
- `01_Rfiles/`: Core R functions and code implementation
- `02_tutorials/`: Documentation and walkthroughs

To use the R implementation, refer to the tutorial in `R_implementation/02_tutorials/`.

## Development

### Setting up for development

```bash
# Clone the repository
git clone https://github.com/yourusername/TREBLE.git
cd TREBLE

# Create a virtual environment with development dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## License

MIT

## Credits

Original R-based TREBLE framework developed by Ryan York.
Python implementation and extension by Claude.