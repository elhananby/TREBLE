"""Functions for extracting windows from time-series data."""

from typing import List, Dict, Union, Tuple, Optional, Any, Sequence
import numpy as np
import pandas as pd


def get_windows(
    features: Union[pd.DataFrame, np.ndarray], 
    window_size: int = 10, 
    step_size: int = 1, 
    name: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract windows of user-defined features with size window_size and stepsize step_size.
    
    Args:
        features: DataFrame or numpy array containing features to extract windows from
        window_size: Size of the windows to extract
        step_size: Step size between windows
        name: Optional name prefix for column names
        
    Returns:
        DataFrame with extracted windows as columns
    """
    # Convert to DataFrame if numpy array
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    
    num_samples = len(features)
    window_indices = range(0, num_samples - window_size + 1, step_size)
    
    # Generate windows
    windows = {}
    for i in window_indices:
        window = features.iloc[i:i+window_size].values.flatten()
        col_name = f"{name}_{i}" if name else str(i)
        windows[col_name] = window
    
    # Convert to DataFrame
    windows_df = pd.DataFrame(windows)
    
    return windows_df


def get_feature_windows(
    features: Union[pd.DataFrame, np.ndarray],
    window_size: int = 10,
    step_size: int = 1,
    feature_indices: Optional[Sequence[int]] = None,
    reference_feature: Optional[int] = None,
    normalize_features: Optional[List[int]] = None,
    symmetrize_features: Optional[List[int]] = None, 
    time_column: Optional[Union[int, str]] = None,
    return_coordinates: bool = False,
    coordinate_indices: Optional[Tuple[int, int]] = None,
    verbose: bool = False,
    name: Optional[str] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract feature windows with configurable normalization and symmetrization.
    
    This is a generalized version of get_velocity_windows that works with any feature set.
    
    Args:
        features: DataFrame or numpy array containing features
        window_size: Size of the windows to extract
        step_size: Step size between windows
        feature_indices: Indices of features to include (if None, use all)
        reference_feature: Feature to use for window labels (if None, use indices)
        normalize_features: List of feature indices to normalize (subtract first value)
        symmetrize_features: List of feature indices to symmetrize (make second value positive)
        time_column: Column to use as time reference (for window labels)
        return_coordinates: Whether to return coordinate windows
        coordinate_indices: Tuple of (x_index, y_index) for coordinate data (required if return_coordinates=True)
        verbose: Whether to print progress
        name: Optional name prefix for window labels
        
    Returns:
        DataFrame with extracted windows as columns, or tuple of (feature_windows, coordinate_windows)
    """
    # Convert to DataFrame if numpy array
    if isinstance(features, np.ndarray):
        orig_features = features.copy()
        features = pd.DataFrame(features)
    else:
        orig_features = features.copy()
    
    # If feature_indices not provided, use all columns
    if feature_indices is None:
        feature_indices = list(range(features.shape[1]))
    
    # Set up containers for windows
    feature_windows = {}
    coord_windows = {}
    
    # Get time column if specified
    if time_column is not None:
        if isinstance(time_column, int):
            time_values = features.iloc[:, time_column].values
        else:
            time_values = features[time_column].values
    
    # Generate windows
    for i in range(0, len(features) - window_size, step_size):
        if verbose and i % 10000 == 0:
            print(f"Processing window {i}")
        
        # Create window slice
        window_slice = features.iloc[i:i+window_size+1]
        
        # Extract feature values for this window
        feature_arrays = []
        for feat_idx in feature_indices:
            if isinstance(feat_idx, int):
                feature_values = window_slice.iloc[:, feat_idx].values
            else:
                feature_values = window_slice[feat_idx].values
            
            # Normalize if requested
            if normalize_features and feat_idx in normalize_features:
                feature_values = feature_values - feature_values[0]
            
            # Symmetrize if requested
            if symmetrize_features and feat_idx in symmetrize_features:
                if len(feature_values) > 1 and feature_values[1] < 0:
                    feature_values = feature_values * (-1)
                feature_values = np.abs(feature_values)
            
            feature_arrays.append(feature_values)
        
        # Extract coordinates if requested
        if return_coordinates:
            if coordinate_indices is None:
                raise ValueError("coordinate_indices must be provided if return_coordinates=True")
            
            x_idx, y_idx = coordinate_indices
            if isinstance(x_idx, int):
                x_values = window_slice.iloc[:, x_idx].values
            else:
                x_values = window_slice[x_idx].values
                
            if isinstance(y_idx, int):
                y_values = window_slice.iloc[:, y_idx].values
            else:
                y_values = window_slice[y_idx].values
        
        # Create window label
        if time_column is not None:
            start_time = time_values[i]
            end_time = time_values[i+window_size]
            window_label = f"{start_time}_{end_time}_{name}" if name else f"{start_time}_{end_time}"
        else:
            window_label = f"{i}_{i+window_size}_{name}" if name else f"{i}_{i+window_size}"
        
        # Store feature windows
        feature_windows[window_label] = np.concatenate(feature_arrays)
        
        # Store coordinate windows if requested
        if return_coordinates:
            coord_windows[window_label] = np.concatenate([x_values, y_values])
    
    # Filter out windows with NaN values
    valid_windows = [k for k in feature_windows if not np.isnan(feature_windows[k]).any()]
    feature_windows = {k: feature_windows[k] for k in valid_windows}
    
    # Convert to DataFrames
    feature_windows_df = pd.DataFrame(feature_windows)
    
    if return_coordinates:
        coord_windows = {k: coord_windows[k] for k in valid_windows}
        coord_windows_df = pd.DataFrame(coord_windows)
        return feature_windows_df, coord_windows_df
    else:
        return feature_windows_df


def get_velocity_windows(
    features: Union[pd.DataFrame, np.ndarray],
    include_sideslip: bool = False,
    return_xy_windows: bool = False,
    window_size: int = 1,
    step_size: int = 1,
    symm: bool = False,
    verbose: bool = False,
    name: Optional[str] = None,
    column_mapping: Optional[Dict[str, Union[str, int]]] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract velocity windows and optionally regularize, with size window_size and stepsize step_size.
    
    This is a specific implementation that uses get_feature_windows under the hood, maintaining 
    backward compatibility with the original TREBLE implementation.
    
    Args:
        features: DataFrame or numpy array with velocity data
        include_sideslip: Whether to include sideslip in the windows
        return_xy_windows: Whether to return xy coordinates as well
        window_size: Size of the windows to extract
        step_size: Step size between windows
        symm: Whether to symmetrize the angular velocity
        verbose: Whether to print progress
        name: Optional name prefix for window labels
        column_mapping: Dictionary mapping column names to their indices or names in the data
                        Keys should include: 'time', 'x', 'y', 'translational_velocity', 
                        'angular_velocity', and optionally 'sideslip'
        
    Returns:
        DataFrame with extracted windows as columns, or tuple of (velocity_windows, xy_windows)
    """
    # Set up default column mapping if not provided
    if column_mapping is None:
        if isinstance(features, pd.DataFrame):
            # Try to use column names if they exist
            required_cols = ['time', 'x', 'y', 'translational_velocity', 'angular_velocity']
            if include_sideslip:
                required_cols.append('sideslip')
                
            # Check if columns exist
            missing_cols = [col for col in required_cols if col not in features.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. Please provide column_mapping.")
                
            column_mapping = {col: col for col in required_cols}
        else:
            # For numpy array, assume default column order
            column_mapping = {
                'time': 0,
                'x': 1,
                'y': 2,
                'translational_velocity': 3,
                'angular_velocity': 4
            }
            if include_sideslip:
                column_mapping['sideslip'] = 5
    
    # Set up feature indices and normalization/symmetrization
    feature_indices = [column_mapping['translational_velocity'], column_mapping['angular_velocity']]
    normalize_features = [column_mapping['angular_velocity']]
    symmetrize_features = [column_mapping['angular_velocity']] if symm else []
    
    if include_sideslip:
        feature_indices.append(column_mapping['sideslip'])
        normalize_features.append(column_mapping['sideslip'])
        if symm:
            symmetrize_features.append(column_mapping['sideslip'])
    
    # Call the generic function
    return get_feature_windows(
        features=features,
        window_size=window_size,
        step_size=step_size,
        feature_indices=feature_indices,
        reference_feature=None,
        normalize_features=normalize_features,
        symmetrize_features=symmetrize_features,
        time_column=column_mapping['time'],
        return_coordinates=return_xy_windows,
        coordinate_indices=(column_mapping['x'], column_mapping['y']),
        verbose=verbose,
        name=name
    )