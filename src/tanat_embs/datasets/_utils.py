#!/usr/bin/env python3
"""
Shared utilities for EventDataset and SequenceDataset.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import polars as pl
import torch

from tanat.metadata.feature import (
    ArrayInfo,
    BooleanInfo,
    CategoricalInfo,
    NumericalInfo,
    StringInfo,
    TemporalInfo,
)
from tanat.metadata.sequence import SequenceMetadata


def validate_and_build_feature_dims(
    metadata: SequenceMetadata,
    feature_names: list[str],
    is_static: bool = False,
) -> dict[str, tuple[int, int]]:
    """Validate that all named features are numeric-compatible and return their tensor layout.

    Args:
        metadata: Pool/sequence metadata.
        feature_names: Ordered list of feature names to validate.
        is_static: Whether to look up in static (True) or entity (False) features.

    Returns:
        Dict mapping each feature name to ``(start_col, end_col)`` — end is exclusive.
        Scalar features have width 1; fixed-size Array features have width == dimension.

    Raises:
        ValueError: If any feature is missing, non-numeric, or variable-length.
    """
    bad_columns: list[str] = []
    dims: dict[str, tuple[int, int]] = {}
    cursor = 0

    for name in feature_names:
        info = metadata.feature_info(name, is_static=is_static)
        if info is None:
            scope = "static" if is_static else "entity"
            raise ValueError(f"Feature '{name}' not found in {scope} features.")

        if isinstance(info, (NumericalInfo, BooleanInfo)):
            width = 1
        elif isinstance(info, ArrayInfo):
            if info.dimension is None:
                raise ValueError(
                    f"Feature '{name}' is a variable-length List column, which is not "
                    "supported. Use a fixed-size Array(numeric, N) column instead."
                )
            width = info.dimension
        else:
            bad_columns.append(name)
            continue

        dims[name] = (cursor, cursor + width)
        cursor += width

    if bad_columns:
        raise ValueError(
            f"Non-numeric features found: {bad_columns}. "
            "All dataset features must be numeric (Int*, Float*, Boolean) or "
            "fixed-size numeric arrays (Array(numeric, N)). "
            "To encode categorical columns, apply CategoricalEncoder first."
        )

    return dims


def _temporal_series_to_float(
    series: pl.Series,
    is_datetime: bool,
) -> np.ndarray:
    """Convert a Polars temporal Series to a float64 numpy array (pre-delay subtraction)."""
    if is_datetime:
        return series.dt.epoch("s").to_numpy().astype(np.float64)
    return series.to_numpy().astype(np.float64)


def encode_temporal(
    df: pl.DataFrame,
    temporal_cols: list[str],
    is_datetime: bool,
    encoding: Literal["absolute", "delay"],
    t0: Any = None,
) -> np.ndarray:
    """Encode temporal column(s) from a DataFrame into a float32 numpy array.

    Args:
        df: DataFrame containing the temporal columns.
        temporal_cols: Column names — 1 for event, 2 for interval/state.
        is_datetime: True if the temporal index is Datetime/Date.
        encoding: 'absolute' or 'delay'.
        t0: T0 value matching the temporal column type; required when encoding='delay'.

    Returns:
        shape ``(T,)`` for a single temporal column, ``(T, 2)`` for two columns.
    """
    arrays = []
    for col in temporal_cols:
        series = df[col]
        if encoding == "delay":
            diff = series - t0
            if is_datetime:
                arr = diff.dt.total_seconds().to_numpy().astype(np.float32)
            else:
                arr = diff.to_numpy().astype(np.float32)
        else:
            arr = _temporal_series_to_float(series, is_datetime).astype(np.float32)
        arrays.append(arr)

    if len(arrays) == 1:
        return arrays[0]
    return np.stack(arrays, axis=1)  # (T, 2)


def apply_fill_value(
    arr: np.ndarray,
    fill_value: float,
    seq_id: Any,
    col_names: list[str],
) -> np.ndarray:
    """Replace NaN values or raise ValueError depending on fill_value.

    Args:
        arr: float32 numpy array potentially containing NaNs.
        fill_value: float('nan') to raise on NaN, or a numeric value to fill with.
        seq_id: Used in the error message.
        col_names: Feature column names, used in the error message.

    Returns:
        Array with NaNs replaced (or unchanged if no NaNs).

    Raises:
        ValueError: If fill_value is nan and any NaN is present.
    """
    if not np.isnan(arr).any():
        return arr
    if math.isnan(fill_value):
        raise ValueError(
            f"NaN value found in sequence '{seq_id}' for feature(s) {col_names}. "
            "Set fill_value to a numeric value (e.g. fill_value=0.0) to replace NaNs, "
            "or clean the data before constructing the dataset."
        )
    return np.where(np.isnan(arr), np.float32(fill_value), arr).astype(np.float32)


def build_feature_array(
    row_df: pl.DataFrame,
    feature_names: list[str],
    feature_dims: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Build a float32 numpy array from a DataFrame slice.

    Scalar features contribute 1 column; fixed-size Array features contribute
    ``dimension`` columns, unpacked and concatenated.

    Args:
        row_df: One or more rows from the entity DataFrame.
        feature_names: Ordered list of feature names to include.
        feature_dims: ``{name: (start, end)}`` mapping from validate_and_build_feature_dims.

    Returns:
        shape ``(F,)`` for a single row, ``(T, F)`` for multiple rows.
    """
    n_rows = len(row_df)
    if not feature_names:
        return np.zeros((n_rows, 0), dtype=np.float32).squeeze(0) if n_rows == 1 else np.zeros((n_rows, 0), dtype=np.float32)

    total_width = max(end for _, end in feature_dims.values())
    out = np.empty((n_rows, total_width), dtype=np.float32)

    for name in feature_names:
        start, end = feature_dims[name]
        col = row_df[name]
        if end - start == 1:
            out[:, start] = col.cast(pl.Float32).to_numpy()
        else:
            arr = np.array(col.to_list(), dtype=np.float32)  # (n_rows, width)
            out[:, start:end] = arr

    if n_rows == 1:
        return out[0]  # shape (F,)
    return out  # shape (T, F)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a float32 numpy array to a torch.Tensor."""
    return torch.from_numpy(arr.astype(np.float32))
