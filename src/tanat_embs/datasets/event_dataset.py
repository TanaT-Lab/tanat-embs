#!/usr/bin/env python3
"""
EventDataset: row-level PyTorch Dataset wrapping a TanaT SequencePool or Sequence.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from tanat.sequence.base.pool import SequencePool
from tanat.sequence.base.sequence import Sequence
from tanat.zeroing.base import _T0

from ._utils import (
    apply_fill_value,
    build_feature_array,
    encode_temporal,
    to_tensor,
    validate_and_build_feature_dims,
)


class EventDataset(Dataset):
    """A PyTorch Dataset where each item is a single entity row from a TanaT pool.

    The pool is flattened across all sequences: iterating the dataset visits
    every event in every sequence, in pool order.

    A flat index ``[(seq_id, row_rank), ...]`` is built at construction time so
    that ``__getitem__`` is O(1). All data is collected eagerly — no reference to
    the original pool is retained after construction.

    Args:
        pool: Any ``SequencePool`` (event, interval, or state) or a single
            ``Sequence``.
        features: Entity feature columns to include.
            ``None`` = all entity features. ``[]`` raises ``ValueError``.
        static_features: Static feature columns to include.
            ``None`` = all static features. ``[]`` = no static features.
        label_feature: Feature name to use as the label. Static features take
            priority over entity features when the name appears in both.
            The label column is removed from the corresponding feature list.
            ``None`` = no label.
        fill_value: How to handle NaN in numeric features.
            ``float('nan')`` (default) raises ``ValueError`` on any NaN encountered
            at access time. Any numeric value silently fills NaNs.
        temporal_encoding: ``'absolute'`` returns raw temporal values cast to
            float32. ``'delay'`` returns the signed offset from each sequence's
            T0 (in seconds for datetime sequences).
    """

    def __init__(
        self,
        pool: SequencePool | Sequence,
        features: list[str] | None = None,
        static_features: list[str] | None = None,
        label_feature: str | None = None,
        fill_value: float = float("nan"),
        temporal_encoding: Literal["absolute", "delay"] = "absolute",
    ) -> None:
        # ------------------------------------------------------------------ #
        # 1. Normalise input: Sequence → treat like a single-sequence pool    #
        # ------------------------------------------------------------------ #
        if isinstance(pool, Sequence):
            seq = pool
            metadata = seq.metadata
            settings = seq.settings
            entity_df = seq.temporal_data(fmt="polars")
            static_df_raw = seq.static_data(fmt="polars")
            t0_dict: dict[Any, Any] = {seq.id_value: seq.t0}
        elif isinstance(pool, SequencePool):
            metadata = pool.metadata
            settings = pool.settings
            entity_df = pool.temporal_data(fmt="polars")
            static_df_raw = pool.static_data(fmt="polars")
            if temporal_encoding == "delay":
                t0_df = pool.t0_data(fmt="polars")
                id_col_name = settings.id_column
                t0_dict = dict(zip(t0_df[id_col_name].to_list(), t0_df[_T0].to_list()))
            else:
                t0_dict = {}
        else:
            raise TypeError(
                f"Expected SequencePool or Sequence, got {type(pool).__name__}."
            )

        id_col = settings.id_column
        temporal_cols = settings.get_time_columns()
        is_datetime = metadata.time_index.is_datetime

        # ------------------------------------------------------------------ #
        # 2. Resolve feature lists                                            #
        # ------------------------------------------------------------------ #
        all_entity_names = [f.name for f in metadata.entity_features]
        all_static_names = (
            [f.name for f in metadata.static_features]
            if metadata.static_features is not None
            else []
        )

        if features is None:
            entity_names = list(all_entity_names)
        elif len(features) == 0:
            raise ValueError(
                "features=[] is not allowed. Provide at least one feature name "
                "or use features=None to include all entity features."
            )
        else:
            entity_names = list(features)

        if static_features is None:
            static_names = list(all_static_names)
        else:
            static_names = list(static_features)  # [] = no static features

        # ------------------------------------------------------------------ #
        # 3. Resolve label_feature                                            #
        # ------------------------------------------------------------------ #
        label_is_static: bool | None = None
        if label_feature is not None:
            if label_feature in all_static_names:
                # Static takes priority
                label_is_static = True
                if label_feature in static_names:
                    static_names.remove(label_feature)
            elif label_feature in all_entity_names:
                label_is_static = False
                if label_feature in entity_names:
                    entity_names.remove(label_feature)
            else:
                raise ValueError(
                    f"label_feature='{label_feature}' not found in either "
                    "static or entity features."
                )

        # ------------------------------------------------------------------ #
        # 4. Validate feature types and build dimension maps                  #
        # ------------------------------------------------------------------ #
        feature_dims = validate_and_build_feature_dims(
            metadata, entity_names, is_static=False
        )
        static_feature_dims = (
            validate_and_build_feature_dims(metadata, static_names, is_static=True)
            if static_names
            else {}
        )

        if label_feature is not None:
            assert label_is_static is not None
            validate_and_build_feature_dims(
                metadata, [label_feature], is_static=label_is_static
            )

        # ------------------------------------------------------------------ #
        # 5. Warn if any sequence lacks T0 (delay mode)                       #
        # ------------------------------------------------------------------ #
        if temporal_encoding == "delay":
            missing_t0 = [str(sid) for sid, t0 in t0_dict.items() if t0 is None]
            if missing_t0:
                warnings.warn(
                    f"Sequences without a T0 value: {missing_t0}. "
                    "Accessing these items with temporal_encoding='delay' will raise "
                    "a ValueError. Call pool.set_t0(...) to assign T0 values.",
                    UserWarning,
                    stacklevel=2,
                )

        # ------------------------------------------------------------------ #
        # 6. Build static lookup dict {seq_id → float32 array (S,)}          #
        # ------------------------------------------------------------------ #
        static_lookup: dict[Any, np.ndarray] = {}
        if static_names and static_df_raw is not None:
            for row in static_df_raw.iter_rows(named=True):
                sid = row[id_col]
                vals = _extract_static_values(row, static_names, static_feature_dims)
                static_lookup[sid] = vals

        label_lookup: dict[Any, Any] = {}
        if label_feature is not None and label_is_static:
            if static_df_raw is not None:
                for row in static_df_raw.iter_rows(named=True):
                    sid = row[id_col]
                    label_lookup[sid] = (
                        float("nan")
                        if row[label_feature] is None
                        else float(row[label_feature])
                    )

        # ------------------------------------------------------------------ #
        # 7. Store collected data                                             #
        # ------------------------------------------------------------------ #
        # Keep only the columns we need in the entity DataFrame
        keep_cols = [id_col] + temporal_cols + entity_names
        if label_feature is not None and not label_is_static:
            keep_cols.append(label_feature)
        self._entity_df = entity_df.select(
            [c for c in keep_cols if c in entity_df.columns]
        )

        self._id_col = id_col
        self._temporal_cols = temporal_cols
        self._is_datetime = is_datetime
        self._temporal_encoding = temporal_encoding
        self._entity_names = entity_names
        self._static_names = static_names
        self._label_feature = label_feature
        self._label_is_static = label_is_static
        self._fill_value = fill_value
        self._t0_dict = t0_dict
        self._static_lookup = static_lookup
        self._label_lookup = label_lookup

        # Public attributes
        self.feature_dims: dict[str, tuple[int, int]] = feature_dims
        self.static_feature_dims: dict[str, tuple[int, int]] | None = (
            static_feature_dims if static_names else None
        )

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Total number of entity rows across all sequences."""
        return len(self._entity_df)

    def __repr__(self) -> str:
        static_names = self._static_names or []
        return (
            f"{self.__class__.__name__}(len={len(self)}, "
            f"features={self._entity_names}, "
            f"static_features={static_names}, "
            f"temporal_encoding={self._temporal_encoding})"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single entity row as a dict of tensors.

        Keys:
            - ``'temporal'``: shape ``()`` for event, ``(2,)`` for interval/state.
            - ``'features'``: shape ``(F,)``.
            - ``'static'``: shape ``(S,)`` — absent if no static features.
            - ``'label'``: shape ``()`` — absent if ``label_feature`` is ``None``.

        Raises:
            IndexError: If idx is out of range.
            ValueError: If NaN is encountered and fill_value is nan.
            ValueError: If temporal_encoding='delay' and the sequence has no T0.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}."
            )

        row = self._entity_df[idx]
        seq_id = row[self._id_col][0]

        # Temporal
        if self._temporal_encoding == "delay":
            t0 = self._t0_dict.get(seq_id)
            if t0 is None:
                raise ValueError(
                    f"Sequence '{seq_id}' has no T0 value. "
                    "Call pool.set_t0(...) before constructing the dataset."
                )
        else:
            t0 = None

        temporal_arr = encode_temporal(
            row, self._temporal_cols, self._is_datetime, self._temporal_encoding, t0
        )
        temporal_tensor = to_tensor(temporal_arr)
        # encode_temporal returns (T,) or (T, 2). For EventDataset, T=1 always.
        # Squeeze: (1,) → (), (1, 2) → (2,)
        temporal_tensor = temporal_tensor.squeeze(0)

        # Entity features
        feat_arr = build_feature_array(row, self._entity_names, self.feature_dims)
        feat_arr = apply_fill_value(
            (
                feat_arr.reshape(-1).astype(np.float32)
                if feat_arr.ndim == 0
                else feat_arr.astype(np.float32)
            ),
            self._fill_value,
            seq_id,
            self._entity_names,
        )
        result: dict[str, Any] = {
            "temporal": temporal_tensor,
            "features": to_tensor(feat_arr),
        }

        # Static features
        if self._static_names and seq_id in self._static_lookup:
            static_arr = self._static_lookup[seq_id]
            static_arr = apply_fill_value(
                static_arr, self._fill_value, seq_id, self._static_names
            )
            result["static"] = to_tensor(static_arr)

        # Label
        if self._label_feature is not None:
            if self._label_is_static:
                label_val = self._label_lookup.get(seq_id, float("nan"))
                label_arr = np.array(label_val, dtype=np.float32)
            else:
                label_row = row[self._label_feature]
                label_arr = label_row.cast(pl.Float32).to_numpy().astype(np.float32)
                label_arr = label_arr[0] if len(label_arr) == 1 else label_arr
            label_arr = apply_fill_value(
                np.atleast_1d(np.array(label_arr, dtype=np.float32)),
                self._fill_value,
                seq_id,
                [self._label_feature],
            )
            result["label"] = torch.tensor(
                label_arr[0]
                if label_arr.ndim == 1 and len(label_arr) == 1
                else label_arr[0]
            )

        return result


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _extract_static_values(
    row: dict[str, Any],
    static_names: list[str],
    feature_dims: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Build a float32 (S,) array from a static row dict."""
    if not static_names:
        return np.zeros(0, dtype=np.float32)
    total_width = max(end for _, end in feature_dims.values())
    out = np.empty(total_width, dtype=np.float32)
    for name in static_names:
        start, end = feature_dims[name]
        val = row[name]
        if end - start == 1:
            out[start] = float("nan") if val is None else float(val)
        else:
            out[start:end] = np.array(val, dtype=np.float32)
    return out
