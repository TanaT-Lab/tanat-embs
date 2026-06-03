#!/usr/bin/env python3
"""
SequenceDataset: sequence-level PyTorch Dataset wrapping a TanaT SequencePool.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from tanat.sequence.base.pool import SequencePool
from tanat.zeroing.base import _T0

from ._utils import (
    apply_fill_value,
    build_feature_array,
    encode_temporal,
    to_tensor,
    validate_and_build_feature_dims,
)
from .event_dataset import _extract_static_values


class SequenceDataset(Dataset):
    """A PyTorch Dataset where each item is a full sequence from a TanaT SequencePool.

    Variable-length sequences can be padded to a common length using ``max_len``.
    When padding is enabled, a boolean mask and the actual sequence length are
    included in each item dict.

    All data is collected eagerly at construction time — no reference to the
    original pool is retained.

    Args:
        pool: Any ``SequencePool`` (event, interval, or state).
        features: Entity feature columns to include.
            ``None`` = all entity features. ``[]`` raises ``ValueError``.
        static_features: Static feature columns to include.
            ``None`` = all static features. ``[]`` = no static features.
        label_feature: Static feature name to use as the label.
            Removed from the static feature list. Must be a static feature;
            providing an entity feature name raises ``ValueError``.
            ``None`` = no label.
        max_len: Fixed sequence length. Shorter sequences are zero-padded;
            longer sequences are truncated from the end.
            ``None`` (default) = no padding; tensors have shape ``(T, ...)``.
        fill_value: How to handle NaN in numeric features.
            ``float('nan')`` (default) raises ``ValueError`` on any NaN.
            Any numeric value silently fills NaNs.
        temporal_encoding: ``'absolute'`` returns raw temporal values cast to
            float32. ``'delay'`` returns the signed offset from each sequence's
            T0 (in seconds for datetime sequences).
    """

    def __init__(
        self,
        pool: SequencePool,
        features: list[str] | None = None,
        static_features: list[str] | None = None,
        label_feature: str | None = None,
        max_len: int | None = None,
        fill_value: float = float("nan"),
        temporal_encoding: Literal["absolute", "delay"] = "absolute",
    ) -> None:
        if not isinstance(pool, SequencePool):
            raise TypeError(f"Expected SequencePool, got {type(pool).__name__}.")

        metadata = pool.metadata
        settings = pool.settings
        id_col = settings.id_column
        temporal_cols = settings.get_time_columns()
        is_datetime = metadata.time_index.is_datetime

        # ------------------------------------------------------------------ #
        # 1. Resolve feature lists                                            #
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
            static_names = list(static_features)

        # ------------------------------------------------------------------ #
        # 2. Resolve label_feature (static only)                             #
        # ------------------------------------------------------------------ #
        if label_feature is not None:
            if label_feature not in all_static_names:
                if label_feature in all_entity_names:
                    raise ValueError(
                        f"label_feature='{label_feature}' is an entity feature. "
                        "SequenceDataset only supports static labels. "
                        "Use EventDataset if you need per-event labels."
                    )
                raise ValueError(
                    f"label_feature='{label_feature}' not found in static features."
                )
            if label_feature in static_names:
                static_names.remove(label_feature)

        # ------------------------------------------------------------------ #
        # 3. Validate feature types and build dimension maps                  #
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
            validate_and_build_feature_dims(metadata, [label_feature], is_static=True)

        # ------------------------------------------------------------------ #
        # 4. Collect data from pool                                           #
        # ------------------------------------------------------------------ #
        entity_df = pool.temporal_data(fmt="polars")
        static_df_raw = pool.static_data(fmt="polars")

        if temporal_encoding == "delay":
            t0_df = pool.t0_data(fmt="polars")
            t0_dict: dict[Any, Any] = dict(
                zip(t0_df[id_col].to_list(), t0_df[_T0].to_list())
            )
            missing_t0 = [str(sid) for sid, t0 in t0_dict.items() if t0 is None]
            if missing_t0:
                warnings.warn(
                    f"Sequences without a T0 value: {missing_t0}. "
                    "Accessing these items with temporal_encoding='delay' will raise "
                    "a ValueError. Call pool.set_t0(...) to assign T0 values.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            t0_dict = {}

        # ------------------------------------------------------------------ #
        # 5. Build per-sequence group index                                   #
        # ------------------------------------------------------------------ #
        # Keep only columns we need
        keep_entity = [id_col] + temporal_cols + entity_names
        entity_df = entity_df.select([c for c in keep_entity if c in entity_df.columns])

        # Group row indices by seq_id (preserving pool order)
        seq_ids = pool.unique_ids
        id_series = entity_df[id_col]
        seq_row_ranges: dict[Any, tuple[int, int]] = {}
        pos = 0
        for sid in seq_ids:
            length = (id_series == sid).sum()
            seq_row_ranges[sid] = (pos, pos + length)
            pos += length

        # Static lookup
        static_lookup: dict[Any, np.ndarray] = {}
        label_lookup: dict[Any, float] = {}

        if static_df_raw is not None:
            for row in static_df_raw.iter_rows(named=True):
                sid = row[id_col]
                if static_names:
                    static_lookup[sid] = _extract_static_values(
                        row, static_names, static_feature_dims
                    )
                if label_feature is not None:
                    label_lookup[sid] = (
                        float("nan")
                        if row[label_feature] is None
                        else float(row[label_feature])
                    )

        # ------------------------------------------------------------------ #
        # 6. Store state                                                      #
        # ------------------------------------------------------------------ #
        self._entity_df = entity_df
        self._id_col = id_col
        self._temporal_cols = temporal_cols
        self._is_datetime = is_datetime
        self._temporal_encoding = temporal_encoding
        self._entity_names = entity_names
        self._static_names = static_names
        self._label_feature = label_feature
        self._fill_value = fill_value
        self._max_len = max_len
        self._t0_dict = t0_dict
        self._static_lookup = static_lookup
        self._label_lookup = label_lookup
        self._seq_ids = seq_ids
        self._seq_row_ranges = seq_row_ranges

        self.feature_dims: dict[str, tuple[int, int]] = feature_dims
        self.static_feature_dims: dict[str, tuple[int, int]] | None = (
            static_feature_dims if static_names else None
        )

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Number of sequences in the pool."""
        return len(self._seq_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a full sequence as a dict of tensors.

        Without padding (``max_len=None``):
            - ``'id'``: original sequence ID.
            - ``'temporal'``: shape ``(T,)`` for event, ``(T, 2)`` for interval/state.
            - ``'features'``: shape ``(T, F)``.
            - ``'static'``: shape ``(S,)`` — absent if no static features.
            - ``'label'``: shape ``()`` — absent if ``label_feature`` is ``None``.

        With padding (``max_len`` set):
            Same as above, plus:
            - ``'mask'``: shape ``(max_len,)`` bool, ``True`` at valid positions.
            - ``'length'``: actual sequence length before padding/truncation.
            Tensors are zero-padded (or truncated) to ``(max_len, ...)``.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}."
            )

        seq_id = self._seq_ids[idx]
        start, end = self._seq_row_ranges[seq_id]
        seq_df = self._entity_df[start:end]
        T = len(seq_df)

        # T0 for delay encoding
        if self._temporal_encoding == "delay":
            t0 = self._t0_dict.get(seq_id)
            if t0 is None:
                raise ValueError(
                    f"Sequence '{seq_id}' has no T0 value. "
                    "Call pool.set_t0(...) before constructing the dataset."
                )
        else:
            t0 = None

        # Temporal: (T,) or (T, 2)
        temporal_arr = encode_temporal(
            seq_df, self._temporal_cols, self._is_datetime, self._temporal_encoding, t0
        )

        # Features: (T, F)
        feat_arr = build_feature_array(seq_df, self._entity_names, self.feature_dims)
        if feat_arr.ndim == 1 and T > 1:
            feat_arr = feat_arr.reshape(T, -1)
        feat_arr = feat_arr.astype(np.float32)
        feat_arr = apply_fill_value(
            feat_arr, self._fill_value, seq_id, self._entity_names
        )

        # Padding / truncation
        if self._max_len is not None:
            L = self._max_len
            actual_T = min(T, L)

            # Truncate if needed
            temporal_arr = (
                temporal_arr[:actual_T]
                if temporal_arr.ndim == 1
                else temporal_arr[:actual_T]
            )
            feat_arr = (
                feat_arr[:actual_T] if feat_arr.ndim == 2 else feat_arr[:actual_T]
            )

            # Build padded arrays
            if temporal_arr.ndim == 1:
                temporal_padded = np.zeros(L, dtype=np.float32)
                temporal_padded[:actual_T] = temporal_arr
            else:
                temporal_padded = np.zeros((L, temporal_arr.shape[1]), dtype=np.float32)
                temporal_padded[:actual_T] = temporal_arr

            F = feat_arr.shape[1] if feat_arr.ndim == 2 else 0
            if feat_arr.ndim == 2:
                feat_padded = np.zeros((L, F), dtype=np.float32)
                feat_padded[:actual_T] = feat_arr
            else:
                feat_padded = np.zeros((L, 0), dtype=np.float32)

            mask = np.zeros(L, dtype=bool)
            mask[:actual_T] = True

            result: dict[str, Any] = {
                "id": seq_id,
                "temporal": to_tensor(temporal_padded),
                "features": to_tensor(feat_padded),
                "mask": torch.from_numpy(mask),
                "length": actual_T,
            }
        else:
            result = {
                "id": seq_id,
                "temporal": to_tensor(temporal_arr.astype(np.float32)),
                "features": to_tensor(feat_arr),
            }

        # Static
        if self._static_names and seq_id in self._static_lookup:
            static_arr = self._static_lookup[seq_id]
            static_arr = apply_fill_value(
                static_arr, self._fill_value, seq_id, self._static_names
            )
            result["static"] = to_tensor(static_arr)

        # Label (static scalar)
        if self._label_feature is not None:
            label_val = self._label_lookup.get(seq_id, float("nan"))
            label_arr = np.array(label_val, dtype=np.float32)
            label_arr = apply_fill_value(
                np.atleast_1d(label_arr),
                self._fill_value,
                seq_id,
                [self._label_feature],
            )
            result["label"] = torch.tensor(label_arr[0])

        return result
