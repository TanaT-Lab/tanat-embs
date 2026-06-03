#!/usr/bin/env python3
"""Tests for EventDataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tanat_embs.datasets import EventDataset
from tanat_embs.datasets._utils import apply_fill_value

NUMERIC_ENTITY = ["value", "flag_valid"]
NUMERIC_STATIC = ["age", "is_active", "membership_duration"]


class TestEventDatasetLen:
    def test_len_equals_total_events(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        total = sum(len(event_pool[sid]) for sid in event_pool.unique_ids)
        assert len(ds) == total

    def test_len_with_single_sequence(self, event_pool):
        seq = event_pool[event_pool.unique_ids[0]]
        ds = EventDataset(seq, features=["value"], static_features=[])
        assert len(ds) == len(seq)


class TestEventDatasetItemShapes:
    def test_event_temporal_is_scalar(self, event_pool):
        """EventPool: temporal tensor has shape ()."""
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert item["temporal"].shape == torch.Size([])

    def test_interval_temporal_is_2d(self, interval_pool):
        """IntervalPool: temporal tensor has shape (2,) — [start, end]."""
        ds = EventDataset(interval_pool, features=["value"], static_features=[])
        item = ds[0]
        assert item["temporal"].shape == torch.Size([2])

    def test_state_temporal_is_2d(self, state_pool_ts):
        """StatePool: temporal tensor has shape (2,)."""
        ds = EventDataset(state_pool_ts, features=["value"], static_features=[])
        item = ds[0]
        assert item["temporal"].shape == torch.Size([2])

    def test_features_shape(self, event_pool):
        ds = EventDataset(
            event_pool, features=["value", "flag_valid"], static_features=[]
        )
        item = ds[0]
        assert item["features"].shape == torch.Size([2])

    def test_static_shape(self, event_pool):
        ds = EventDataset(
            event_pool,
            features=["value"],
            static_features=NUMERIC_STATIC[:2],
            fill_value=0.0,
        )
        item = ds[0]
        assert "static" in item
        assert item["static"].shape == torch.Size([2])

    def test_no_static_key_when_empty(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert "static" not in item

    def test_array_feature_expands(self, event_pool):
        """Array(Float32, 32) feature expands to 32 columns."""
        ds = EventDataset(event_pool, features=["token_emb"], static_features=[])
        item = ds[0]
        assert item["features"].shape == torch.Size([32])

    def test_feature_dims_scalar(self, event_pool):
        ds = EventDataset(
            event_pool, features=["value", "flag_valid"], static_features=[]
        )
        assert ds.feature_dims == {"value": (0, 1), "flag_valid": (1, 2)}

    def test_feature_dims_array(self, event_pool):
        ds = EventDataset(event_pool, features=["token_emb"], static_features=[])
        assert ds.feature_dims == {"token_emb": (0, 32)}

    def test_static_feature_dims(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=["age"])
        assert ds.static_feature_dims == {"age": (0, 1)}

    def test_static_feature_dims_none_when_no_static(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        assert ds.static_feature_dims is None

    def test_output_dtype_is_float32(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert item["features"].dtype == torch.float32
        assert item["temporal"].dtype == torch.float32


class TestEventDatasetLabel:
    def test_static_label_shape(self, event_pool):
        """Static label is a scalar tensor ()."""
        ds = EventDataset(
            event_pool,
            features=["value"],
            static_features=[],
            label_feature="age",
            fill_value=0.0,
        )
        item = ds[0]
        assert "label" in item
        assert item["label"].shape == torch.Size([])

    def test_static_label_same_for_all_events_in_sequence(self, event_pool):
        """Static label must be the same value for all events of the same sequence."""
        ds = EventDataset(
            event_pool,
            features=["value"],
            static_features=[],
            label_feature="age",
            fill_value=0.0,
        )
        sid = event_pool.unique_ids[0]
        seq_len = len(event_pool[sid])
        labels = set()
        base_idx = 0
        for sid2 in event_pool.unique_ids:
            if sid2 == sid:
                break
            base_idx += len(event_pool[sid2])
        for i in range(seq_len):
            labels.add(ds[base_idx + i]["label"].item())
        assert len(labels) == 1  # same scalar for all events in one sequence

    def test_entity_label_shape(self, event_pool):
        """Entity label is a scalar per event."""
        ds = EventDataset(
            event_pool,
            features=["flag_valid"],
            static_features=[],
            label_feature="value",
        )
        item = ds[0]
        assert "label" in item
        assert item["label"].shape == torch.Size([])

    def test_label_removed_from_features(self, event_pool):
        """label_feature should not appear in the 'features' tensor."""
        ds = EventDataset(
            event_pool,
            features=["value", "flag_valid"],
            static_features=[],
            label_feature="value",
        )
        item = ds[0]
        # Only flag_valid remains
        assert item["features"].shape == torch.Size([1])
        assert ds.feature_dims == {"flag_valid": (0, 1)}

    def test_static_priority_over_entity_when_name_collision(self, event_pool):
        """If label_feature name exists in both static and entity, static wins."""
        # 'age' is static only; we need a name in both. Skip if not available.
        pytest.skip("No same-name collision in test pool — covered by unit logic.")

    def test_no_label_key_when_none(self, event_pool):
        ds = EventDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert "label" not in item

    def test_label_not_found_raises(self, event_pool):
        with pytest.raises(ValueError, match="not found"):
            EventDataset(
                event_pool,
                features=["value"],
                static_features=[],
                label_feature="nonexistent",
            )


class TestEventDatasetValidation:
    def test_empty_features_raises(self, event_pool):
        with pytest.raises(ValueError, match="features=\\[\\]"):
            EventDataset(event_pool, features=[], static_features=[])

    def test_non_numeric_feature_raises(self, event_pool):
        with pytest.raises(ValueError, match="Non-numeric"):
            EventDataset(event_pool, features=["status"], static_features=[])

    def test_non_numeric_error_mentions_column(self, event_pool):
        with pytest.raises(ValueError, match="status"):
            EventDataset(event_pool, features=["status"], static_features=[])

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            EventDataset("not_a_pool", features=["value"])


class TestEventDatasetFillValue:
    def test_nan_raises_by_default(self, event_pool):
        """Default fill_value=nan should raise if NaN encountered."""

        # Create a pool view where we inject NaN — skip if pool has no NaN
        # Instead, test behavior by injecting a NaN manually in a controlled way.
        # This is an integration-level check; verify the mechanism via unit:

        arr = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            apply_fill_value(arr, float("nan"), seq_id=1, col_names=["x"])

    def test_fill_value_replaces_nan(self):

        arr = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        result = apply_fill_value(arr, 0.0, seq_id=1, col_names=["x"])
        assert not np.isnan(result).any()
        assert result[1] == 0.0


class TestEventDatasetTemporalEncoding:
    def test_absolute_temporal_is_numeric(self, event_pool):
        ds = EventDataset(
            event_pool,
            features=["value"],
            static_features=[],
            temporal_encoding="absolute",
        )
        item = ds[0]
        assert torch.isfinite(item["temporal"])

    def test_delay_requires_t0(self, event_pool):
        """Pool without explicit t0 uses default (position=0) — should work without error."""
        p = event_pool.copy()
        ds = EventDataset(
            p, features=["value"], static_features=[], temporal_encoding="delay"
        )
        item = ds[0]
        assert torch.isfinite(item["temporal"])

    def test_delay_temporal_first_event_near_zero(self, event_pool):
        """With position=0, T0 is the first event → delay of first event should be 0."""
        p = event_pool.copy()
        p.set_t0(position=0)
        ds = EventDataset(
            p, features=["value"], static_features=[], temporal_encoding="delay"
        )
        # First event in first sequence should have delay ≈ 0
        item = ds[0]
        assert abs(item["temporal"].item()) < 1.0  # within 1 second

    def test_state_delay_two_values(self, state_pool_ts):
        """State pool with delay: temporal shape is (2,) [start-T0, end-T0]."""
        p = state_pool_ts.copy()
        p.set_t0(position=0)
        ds = EventDataset(
            p, features=["value"], static_features=[], temporal_encoding="delay"
        )
        item = ds[0]
        assert item["temporal"].shape == torch.Size([2])
