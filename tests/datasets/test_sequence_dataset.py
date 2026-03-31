#!/usr/bin/env python3
"""Tests for SequenceDataset."""

from __future__ import annotations

import pytest
import torch

from tanat_embs.datasets import SequenceDataset


class TestSequenceDatasetLen:
    def test_len_equals_number_of_sequences(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        assert len(ds) == len(event_pool)

    def test_len_state_pool(self, state_pool_ts):
        ds = SequenceDataset(state_pool_ts, features=["value"], static_features=[])
        assert len(ds) == len(state_pool_ts)


class TestSequenceDatasetItemShapesNoPadding:
    def test_id_present(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert "id" in item

    def test_temporal_shape_event(self, event_pool):
        """Event pool: temporal shape is (T,) where T = actual sequence length."""
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        sid = item["id"]
        T = len(event_pool[sid])
        assert item["temporal"].shape == torch.Size([T])

    def test_temporal_shape_state(self, state_pool_ts):
        """State pool: temporal shape is (T, 2) — [start, end] per step."""
        ds = SequenceDataset(state_pool_ts, features=["value"], static_features=[])
        item = ds[0]
        sid = item["id"]
        T = len(state_pool_ts[sid])
        assert item["temporal"].shape == torch.Size([T, 2])

    def test_features_shape(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value", "flag_valid"], static_features=[])
        item = ds[0]
        sid = item["id"]
        T = len(event_pool[sid])
        assert item["features"].shape == torch.Size([T, 2])

    def test_static_shape(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=["age"])
        item = ds[0]
        assert item["static"].shape == torch.Size([1])

    def test_no_mask_without_max_len(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert "mask" not in item
        assert "length" not in item

    def test_no_static_key_when_empty(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert "static" not in item

    def test_dtype_float32(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        item = ds[0]
        assert item["features"].dtype == torch.float32
        assert item["temporal"].dtype == torch.float32


class TestSequenceDatasetItemShapesWithPadding:
    def test_temporal_shape_padded(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=20)
        item = ds[0]
        assert item["temporal"].shape == torch.Size([20])

    def test_temporal_shape_padded_state(self, state_pool_ts):
        ds = SequenceDataset(state_pool_ts, features=["value"], static_features=[], max_len=20)
        item = ds[0]
        assert item["temporal"].shape == torch.Size([20, 2])

    def test_features_shape_padded(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value", "flag_valid"], static_features=[], max_len=20)
        item = ds[0]
        assert item["features"].shape == torch.Size([20, 2])

    def test_mask_present_with_max_len(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=20)
        item = ds[0]
        assert "mask" in item
        assert "length" in item
        assert item["mask"].dtype == torch.bool
        assert item["mask"].shape == torch.Size([20])

    def test_mask_true_at_valid_positions(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=20)
        item = ds[0]
        length = item["length"]
        assert item["mask"][:length].all()
        assert not item["mask"][length:].any()

    def test_padded_positions_are_zero(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=20)
        item = ds[0]
        length = item["length"]
        if length < 20:
            assert (item["features"][length:] == 0).all()
            assert (item["temporal"][length:] == 0).all()

    def test_truncation_to_max_len(self, event_pool):
        """Sequences longer than max_len are truncated."""
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=3)
        item = ds[0]
        assert item["length"] <= 3
        assert item["mask"].shape == torch.Size([3])

    def test_length_capped_at_max_len(self, event_pool):
        """length field should not exceed max_len."""
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], max_len=3)
        for i in range(len(ds)):
            assert ds[i]["length"] <= 3


class TestSequenceDatasetLabel:
    def test_static_label_is_scalar(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[], label_feature="age")
        item = ds[0]
        assert "label" in item
        assert item["label"].shape == torch.Size([])

    def test_static_label_removed_from_static_features(self, event_pool):
        """label_feature='age' removes 'age' from static features."""
        ds = SequenceDataset(
            event_pool,
            features=["value"],
            static_features=["age", "membership_duration"],
            label_feature="age",
        )
        assert ds.static_feature_dims is not None
        assert "age" not in ds.static_feature_dims
        assert "membership_duration" in ds.static_feature_dims
        item = ds[0]
        assert item["static"].shape == torch.Size([1])

    def test_entity_label_raises(self, event_pool):
        """Entity feature used as label_feature should raise ValueError."""
        with pytest.raises(ValueError, match="entity feature"):
            SequenceDataset(event_pool, features=["flag_valid"], static_features=[], label_feature="value")

    def test_label_not_found_raises(self, event_pool):
        with pytest.raises(ValueError, match="not found"):
            SequenceDataset(event_pool, features=["value"], static_features=[], label_feature="nonexistent")

    def test_no_label_key_when_none(self, event_pool):
        ds = SequenceDataset(event_pool, features=["value"], static_features=[])
        assert "label" not in ds[0]


class TestSequenceDatasetValidation:
    def test_empty_features_raises(self, event_pool):
        with pytest.raises(ValueError, match="features=\\[\\]"):
            SequenceDataset(event_pool, features=[], static_features=[])

    def test_non_pool_input_raises(self):
        with pytest.raises(TypeError):
            SequenceDataset("not_a_pool", features=["value"])

    def test_non_numeric_feature_raises(self, event_pool):
        with pytest.raises(ValueError, match="Non-numeric"):
            SequenceDataset(event_pool, features=["status"], static_features=[])


class TestSequenceDatasetTemporalEncoding:
    def test_delay_encoding_works(self, state_pool_ts):
        p = state_pool_ts.copy()
        p.set_t0(position=0)
        ds = SequenceDataset(p, features=["value"], static_features=[], temporal_encoding="delay")
        item = ds[0]
        assert item["temporal"].shape[0] > 0

    def test_delay_first_state_near_zero(self, state_pool_ts):
        """With t0 at position=0, the start of the first state ≈ 0."""
        p = state_pool_ts.copy()
        p.set_t0(position=0, anchor="start")
        ds = SequenceDataset(p, features=["value"], static_features=[], temporal_encoding="delay")
        item = ds[0]
        # temporal is (T, 2); first row start should be very close to 0
        assert abs(item["temporal"][0][0].item()) < 1.0
