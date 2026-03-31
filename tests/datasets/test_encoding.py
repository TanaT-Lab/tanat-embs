#!/usr/bin/env python3
"""Tests for CategoricalEncoder and CategoricalDecoder."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tanat_embs.datasets import CategoricalEncoder, CategoricalDecoder


class TestCategoricalEncoderFitTransform:
    def test_onehot_adds_array_column(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], encoder="onehot", scope="entity")
        encoded = enc.fit_transform(event_pool)
        assert "status_enc" in encoded.metadata.entity_features or any(
            f.name == "status_enc" for f in encoded.metadata.entity_features
        )

    def test_label_adds_int64_column(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], encoder="label", scope="entity")
        encoded = enc.fit_transform(event_pool)
        meta = encoded.metadata
        info = meta.feature_info("status_enc", is_static=False)
        assert info is not None
        assert "Int64" in info.dtype

    def test_static_column_encoded(self, event_pool):
        enc = CategoricalEncoder(columns=["group"], encoder="label", scope="static")
        encoded = enc.fit_transform(event_pool)
        info = encoded.metadata.feature_info("group_enc", is_static=True)
        assert info is not None

    def test_drop_original_removes_source_column(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", drop_original=True, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        entity_names = [f.name for f in encoded.metadata.entity_features]
        assert "status" not in entity_names

    def test_original_retained_when_not_dropped(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", drop_original=False, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        entity_names = [f.name for f in encoded.metadata.entity_features]
        assert "status" in entity_names
        assert "status_enc" in entity_names

    def test_custom_suffix(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", suffix="_idx", scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        entity_names = [f.name for f in encoded.metadata.entity_features]
        assert "status_idx" in entity_names

    def test_input_pool_not_modified(self, event_pool):
        entity_names_before = [f.name for f in event_pool.metadata.entity_features]
        enc = CategoricalEncoder(columns=["status"], encoder="label", scope="entity")
        enc.fit_transform(event_pool)
        entity_names_after = [f.name for f in event_pool.metadata.entity_features]
        assert entity_names_before == entity_names_after

    def test_auto_detect_categorical_columns(self, event_pool):
        """columns=None should auto-detect String/Categorical columns."""
        enc = CategoricalEncoder(encoder="label", scope="entity")
        enc.fit(event_pool)
        assert "status" in enc.fitted_state["columns"]


class TestCategoricalEncoderState:
    def test_is_fitted_false_before_fit(self, event_pool):
        enc = CategoricalEncoder()
        assert enc.is_fitted is False

    def test_is_fitted_true_after_fit(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], scope="entity")
        enc.fit(event_pool)
        assert enc.is_fitted is True

    def test_call_before_fit_raises(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], scope="entity")
        with pytest.raises(RuntimeError, match="fitted"):
            enc(event_pool)

    def test_fitted_state_dict(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], encoder="label", scope="entity")
        enc.fit(event_pool)
        state = enc.fitted_state
        assert isinstance(state, dict)
        assert "encodings" in state
        assert "status" in state["encodings"]

    def test_fitted_state_raises_if_not_fitted(self):
        enc = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = enc.fitted_state

    def test_fit_returns_self(self, event_pool):
        enc = CategoricalEncoder(columns=["status"], scope="entity")
        result = enc.fit(event_pool)
        assert result is enc


class TestCategoricalEncoderValidation:
    def test_nonexistent_column_raises(self, event_pool):
        enc = CategoricalEncoder(columns=["nonexistent"], scope="entity")
        with pytest.raises(ValueError):
            enc.fit(event_pool)

    def test_numeric_column_raises(self, event_pool):
        """Trying to encode a numeric column should raise ValueError."""
        enc = CategoricalEncoder(columns=["value"], scope="entity")
        with pytest.raises(ValueError):
            enc.fit(event_pool)


class TestCategoricalDecoder:
    def test_decoder_from_fitted_encoder(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", drop_original=True, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        dec = CategoricalDecoder(enc)
        restored = dec(encoded)
        entity_names = [f.name for f in restored.metadata.entity_features]
        assert "status" in entity_names
        assert "status_enc" not in entity_names

    def test_decoder_from_fitted_state_dict(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", drop_original=True, scope="entity"
        )
        enc.fit(event_pool)
        encoded = enc(event_pool)
        dec = CategoricalDecoder(enc.fitted_state)
        restored = dec(encoded)
        entity_names = [f.name for f in restored.metadata.entity_features]
        assert "status" in entity_names

    def test_onehot_decode_removes_encoded_col(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="onehot", drop_original=True, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        dec = CategoricalDecoder(enc)
        restored = dec(encoded)
        entity_names = [f.name for f in restored.metadata.entity_features]
        assert "status_enc" not in entity_names

    def test_onehot_decode_reconstructs_original(self, event_pool):
        enc = CategoricalEncoder(
            columns=["status"], encoder="onehot", drop_original=True, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        dec = CategoricalDecoder(enc)
        restored = dec(encoded)
        # Original 'status' column should be present and contain string values
        entity_names = [f.name for f in restored.metadata.entity_features]
        assert "status" in entity_names

    def test_label_roundtrip_values(self, event_pool):
        """Decode should approximately recover the original categorical values."""
        orig_values = (
            event_pool.sequence_data(features=["status"], output_format="polars")["status"]
            .to_list()
        )
        enc = CategoricalEncoder(
            columns=["status"], encoder="label", drop_original=True, scope="entity"
        )
        encoded = enc.fit_transform(event_pool)
        dec = CategoricalDecoder(enc)
        restored = dec(encoded)
        restored_values = (
            restored.sequence_data(features=["status"], output_format="polars")["status"]
            .to_list()
        )
        assert orig_values == restored_values

    def test_unfitted_encoder_raises(self):
        enc = CategoricalEncoder()
        with pytest.raises(ValueError, match="unfitted"):
            CategoricalDecoder(enc)

    def test_static_column_decode(self, event_pool):
        enc = CategoricalEncoder(
            columns=["group"], encoder="label", drop_original=True, scope="static"
        )
        encoded = enc.fit_transform(event_pool)
        dec = CategoricalDecoder(enc)
        restored = dec(encoded)
        static_names = [f.name for f in restored.metadata.static_features]
        assert "group" in static_names
        assert "group_enc" not in static_names
