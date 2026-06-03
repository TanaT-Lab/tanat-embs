#!/usr/bin/env python3
"""
CategoricalEncoder / CategoricalDecoder for TanaT SequencePools.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tanat.metadata.feature import CategoricalInfo, StringInfo
from tanat.sequence.base.pool import SequencePool


def _get_categorical_columns(
    pool: SequencePool, scope: Literal["entity", "static", "both"]
) -> list[tuple[str, bool]]:
    """Return ``[(col_name, is_static), ...]`` for all categorical columns in the scope."""
    result = []
    if scope in ("entity", "both"):
        for info in pool.metadata.entity_features:
            if isinstance(info, (CategoricalInfo, StringInfo)):
                result.append((info.name, False))
    if scope in ("static", "both"):
        if pool.metadata.static_features:
            for info in pool.metadata.static_features:
                if isinstance(info, (CategoricalInfo, StringInfo)):
                    result.append((info.name, True))
    return result


class CategoricalEncoder:
    """Encodes categorical columns in a TanaT SequencePool to numeric vectors.

    Follows a sklearn-style fit/transform API. The encoder must be fitted
    before calling it.

    Args:
        columns: Columns to encode. ``None`` = auto-detect all
            ``Categorical``, ``String``, and ``Enum`` columns within *scope*.
        encoder: Encoding strategy:
            - ``'onehot'``: sklearn ``OneHotEncoder`` → ``Array(Float32, K)`` column.
            - ``'label'``: sklearn ``LabelEncoder`` → ``Int64`` scalar column.
            - A pre-fitted sklearn-compatible estimator can also be passed directly.
        drop_original: If ``True``, the original categorical columns are removed
            from the returned pool.
        suffix: Postfix appended to the original column name to form the new
            encoded column name. Default ``'_enc'``.
        scope: Which feature set to target:
            ``'entity'``, ``'static'``, or ``'both'``. Default ``'both'``.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        encoder: Literal["onehot", "label"] | BaseEstimator = "onehot",
        drop_original: bool = False,
        suffix: str = "_enc",
        scope: Literal["entity", "static", "both"] = "both",
    ) -> None:
        self._columns = columns
        self._encoder_spec = encoder
        self._drop_original = drop_original
        self._suffix = suffix
        self._scope = scope
        self._fitted_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, pool: SequencePool) -> "CategoricalEncoder":
        """Fit the encoder on the pool's data.

        Args:
            pool: Input SequencePool to fit on.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If a specified column does not exist or is not categorical.
        """
        if not isinstance(pool, SequencePool):
            raise TypeError(f"Expected SequencePool, got {type(pool).__name__}.")

        # Resolve columns to encode
        if self._columns is None:
            col_list = _get_categorical_columns(pool, self._scope)
        else:
            col_list = self._resolve_explicit_columns(pool, self._columns)

        if not col_list:
            self._fitted_state = {
                "columns": [],
                "scope": self._scope,
                "suffix": self._suffix,
                "drop_original": self._drop_original,
                "encodings": {},
            }
            return self

        # Collect values and fit one sklearn encoder per column
        entity_df = pool.temporal_data(fmt="polars")
        static_df = pool.static_data(fmt="polars")

        encodings: dict[str, dict] = {}
        for col_name, is_static in col_list:
            values = self._get_column_values(col_name, is_static, entity_df, static_df)
            sk_enc = self._make_sklearn_encoder()
            if isinstance(sk_enc, LabelEncoder):
                sk_enc.fit(values)
                output_dtype = "Int64"
                n_categories = len(sk_enc.classes_)
            else:
                sk_enc.fit(values.reshape(-1, 1))
                output_dtype = f"Array(Float32, {len(sk_enc.categories_[0])})"
                n_categories = len(sk_enc.categories_[0])

            # Store original Polars dtype for decoder reconstruction
            if is_static and static_df is not None:
                orig_dtype = static_df.schema[col_name]
            else:
                orig_dtype = entity_df.schema[col_name]

            encodings[col_name] = {
                "sklearn_encoder": sk_enc,
                "is_static": is_static,
                "output_dtype": output_dtype,
                "n_categories": n_categories,
                "original_dtype": orig_dtype,
            }

        self._fitted_state = {
            "columns": [c for c, _ in col_list],
            "scope": self._scope,
            "suffix": self._suffix,
            "drop_original": self._drop_original,
            "encodings": encodings,
        }
        return self

    def fit_transform(self, pool: SequencePool) -> SequencePool:
        """Fit on *pool* and immediately transform it.

        Equivalent to ``encoder.fit(pool)(pool)``.
        """
        return self.fit(pool)(pool)

    # ------------------------------------------------------------------ #
    # Transform                                                            #
    # ------------------------------------------------------------------ #

    def __call__(self, pool: SequencePool) -> SequencePool:
        """Apply the fitted encoding to *pool*.

        Returns a new pool with encoded columns added (and originals optionally
        removed). The input pool is not modified.

        Raises:
            RuntimeError: If the encoder has not been fitted yet.
        """
        if self._fitted_state is None:
            raise RuntimeError(
                "CategoricalEncoder must be fitted before calling. "
                "Use fit(pool) or fit_transform(pool) first."
            )
        new_pool = pool.copy()
        encodings = self._fitted_state["encodings"]
        suffix = self._fitted_state["suffix"]
        drop_original = self._fitted_state["drop_original"]
        id_col = pool.settings.id_column

        if not encodings:
            return new_pool

        entity_df = pool.temporal_data(fmt="polars")
        static_df = pool.static_data(fmt="polars")

        for col_name, enc_info in encodings.items():
            is_static = enc_info["is_static"]
            sk_enc = enc_info["sklearn_encoder"]
            enc_col_name = col_name + suffix

            values = self._get_column_values(col_name, is_static, entity_df, static_df)

            if isinstance(sk_enc, LabelEncoder):
                encoded = sk_enc.transform(values).astype(np.int64)
                if is_static:
                    ids = static_df[id_col].to_list()
                    enc_series = pl.Series(
                        enc_col_name, encoded.tolist(), dtype=pl.Int64
                    )
                    enc_df = pl.DataFrame({id_col: ids, enc_col_name: enc_series})
                    new_pool.add_static_features(enc_df)
                else:
                    enc_df = pl.DataFrame(
                        {
                            enc_col_name: pl.Series(
                                enc_col_name, encoded.tolist(), dtype=pl.Int64
                            )
                        }
                    )
                    new_pool.add_entity_features(enc_df)
            else:
                # OneHotEncoder
                dense = sk_enc.transform(values.reshape(-1, 1)).astype(np.float32)
                K = dense.shape[1]
                rows = dense.tolist()
                if is_static:
                    ids = static_df[id_col].to_list()
                    enc_series = pl.Series(
                        enc_col_name, rows, dtype=pl.Array(pl.Float32, K)
                    )
                    enc_df = pl.DataFrame({id_col: ids, enc_col_name: enc_series})
                    new_pool.add_static_features(enc_df)
                else:
                    enc_series = pl.Series(
                        enc_col_name, rows, dtype=pl.Array(pl.Float32, K)
                    )
                    enc_df = pl.DataFrame({enc_col_name: enc_series})
                    new_pool.add_entity_features(enc_df)

            if drop_original:
                new_pool.drop_features([col_name], is_static=is_static)

        return new_pool

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called."""
        return self._fitted_state is not None

    @property
    def fitted_state(self) -> dict[str, Any]:
        """The internal fitted state dict.

        Can be passed to ``CategoricalDecoder`` for decoupled usage or
        after serialization.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if self._fitted_state is None:
            raise RuntimeError("CategoricalEncoder has not been fitted yet.")
        return self._fitted_state

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _make_sklearn_encoder(self) -> BaseEstimator:
        spec = self._encoder_spec
        if isinstance(spec, str):
            if spec == "label":
                return LabelEncoder()
            if spec == "onehot":
                return OneHotEncoder(sparse_output=False)
            raise ValueError(f"Unknown encoder spec '{spec}'. Use 'label' or 'onehot'.")
        # Pre-fitted or unfitted estimator instance — clone it for independence

        return clone(spec)

    def _resolve_explicit_columns(
        self, pool: SequencePool, columns: list[str]
    ) -> list[tuple[str, bool]]:
        """Resolve explicit column names to (name, is_static) pairs.

        If a column exists in both entity and static, both are encoded.
        """
        all_entity = {f.name for f in pool.metadata.entity_features}
        all_static = (
            {f.name for f in pool.metadata.static_features}
            if pool.metadata.static_features
            else set()
        )
        entity_schema = {
            name: pool.metadata.feature_info(name, is_static=False)
            for name in all_entity
        }
        static_schema = {
            name: pool.metadata.feature_info(name, is_static=True)
            for name in all_static
        }

        result = []
        for col in columns:
            found = False
            if self._scope in ("entity", "both") and col in all_entity:
                info = entity_schema[col]
                if not isinstance(info, (CategoricalInfo, StringInfo)):
                    raise ValueError(
                        f"Column '{col}' is not a categorical/string feature "
                        f"(got {type(info).__name__})."
                    )
                result.append((col, False))
                found = True
            if self._scope in ("static", "both") and col in all_static:
                info = static_schema[col]
                if not isinstance(info, (CategoricalInfo, StringInfo)):
                    raise ValueError(
                        f"Static column '{col}' is not a categorical/string feature "
                        f"(got {type(info).__name__})."
                    )
                result.append((col, True))
                found = True
            if not found:
                raise ValueError(
                    f"Column '{col}' not found in "
                    f"{'entity or static' if self._scope == 'both' else self._scope} features."
                )
        return result

    @staticmethod
    def _get_column_values(
        col_name: str,
        is_static: bool,
        entity_df: pl.DataFrame | None,
        static_df: pl.DataFrame | None,
    ) -> np.ndarray:
        """Extract all values of *col_name* as a 1D numpy array of strings."""
        if is_static:
            if static_df is None:
                raise ValueError(
                    f"Pool has no static features; cannot encode '{col_name}'."
                )
            return static_df[col_name].cast(pl.String).to_numpy()
        if entity_df is None:
            raise ValueError(f"Pool has no entity data; cannot encode '{col_name}'.")
        return entity_df[col_name].cast(pl.String).to_numpy()


# --------------------------------------------------------------------------- #


class CategoricalDecoder:
    """Reverses a CategoricalEncoder transformation on a SequencePool.

    Removes encoded columns (``{original_name}{suffix}``) and reconstructs
    the original categorical columns.

    Args:
        encoder: A **fitted** ``CategoricalEncoder`` instance, or a
            ``fitted_state`` dict obtained from ``encoder.fitted_state``.

    Raises:
        ValueError: If an unfitted encoder is passed.
    """

    def __init__(self, encoder: "CategoricalEncoder | dict") -> None:
        if isinstance(encoder, CategoricalEncoder):
            if not encoder.is_fitted:
                raise ValueError(
                    "Cannot create CategoricalDecoder from an unfitted CategoricalEncoder. "
                    "Call encoder.fit(pool) first."
                )
            self._state = encoder.fitted_state
        elif isinstance(encoder, dict):
            self._state = encoder
        else:
            raise TypeError(
                f"Expected CategoricalEncoder or dict, got {type(encoder).__name__}."
            )

    def __call__(self, pool: SequencePool) -> SequencePool:
        """Reverse the encoding on *pool*.

        Removes encoded columns and reconstructs the original categorical
        columns. The input pool is not modified — a new pool is returned.

        For one-hot encoding, argmax is used to select the category (best-effort:
        all-zero rows and ties resolve to the first maximum).

        Raises:
            ValueError: If an encoded column is not found in the pool.
        """
        new_pool = pool.copy()
        encodings = self._state["encodings"]
        suffix = self._state["suffix"]

        if not encodings:
            return new_pool

        id_col = pool.settings.id_column
        entity_df = pool.temporal_data(fmt="polars")
        static_df = pool.static_data(fmt="polars")

        for col_name, enc_info in encodings.items():
            is_static = enc_info["is_static"]
            sk_enc = enc_info["sklearn_encoder"]
            enc_col_name = col_name + suffix

            # Check the encoded column is present
            if is_static:
                available = set(
                    pool.metadata.static_features
                    and [f.name for f in pool.metadata.static_features]
                    or []
                )
            else:
                available = {f.name for f in pool.metadata.entity_features}

            if enc_col_name not in available:
                raise ValueError(
                    f"Encoded column '{enc_col_name}' not found in pool. "
                    "Make sure you are decoding a pool that was produced by this encoder."
                )

            # Check if original is already present (drop_original=False case)
            original_present = col_name in available

            if not original_present:
                # Reconstruct the original column
                if isinstance(sk_enc, LabelEncoder):
                    if is_static:
                        int_vals = static_df[enc_col_name].to_numpy().astype(np.int64)
                    else:
                        int_vals = entity_df[enc_col_name].to_numpy().astype(np.int64)
                    categories = sk_enc.inverse_transform(int_vals)
                else:
                    # OneHotEncoder — argmax to pick category
                    if is_static:
                        raw = np.array(
                            static_df[enc_col_name].to_list(), dtype=np.float32
                        )
                    else:
                        raw = np.array(
                            entity_df[enc_col_name].to_list(), dtype=np.float32
                        )
                    indices = np.argmax(raw, axis=1)
                    # Wrap indices for inverse_transform (expects 2D one-hot)
                    onehot = np.zeros_like(raw)
                    onehot[np.arange(len(indices)), indices] = 1.0
                    categories = sk_enc.inverse_transform(onehot).flatten()

                # Add original column back
                orig_series = pl.Series(col_name, categories.tolist()).cast(pl.String)
                if is_static:
                    ids = static_df[id_col].to_list()
                    rec_df = pl.DataFrame({id_col: ids, col_name: orig_series})
                    new_pool.add_static_features(rec_df)
                else:
                    rec_df = pl.DataFrame({col_name: orig_series})
                    new_pool.add_entity_features(rec_df)

            # Drop the encoded column
            new_pool.drop_features([enc_col_name], is_static=is_static)

        return new_pool
