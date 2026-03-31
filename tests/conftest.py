#!/usr/bin/env python3
"""
Shared pytest fixtures for TanaT-embs tests.

Pools are built from the TanaT submodule's test data. A session-scoped
workspace ensures stores land in a temp directory rather than ~/.tanat_workspace.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat import set_workspace, get_workspace
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

TANAT_DATA = Path(__file__).parent.parent / "TanaT" / "tests" / "data"


# --------------------------------------------------------------------------- #
# Workspace isolation                                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session", autouse=True)
def workspace(tmp_path_factory: pytest.TempPathFactory):
    """Isolated TanaT workspace in a temp directory for the whole test session."""
    ws_path = tmp_path_factory.mktemp("tanat_workspace")
    set_workspace(str(ws_path))
    ws = get_workspace()
    ws.clear()
    return ws


# --------------------------------------------------------------------------- #
# Helper: filtered parquet (only IDs present in static.csv)                   #
# --------------------------------------------------------------------------- #

def _filtered_parquet(src: Path, id_column: str, static_csv: Path, tmp: Path) -> Path:
    """Write a filtered copy of src parquet, keeping only IDs in static_csv.

    This ensures every sequence in the returned pool has a matching static row,
    preventing NaN static values in tests that check static feature shapes/values.
    """
    static_ids = pl.read_csv(static_csv)[id_column].to_list()
    df = pl.read_parquet(src).filter(pl.col(id_column).is_in(static_ids))
    out = tmp / src.name
    df.write_parquet(out)
    return out


# --------------------------------------------------------------------------- #
# Event pool (datetime temporal index)                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def event_store(workspace, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build an EventSequencePool store from TanaT datetime test data.

    The sequence parquet is pre-filtered to IDs that also appear in static.csv
    so that every sequence in the pool has a complete static row.
    """
    seq = TANAT_DATA / "datetime"
    sta = TANAT_DATA / "static"
    tmp = tmp_path_factory.mktemp("filtered_data")
    seq_path = _filtered_parquet(seq / "sequence_main.parquet", "id", sta / "static.csv", tmp)
    return (
        EventSequencePool.builder()
        .add_parquet(
            seq_path,
            id_column="id",
            time_column="start",
            features=["value", "status", "flag_valid", "token_emb"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .build("test_event_pool")
    )


@pytest.fixture(scope="session")
def event_pool(event_store: Path) -> EventSequencePool:
    """EventSequencePool with datetime temporal index, entity and static features."""
    return EventSequencePool(store=event_store)


# --------------------------------------------------------------------------- #
# Interval pool with timestep temporal index (2-column temporal)               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def state_store_ts(workspace) -> Path:
    """Build an IntervalSequencePool store from TanaT timestep test data.

    Uses IntervalSequencePool because the timestep data has overlapping intervals
    (non-contiguous), which is incompatible with StateSequencePool continuity checks.
    The 2-column temporal format (start, end) is identical for both pool types.
    """
    seq = TANAT_DATA / "timestep"
    sta = TANAT_DATA / "static"
    return (
        IntervalSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "status", "flag_valid"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .build("test_state_pool_ts")
    )


@pytest.fixture(scope="session")
def state_pool_ts(state_store_ts: Path) -> IntervalSequencePool:
    """IntervalSequencePool with numeric (timestep) temporal index (2-column temporal)."""
    return IntervalSequencePool(store=state_store_ts)


# --------------------------------------------------------------------------- #
# Interval pool (datetime temporal index)                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def interval_store(workspace) -> Path:
    """Build an IntervalSequencePool store from TanaT datetime test data."""
    seq = TANAT_DATA / "datetime"
    sta = TANAT_DATA / "static"
    return (
        IntervalSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "flag_valid"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "membership_duration"],
        )
        .build("test_interval_pool")
    )


@pytest.fixture(scope="session")
def interval_pool(interval_store: Path) -> IntervalSequencePool:
    """IntervalSequencePool with datetime temporal index."""
    return IntervalSequencePool(store=interval_store)
