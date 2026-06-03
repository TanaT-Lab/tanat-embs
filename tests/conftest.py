#!/usr/bin/env python3
"""
Shared session-scoped fixtures for the TanaT-embs test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tanat import set_workspace, get_workspace
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool

if TYPE_CHECKING:
    from tanat.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Absolute path to tests/data/ (exposes static/, datetime/, timestep/ subdirs)."""
    return Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """Isolated workspace in a temp directory, cleared and torn down after the session.

    autouse=True ensures this runs before any other session-scoped fixture
    (including the pool builders), so all stores land in the temp workspace,
    never in ~/.tanat_workspace.
    """
    ws_path = tmp_path_factory.mktemp("tanat_workspace")
    set_workspace(str(ws_path))
    ws = get_workspace()
    ws.clear()
    return ws


# ---------------------------------------------------------------------------
# Sequence pools: datetime variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_store(data_dir: Path) -> Path:
    """Store path for the datetime EventSequencePool (uses start as event time)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    return (
        EventSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
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
        .build("event_pool")
    )


@pytest.fixture(scope="session")
def event_pool(event_store: Path) -> EventSequencePool:
    """EventSequencePool built from datetime/ (uses start as event time)."""
    return EventSequencePool(store=event_store)


@pytest.fixture(scope="session")
def interval_store(data_dir: Path) -> Path:
    """Store path for the datetime IntervalSequencePool."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    return (
        IntervalSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "status", "flag_valid", "token_emb"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .build("interval_pool")
    )


@pytest.fixture(scope="session")
def interval_pool(interval_store: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from datetime/."""
    return IntervalSequencePool(store=interval_store)


# ---------------------------------------------------------------------------
# Sequence pools: timestep variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def state_store_ts(data_dir: Path) -> Path:
    """Store path for the timestep IntervalSequencePool.

    Uses IntervalSequencePool because the timestep data has overlapping intervals
    (non-contiguous), which is incompatible with StateSequencePool continuity checks.
    """
    seq = data_dir / "timestep"
    sta = data_dir / "static"
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
        .build("interval_pool_ts")
    )


@pytest.fixture(scope="session")
def state_pool_ts(state_store_ts: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from timestep/ (numeric temporal index)."""
    return IntervalSequencePool(store=state_store_ts)
