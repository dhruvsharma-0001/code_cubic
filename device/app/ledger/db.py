"""device/app/ledger/db.py — Backwards-compatible database initialization shim."""
from __future__ import annotations
from pathlib import Path
from typing import Any

from device.app.ledger.ledger import SQLiteLedger


def init_db(db_path: str | Path | None = None) -> SQLiteLedger:
    ledger = SQLiteLedger(db_path=db_path)
    ledger._init_db()
    return ledger
