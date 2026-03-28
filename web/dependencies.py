from __future__ import annotations

from web.services.data_store import DataStore
from web.services.feedback_db import FeedbackDB

# Singletons set during app lifespan
_store: DataStore | None = None
_db: FeedbackDB | None = None


def set_store(store: DataStore) -> None:
    global _store
    _store = store


def set_db(db: FeedbackDB) -> None:
    global _db
    _db = db


def get_store() -> DataStore:
    assert _store is not None, "DataStore not initialized"
    return _store


def get_db() -> FeedbackDB:
    assert _db is not None, "FeedbackDB not initialized"
    return _db
