"""
Bounded Cache Module

LRU cache with optional TTL and JSON persistence. Built specifically for the
agent response caches (developer, critic) that previously used the unbounded
``core.memory.Memory`` and grew without limit.

Why not reuse ``Memory``? Memory is used by the orchestrator for session state
(last_prompt, last_tasks, last_test_run) where the key set is finite and
eviction would be wrong. The agent caches need a different policy: high-volume
keys (one per prompt+feedback combination), long-lived files, and no value in
keeping every entry forever.

Storage format on disk is forward-compatible with the old Memory JSON: a flat
``{key: value}`` dict, so existing files load cleanly. On save we add a tiny
``__cache_meta__`` envelope when TTL is active so timestamps survive restart.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Any


_META_KEY = "__cache_meta__"


class BoundedCache:
    """LRU cache with size cap, optional TTL, and JSON persistence.

    Args:
        filepath: Path to the JSON file backing this cache. None = in-memory only.
        max_size: Maximum number of entries. Oldest (LRU) are evicted on overflow.
            Default 1000.
        ttl_seconds: Optional time-to-live for entries. None disables TTL.
            Default None.

    Backwards compatibility:
        Old ``Memory`` JSON files (flat dict) load without modification. Their
        entries get current-time timestamps on first load; if you set TTL after
        upgrading, those entries will eventually age out as expected.
    """

    def __init__(
        self,
        filepath: str | None = None,
        max_size: int = 1000,
        ttl_seconds: int | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when set")

        self.filepath = filepath
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # OrderedDict preserves insertion order; we move keys to end on access
        # to implement LRU. Each value is stored as (timestamp, value).
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

        if filepath and os.path.exists(filepath):
            self._load()

    # ===========================================
    # Core operations
    # ===========================================

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._data.get(key)
        if entry is None:
            return default

        ts, value = entry
        if self._is_expired(ts):
            del self._data[key]
            self._save()
            return default

        # LRU bump: move to end
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        serialized = _serialize(value)
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.time(), serialized)

        # Evict oldest entries until we're under the size cap
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

        self._save()

    def delete(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
            self._save()

    def clear(self) -> None:
        self._data.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        # Respect TTL for membership testing
        return self.get(key, _MISSING) is not _MISSING

    # ===========================================
    # Maintenance
    # ===========================================

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns the number evicted.

        Useful to call once at startup or periodically; otherwise expired
        entries are evicted lazily on access.
        """
        if self.ttl_seconds is None:
            return 0
        evicted = 0
        for key in list(self._data.keys()):
            ts, _ = self._data[key]
            if self._is_expired(ts):
                del self._data[key]
                evicted += 1
        if evicted:
            self._save()
        return evicted

    # ===========================================
    # Persistence
    # ===========================================

    def _is_expired(self, timestamp: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    def _load(self) -> None:
        try:
            with open(self.filepath) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"BoundedCache: failed to load {self.filepath}: {e}")
            return

        if not isinstance(raw, dict):
            return

        meta = raw.get(_META_KEY)
        now = time.time()

        if isinstance(meta, dict) and meta.get("format") == "bounded_cache_v1":
            # Native format: each entry has a timestamp.
            entries = raw.get("entries", {})
            for key, payload in entries.items():
                if isinstance(payload, dict) and "ts" in payload and "v" in payload:
                    self._data[key] = (float(payload["ts"]), payload["v"])
                else:
                    # Defensive: malformed entry, treat as fresh
                    self._data[key] = (now, payload)
        else:
            # Legacy format: flat {key: value}. Stamp everything with now.
            for key, value in raw.items():
                if key == _META_KEY:
                    continue
                self._data[key] = (now, value)

        # Evict beyond cap if the loaded file was larger than our limit.
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def _save(self) -> None:
        if not self.filepath:
            return
        try:
            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            payload = {
                _META_KEY: {
                    "format": "bounded_cache_v1",
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl_seconds,
                },
                "entries": {
                    key: {"ts": ts, "v": value}
                    for key, (ts, value) in self._data.items()
                },
            }
            with open(self.filepath, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"BoundedCache: failed to save {self.filepath}: {e}")


# Sentinel for __contains__ that distinguishes "missing" from "stored None".
_MISSING: Any = object()


def _serialize(obj: Any) -> Any:
    """Recursive JSON-friendly conversion (mirrors core.memory.Memory behavior)."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj
