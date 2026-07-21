"""B3: BoundedCache — LRU + TTL + legacy-JSON compatibility."""

from __future__ import annotations

import json
import time

import pytest

from core.cache import BoundedCache

# ===========================================
# Construction
# ===========================================


class TestConstruction:
    def test_rejects_non_positive_max_size(self):
        with pytest.raises(ValueError):
            BoundedCache(max_size=0)

    def test_rejects_non_positive_ttl(self):
        with pytest.raises(ValueError):
            BoundedCache(ttl_seconds=0)

    def test_in_memory_only(self):
        cache = BoundedCache(filepath=None, max_size=10)
        cache.set("k", "v")
        assert cache.get("k") == "v"


# ===========================================
# LRU eviction
# ===========================================


class TestLRU:
    def test_eviction_at_size_cap(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"), max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert len(cache) == 3

    def test_get_bumps_to_most_recent(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"), max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to make it most-recently used
        assert cache.get("a") == 1
        cache.set("d", 4)  # should evict "b" now, not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_set_existing_key_does_not_evict(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"), max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 99)  # update, not insert
        assert len(cache) == 2
        assert cache.get("a") == 99
        assert cache.get("b") == 2


# ===========================================
# TTL
# ===========================================


class TestTTL:
    def test_no_ttl_means_no_expiry(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"), ttl_seconds=None)
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_get_returns_none_for_expired(self, tmp_path, monkeypatch):
        cache = BoundedCache(str(tmp_path / "c.json"), ttl_seconds=10)
        cache.set("k", "v")
        # Move time forward past the TTL
        future = time.time() + 100
        monkeypatch.setattr("core.cache.time.time", lambda: future)
        assert cache.get("k") is None

    def test_evict_expired_clears_them(self, tmp_path, monkeypatch):
        cache = BoundedCache(str(tmp_path / "c.json"), ttl_seconds=10)
        cache.set("a", 1)
        cache.set("b", 2)
        future = time.time() + 100
        monkeypatch.setattr("core.cache.time.time", lambda: future)
        evicted = cache.evict_expired()
        assert evicted == 2
        assert len(cache) == 0

    def test_contains_respects_ttl(self, tmp_path, monkeypatch):
        cache = BoundedCache(str(tmp_path / "c.json"), ttl_seconds=10)
        cache.set("k", None)
        assert "k" in cache  # stored value is None but key exists
        future = time.time() + 100
        monkeypatch.setattr("core.cache.time.time", lambda: future)
        assert "k" not in cache


# ===========================================
# Persistence
# ===========================================


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "c.json"
        c1 = BoundedCache(str(path), max_size=5)
        c1.set("a", {"nested": [1, 2, 3]})
        c1.set("b", "hello")

        c2 = BoundedCache(str(path), max_size=5)
        assert c2.get("a") == {"nested": [1, 2, 3]}
        assert c2.get("b") == "hello"

    def test_legacy_flat_json_loads(self, tmp_path):
        # Simulate the old core.memory.Memory format: flat {key: value}
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"x": 1, "y": "two"}))

        cache = BoundedCache(str(path))
        assert cache.get("x") == 1
        assert cache.get("y") == "two"

    def test_load_truncates_to_max_size(self, tmp_path):
        path = tmp_path / "big.json"
        path.write_text(json.dumps({f"k{i}": i for i in range(50)}))

        cache = BoundedCache(str(path), max_size=10)
        assert len(cache) == 10

    def test_save_uses_v1_envelope(self, tmp_path):
        path = tmp_path / "c.json"
        cache = BoundedCache(str(path), max_size=5, ttl_seconds=60)
        cache.set("a", 1)
        with open(path) as f:
            saved = json.load(f)
        assert saved["__cache_meta__"]["format"] == "bounded_cache_v1"
        assert saved["entries"]["a"]["v"] == 1


# ===========================================
# Other ops
# ===========================================


class TestOps:
    def test_delete(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"))
        cache.set("k", "v")
        cache.delete("k")
        assert cache.get("k") is None
        cache.delete("nonexistent")  # no-op, no error

    def test_clear(self, tmp_path):
        cache = BoundedCache(str(tmp_path / "c.json"))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_dataclass_serialization(self, tmp_path):
        from dataclasses import dataclass

        @dataclass
        class Foo:
            x: int
            y: str

        cache = BoundedCache(str(tmp_path / "c.json"))
        cache.set("foo", Foo(x=1, y="hi"))
        assert cache.get("foo") == {"x": 1, "y": "hi"}


# ===========================================
# Thread safety
# ===========================================


class TestThreadSafety:
    def test_concurrent_get_set_no_corruption(self, tmp_path):
        """Best-of-N and parallel DAG nodes hammer one shared cache; the
        OrderedDict must not be mutated mid-iteration and the JSON file must
        stay parseable."""
        import json
        import threading

        path = tmp_path / "cache.json"
        cache = BoundedCache(filepath=str(path), max_size=50)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    cache.set(f"k{worker_id}_{i % 20}", {"v": i})
                    cache.get(f"k{(worker_id + 1) % 4}_{i % 20}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # File must be intact JSON after the storm
        raw = json.loads(path.read_text())
        assert raw["__cache_meta__"]["format"] == "bounded_cache_v1"
        assert len(cache) <= 50

    def test_save_is_atomic_no_tmp_leftovers(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = BoundedCache(filepath=str(path))
        cache.set("a", 1)
        leftovers = [p for p in path.parent.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        assert path.exists()
