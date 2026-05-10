"""T2.3: pipeline event bus — emission, replay, multiple subscribers, end-of-stream."""

from __future__ import annotations

import threading
import time

import pytest

from core.events import END_OF_STREAM, EventBus, get_bus, new_job_id


# ===========================================
# Basics
# ===========================================

class TestEventBasics:
    def test_emit_assigns_sequence(self):
        bus = EventBus()
        e1 = bus.emit("job1", "stage_started", {"stage": "planner"})
        e2 = bus.emit("job1", "stage_finished", {"stage": "planner"})
        assert e1.seq == 1
        assert e2.seq == 2

    def test_history_returned_in_order(self):
        bus = EventBus()
        bus.emit("job1", "a")
        bus.emit("job1", "b")
        bus.emit("job1", "c")
        events = bus.history("job1")
        assert [e.type for e in events] == ["a", "b", "c"]

    def test_history_separate_per_job(self):
        bus = EventBus()
        bus.emit("j1", "x")
        bus.emit("j2", "y")
        assert [e.type for e in bus.history("j1")] == ["x"]
        assert [e.type for e in bus.history("j2")] == ["y"]


# ===========================================
# Replay
# ===========================================

class TestReplay:
    def test_subscriber_gets_history_then_live(self):
        bus = EventBus()
        bus.emit("j", "first")
        bus.emit("j", "second")

        collected = []
        done = threading.Event()

        def consume():
            for e in bus.subscribe("j"):
                collected.append(e.type)
                if e.type == "third":
                    done.set()

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        # Give the consumer a moment to yield historical items
        time.sleep(0.05)
        bus.emit("j", "third")
        assert done.wait(2.0)
        bus.end("j")
        t.join(timeout=2.0)
        assert collected[:3] == ["first", "second", "third"]

    def test_subscriber_can_skip_replay(self):
        bus = EventBus()
        bus.emit("j", "before")

        collected = []
        done = threading.Event()

        def consume():
            for e in bus.subscribe("j", replay=False):
                collected.append(e.type)
                if e.type == "live":
                    done.set()

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        time.sleep(0.05)
        bus.emit("j", "live")
        assert done.wait(2.0)
        bus.end("j")
        t.join(timeout=2.0)
        assert collected == ["live"]


# ===========================================
# Multiple subscribers
# ===========================================

class TestFanout:
    def test_two_subscribers_each_get_all_events(self):
        bus = EventBus()

        results: dict[int, list[str]] = {1: [], 2: []}
        ready = threading.Barrier(3)
        done1 = threading.Event()
        done2 = threading.Event()

        def consumer(idx, done_evt):
            ready.wait()
            for e in bus.subscribe("j", replay=False):
                results[idx].append(e.type)
                if e.type == "z":
                    done_evt.set()

        t1 = threading.Thread(target=consumer, args=(1, done1), daemon=True)
        t2 = threading.Thread(target=consumer, args=(2, done2), daemon=True)
        t1.start()
        t2.start()
        ready.wait()
        # Brief pause to ensure both subscribers are inside their loops
        time.sleep(0.05)
        bus.emit("j", "x")
        bus.emit("j", "y")
        bus.emit("j", "z")
        assert done1.wait(2.0) and done2.wait(2.0)
        bus.end("j")
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        assert results[1] == ["x", "y", "z"]
        assert results[2] == ["x", "y", "z"]


# ===========================================
# End of stream
# ===========================================

class TestEnd:
    def test_end_terminates_subscribers(self):
        bus = EventBus()
        bus.emit("j", "only")

        results = []

        def consume():
            for e in bus.subscribe("j"):
                results.append(e.type)

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        time.sleep(0.05)
        bus.end("j")
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert results == ["only"]


# ===========================================
# Singleton helpers
# ===========================================

class TestModuleSingleton:
    def test_get_bus_returns_consistent_instance(self):
        assert get_bus() is get_bus()

    def test_new_job_id_is_unique(self):
        assert new_job_id() != new_job_id()
