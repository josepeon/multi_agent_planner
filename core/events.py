"""
Pipeline Events

Thread-safe pub/sub for streaming pipeline progress to consumers
(web SSE, CLI logger, anything else). Producers (agents, orchestrators) call
``emit(event)``; consumers subscribe with ``subscribe(job_id)`` and read
events off a queue.

Why a separate module: agents and orchestrators must not depend on Flask or
HTTP plumbing. They emit pipeline-domain events; the web layer turns those
into SSE frames. The same event stream powers CLI live mode and the eventual
WebSocket UI.

Event types (loose strings, evolved over time):

  - ``run_started``      payload: {prompt}
  - ``stage_started``    payload: {stage, agent}
  - ``stage_finished``   payload: {stage, agent, duration_s, ok, summary?}
  - ``node_started``     payload: {id, role}          (DAG only)
  - ``node_finished``    payload: {id, role, status, duration_s}  (DAG only)
  - ``task_passed``      payload: {task_id, attempts}
  - ``task_failed``      payload: {task_id, attempts, error?}
  - ``checkpoint``       payload: {stage, rendered}   (asking user)
  - ``checkpoint_decided`` payload: {stage, status}   ("approved"|"edited"|"regenerate")
  - ``test_run``         payload: as_dict() of TestRunResult
  - ``cost``             payload: snapshot of cost_tracker.to_dict()
  - ``run_finished``     payload: {duration_s, ok}
  - ``log``              payload: {message, level}   (free-form fallback)
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Sentinel pushed to signal end-of-stream for SSE consumers.
END_OF_STREAM = object()


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "seq": self.seq,
        }


class EventBus:
    """One bus per process. Multiplexes events per ``job_id``.

    - Producers call ``emit(job_id, event)``; bus appends to the job's queue
      and fans out to live subscribers.
    - Consumers call ``subscribe(job_id)`` to get an iterator that yields
      events until ``end(job_id)`` is called.
    """

    def __init__(self, history_cap: int = 500, max_jobs: int = 50) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, list[queue.Queue]] = {}
        self._history: dict[str, list[Event]] = {}
        self._seq: dict[str, int] = {}
        self._ended: set[str] = set()
        self._history_cap = history_cap
        self._max_jobs = max_jobs

    # ---- producer ----

    def emit(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> Event:
        # seq assignment, history append, and queue snapshot must be one
        # atomic block: with concurrent producers (parallel DAG nodes), doing
        # them under separate lock acquisitions lets a higher-seq event land
        # in history before a lower-seq one.
        with self._lock:
            self._seq[job_id] = self._seq.get(job_id, 0) + 1
            event = Event(type=event_type, payload=payload or {}, seq=self._seq[job_id])

            hist = self._history.setdefault(job_id, [])
            hist.append(event)
            if len(hist) > self._history_cap:
                # Keep most recent N
                self._history[job_id] = hist[-self._history_cap :]

            queues = list(self._queues.get(job_id, []))

        for q in queues:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # consumer is slow; drop rather than block producer

        return event

    def end(self, job_id: str) -> None:
        """Signal that no more events will be emitted for ``job_id``.

        Live subscribers receive END_OF_STREAM and exit their loop.
        Idempotent: safe to call from both the pipeline and its supervisor.
        Ended jobs beyond ``max_jobs`` have their history evicted (oldest
        first) so a long-lived server doesn't grow without bound.
        """
        with self._lock:
            queues = list(self._queues.pop(job_id, []))
            self._ended.add(job_id)

            if len(self._history) > self._max_jobs:
                # dict preserves insertion order → oldest jobs first
                for jid in list(self._history):
                    if len(self._history) <= self._max_jobs:
                        break
                    if jid in self._ended and jid != job_id:
                        self._history.pop(jid, None)
                        self._seq.pop(jid, None)
                        self._ended.discard(jid)

        for q in queues:
            try:
                q.put_nowait(END_OF_STREAM)
            except queue.Full:
                pass

    # ---- consumer ----

    def subscribe(self, job_id: str, replay: bool = True):
        """Iterator of events for ``job_id``. Blocks waiting for more.

        Yields ``Event`` instances and exits cleanly when ``end()`` is called.
        With ``replay=True`` (default), already-emitted events are yielded
        first so a late subscriber gets the full history. Subscribing to a
        job that already ended replays its history and returns immediately
        instead of blocking forever.
        """
        with self._lock:
            history = list(self._history.get(job_id, [])) if replay else []
            if job_id in self._ended:
                q = None
            else:
                q = queue.Queue(maxsize=1000)
                self._queues.setdefault(job_id, []).append(q)

        yield from history
        if q is None:
            return

        while True:
            item = q.get()
            if item is END_OF_STREAM:
                return
            yield item

    def history(self, job_id: str) -> list[Event]:
        with self._lock:
            return list(self._history.get(job_id, []))


# Module-level singleton + helper
_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def emit(job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> Event:
    """Convenience: emit to the module-level bus."""
    return _bus.emit(job_id, event_type, payload)


def ends_bus(fn):
    """Decorator for pipeline entry points: always signal end-of-stream for
    the run's job_id, even when the pipeline raises. Without this, a crashed
    run left SSE subscribers blocked on a queue forever. ``end`` is
    idempotent, so the pipeline's own success-path ``end`` call is fine.

    The wrapped function must take ``job_id`` as a keyword argument.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            jid = kwargs.get("job_id")
            if jid:
                _bus.end(jid)

    return wrapper
