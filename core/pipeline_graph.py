"""
Pipeline DAG

Replaces the linear ``Planner → Architect → Developer×N → QA → Integrator → Tests+Docs``
pipeline with a dependency graph. Nodes declare which earlier-node outputs they
need; the executor batches anything with no remaining dependencies into the
same parallel layer.

What this unlocks:

1. **Parallel module development.** Independent dev tasks run concurrently
   instead of sequentially in a ``for task in tasks:`` loop.
2. **Conditional nodes.** Nodes carry an optional ``predicate`` callable.
   Returning False marks the node skipped without breaking downstream nodes
   that listed it as a dependency — the skipped node's "output" is simply the
   sentinel ``SKIPPED``, which downstream nodes can check.
3. **Re-planning hooks.** A node may return ``Replan(new_nodes=[...])`` to
   inject more nodes into the graph mid-run (e.g. when QA discovers the
   architecture is wrong and the Architect needs to redesign one module).
4. **Mixed in-process / cross-thread execution.** Nodes opt into thread-pool
   execution; the rest run on the calling thread.

Design notes:

- This module knows nothing about the agents themselves. It's a pure scheduler
  that takes ``PipelineNode`` objects with a ``run(inputs) -> output`` callable.
  Agents become thin adapters built on top in ``core.orchestrator_dag``.
- The graph is built dynamically: nodes can be added before ``execute()`` is
  called, and re-planning during execution can add more.
- Cycle detection runs on the static structure at execute() start; replan
  insertions are checked on the fly.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# ===========================================
# Sentinels & helper types
# ===========================================

class _Skipped:
    """Sentinel returned in place of a value for skipped nodes."""

    _instance: "_Skipped | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<SKIPPED>"

    def __bool__(self) -> bool:
        return False


SKIPPED = _Skipped()


@dataclass
class Replan:
    """Returned from a node's run() to inject new nodes into the graph.

    The new nodes can depend on already-completed nodes or on each other.
    They're added to the unscheduled set and picked up on the next layer.
    """

    new_nodes: list["PipelineNode"]


# ===========================================
# Node
# ===========================================

NodeId = str
RunCallable = Callable[[dict[str, Any]], Any]
PredicateCallable = Callable[[dict[str, Any]], bool]


@dataclass
class PipelineNode:
    """A unit of work in the pipeline graph.

    Args:
        id: Unique id within the graph. Used to reference this node from
            ``depends_on`` of other nodes.
        run: Callable taking ``{dep_id: output_of_that_dep}`` and returning
            this node's output. May raise (failure is captured in the result).
            May return a ``Replan(...)`` to add new nodes.
        depends_on: Ids this node needs the output of before it can run.
        parallel: If True, eligible to run in the thread pool. If False, runs
            synchronously on the executor's calling thread.
        predicate: Optional gate; called with the same inputs as ``run``. When
            it returns False the node is recorded as skipped (output = SKIPPED).
        role: Optional human label for cost-attribution / logging.
    """

    id: NodeId
    run: RunCallable
    depends_on: list[NodeId] = field(default_factory=list)
    parallel: bool = True
    predicate: PredicateCallable | None = None
    role: str | None = None


@dataclass
class NodeResult:
    node_id: NodeId
    status: str  # "success" | "skipped" | "failed"
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    layer: int = -1


@dataclass
class GraphResult:
    """End-to-end outcome of a graph execution."""

    nodes: dict[NodeId, NodeResult] = field(default_factory=dict)
    layers_executed: int = 0
    total_duration_seconds: float = 0.0

    def output_of(self, node_id: NodeId) -> Any:
        return self.nodes[node_id].output

    def succeeded(self, node_id: NodeId) -> bool:
        r = self.nodes.get(node_id)
        return r is not None and r.status == "success"

    def all_succeeded(self) -> bool:
        return all(r.status != "failed" for r in self.nodes.values())


# ===========================================
# Graph
# ===========================================

class PipelineGraph:
    """Mutable DAG of pipeline nodes."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, PipelineNode] = {}
        self._lock = threading.Lock()

    def add(self, node: PipelineNode) -> "PipelineGraph":
        with self._lock:
            if node.id in self._nodes:
                raise ValueError(f"Node id already exists: {node.id}")
            self._nodes[node.id] = node
        return self

    def add_many(self, nodes: Iterable[PipelineNode]) -> "PipelineGraph":
        for n in nodes:
            self.add(n)
        return self

    def has(self, node_id: NodeId) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    # ------ static checks ------

    def _validate(self) -> None:
        # Every dep must exist
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Node '{node.id}' depends on unknown node '{dep}'"
                    )
        # Cycle check via Kahn's algorithm
        indeg = {nid: 0 for nid in self._nodes}
        for n in self._nodes.values():
            for d in n.depends_on:
                indeg[n.id] += 1
        ready = [nid for nid, n in indeg.items() if n == 0]
        seen = 0
        while ready:
            current = ready.pop()
            seen += 1
            for n in self._nodes.values():
                if current in n.depends_on:
                    indeg[n.id] -= 1
                    if indeg[n.id] == 0:
                        ready.append(n.id)
        if seen != len(self._nodes):
            unresolved = [nid for nid, deg in indeg.items() if deg > 0]
            raise ValueError(f"Cycle detected: cannot order {unresolved}")

    # ------ execution ------

    def execute(
        self,
        max_workers: int = 4,
        on_node_start: Callable[[PipelineNode], None] | None = None,
        on_node_finish: Callable[[NodeResult], None] | None = None,
    ) -> GraphResult:
        """Run the graph to completion.

        Returns a ``GraphResult`` with one entry per node. A node can finish
        in three states:

        - ``success``: ``run`` returned a non-Replan value
        - ``skipped``: predicate returned False, or all deps were skipped/failed
        - ``failed``: ``run`` raised, or a transitive dep failed
        """
        self._validate()

        result = GraphResult()
        completed: set[NodeId] = set()
        outputs: dict[NodeId, Any] = {}
        layer = 0
        run_start = time.time()

        while len(completed) < len(self._nodes):
            ready = self._ready_nodes(completed)
            if not ready:
                # No progress possible — mark remaining nodes as failed.
                for nid, node in self._nodes.items():
                    if nid not in completed:
                        result.nodes[nid] = NodeResult(
                            node_id=nid,
                            status="failed",
                            error="dependencies could not be satisfied",
                            layer=layer,
                        )
                        completed.add(nid)
                break

            # Mark a node skipped if any required dep was failed.
            # (Skipped deps don't propagate; only failed.)
            to_skip: list[PipelineNode] = []
            to_run: list[PipelineNode] = []
            for node in ready:
                if any(
                    result.nodes[d].status == "failed"
                    for d in node.depends_on
                    if d in result.nodes
                ):
                    to_skip.append(node)
                else:
                    to_run.append(node)

            for node in to_skip:
                node_result = NodeResult(
                    node_id=node.id,
                    status="skipped",
                    output=SKIPPED,
                    error="upstream dependency failed",
                    layer=layer,
                )
                result.nodes[node.id] = node_result
                outputs[node.id] = SKIPPED
                completed.add(node.id)
                if on_node_finish:
                    on_node_finish(node_result)

            replans: list[Replan] = []
            layer_executed = self._execute_layer(
                to_run,
                outputs,
                layer,
                result,
                completed,
                replans,
                max_workers,
                on_node_start,
                on_node_finish,
            )

            # Apply any replans before the next layer
            for replan in replans:
                for new_node in replan.new_nodes:
                    self.add(new_node)

            if to_run or to_skip:
                layer += 1

            # Safety: if a layer ran nothing AND nothing was skipped AND no
            # replans, we'd loop forever. Bail.
            if not to_run and not to_skip and not replans and layer_executed == 0:
                break

        result.layers_executed = layer
        result.total_duration_seconds = time.time() - run_start
        return result

    def _ready_nodes(self, completed: set[NodeId]) -> list[PipelineNode]:
        ready = []
        for node in self._nodes.values():
            if node.id in completed:
                continue
            if all(d in completed for d in node.depends_on):
                ready.append(node)
        return ready

    def _execute_layer(
        self,
        nodes: list[PipelineNode],
        outputs: dict[NodeId, Any],
        layer: int,
        result: GraphResult,
        completed: set[NodeId],
        replans: list[Replan],
        max_workers: int,
        on_node_start: Callable[[PipelineNode], None] | None,
        on_node_finish: Callable[[NodeResult], None] | None,
    ) -> int:
        if not nodes:
            return 0

        parallel_nodes = [n for n in nodes if n.parallel]
        sync_nodes = [n for n in nodes if not n.parallel]

        ran = 0

        # Sync first (cheap; respects ordering for simple roots)
        for node in sync_nodes:
            r = self._run_node(node, outputs, layer, on_node_start)
            self._record(r, node, outputs, result, completed, replans, on_node_finish)
            ran += 1

        # Then parallel batch
        if parallel_nodes:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(parallel_nodes))) as pool:
                futures = {
                    pool.submit(self._run_node, node, outputs, layer, on_node_start): node
                    for node in parallel_nodes
                }
                for fut in as_completed(futures):
                    node = futures[fut]
                    r = fut.result()
                    self._record(r, node, outputs, result, completed, replans, on_node_finish)
                    ran += 1

        return ran

    def _run_node(
        self,
        node: PipelineNode,
        outputs: dict[NodeId, Any],
        layer: int,
        on_node_start: Callable[[PipelineNode], None] | None,
    ) -> NodeResult:
        if on_node_start:
            try:
                on_node_start(node)
            except Exception:
                pass

        inputs = {dep: outputs[dep] for dep in node.depends_on}

        if node.predicate is not None:
            try:
                if not node.predicate(inputs):
                    return NodeResult(
                        node_id=node.id,
                        status="skipped",
                        output=SKIPPED,
                        layer=layer,
                    )
            except Exception as e:
                return NodeResult(
                    node_id=node.id,
                    status="failed",
                    error=f"predicate raised: {type(e).__name__}: {e}",
                    layer=layer,
                )

        start = time.time()
        try:
            output = node.run(inputs)
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                duration_seconds=time.time() - start,
                layer=layer,
            )

        return NodeResult(
            node_id=node.id,
            status="success",
            output=output,
            duration_seconds=time.time() - start,
            layer=layer,
        )

    def _record(
        self,
        r: NodeResult,
        node: PipelineNode,
        outputs: dict[NodeId, Any],
        result: GraphResult,
        completed: set[NodeId],
        replans: list[Replan],
        on_node_finish: Callable[[NodeResult], None] | None,
    ) -> None:
        if isinstance(r.output, Replan):
            replans.append(r.output)
            r.output = None  # don't expose the Replan as a downstream input

        result.nodes[r.node_id] = r
        outputs[r.node_id] = r.output if r.status == "success" else SKIPPED
        completed.add(r.node_id)
        if on_node_finish:
            try:
                on_node_finish(r)
            except Exception:
                pass
