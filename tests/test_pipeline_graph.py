"""T1.1: PipelineGraph — DAG executor with parallel layers, predicates, replan."""

from __future__ import annotations

import threading
import time

import pytest

from core.pipeline_graph import (
    SKIPPED,
    PipelineGraph,
    PipelineNode,
    Replan,
)

# ===========================================
# Static validation
# ===========================================


class TestValidation:
    def test_unknown_dependency_rejected(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1, depends_on=["nonexistent"]))
        with pytest.raises(ValueError, match="unknown node"):
            g.execute()

    def test_cycle_rejected(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1, depends_on=["b"]))
        g.add(PipelineNode(id="b", run=lambda i: 2, depends_on=["a"]))
        with pytest.raises(ValueError, match="Cycle"):
            g.execute()

    def test_duplicate_id_rejected(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1))
        with pytest.raises(ValueError, match="already exists"):
            g.add(PipelineNode(id="a", run=lambda i: 2))


# ===========================================
# Linear execution
# ===========================================


class TestLinearExecution:
    def test_simple_chain(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1))
        g.add(PipelineNode(id="b", run=lambda i: i["a"] + 1, depends_on=["a"]))
        g.add(PipelineNode(id="c", run=lambda i: i["b"] * 10, depends_on=["b"]))

        result = g.execute()
        assert result.all_succeeded()
        assert result.output_of("c") == 20

    def test_failure_propagates_as_skipped_downstream(self):
        g = PipelineGraph()

        def boom(_i):
            raise RuntimeError("intentional")

        g.add(PipelineNode(id="a", run=boom))
        g.add(PipelineNode(id="b", run=lambda i: 99, depends_on=["a"]))

        result = g.execute()
        assert result.nodes["a"].status == "failed"
        assert "intentional" in result.nodes["a"].error
        assert result.nodes["b"].status == "skipped"
        assert result.nodes["b"].error == "upstream dependency failed"


# ===========================================
# Parallel layers
# ===========================================


class TestParallelLayers:
    def test_parallel_runs_concurrently(self):
        # Three independent slow nodes — should run in roughly the time of one,
        # not three times that.
        delay = 0.15

        def slow(_i):
            time.sleep(delay)
            return threading.current_thread().ident

        g = PipelineGraph()
        for i in range(3):
            g.add(PipelineNode(id=f"n{i}", run=slow))

        start = time.time()
        result = g.execute(max_workers=4)
        elapsed = time.time() - start

        # Generous bound: should be well under 3*delay
        assert elapsed < delay * 2.0, f"parallel too slow: {elapsed:.2f}s"
        # All on different threads
        idents = {result.output_of(f"n{i}") for i in range(3)}
        assert len(idents) > 1

    def test_serial_node_blocks_layer(self):
        # One non-parallel node alongside parallel ones still works
        g = PipelineGraph()
        g.add(PipelineNode(id="serial", run=lambda i: "S", parallel=False))
        g.add(PipelineNode(id="par1", run=lambda i: "P1", parallel=True))
        g.add(PipelineNode(id="par2", run=lambda i: "P2", parallel=True))

        result = g.execute()
        assert result.all_succeeded()


# ===========================================
# Predicates
# ===========================================


class TestPredicates:
    def test_predicate_false_skips(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1))
        g.add(
            PipelineNode(
                id="b",
                run=lambda i: 99,
                depends_on=["a"],
                predicate=lambda inputs: inputs["a"] > 100,
            )
        )

        result = g.execute()
        assert result.nodes["a"].status == "success"
        assert result.nodes["b"].status == "skipped"
        assert result.output_of("b") is SKIPPED

    def test_skipped_does_not_propagate_failure(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1))
        g.add(
            PipelineNode(
                id="b",
                run=lambda i: 99,
                depends_on=["a"],
                predicate=lambda inputs: False,
            )
        )
        # c depends on a skipped b — should still run
        g.add(
            PipelineNode(
                id="c",
                run=lambda i: ("got", i["b"]),
                depends_on=["b"],
            )
        )

        result = g.execute()
        assert result.nodes["c"].status == "success"
        assert result.output_of("c") == ("got", SKIPPED)


# ===========================================
# Replan
# ===========================================


class TestReplan:
    def test_node_can_inject_followups(self):
        g = PipelineGraph()

        def planner(_i):
            return Replan(
                new_nodes=[
                    PipelineNode(id="d1", run=lambda i: "one"),
                    PipelineNode(id="d2", run=lambda i: "two"),
                    PipelineNode(
                        id="join",
                        run=lambda i: f"{i['d1']}+{i['d2']}",
                        depends_on=["d1", "d2"],
                    ),
                ]
            )

        g.add(PipelineNode(id="planner", run=planner))

        result = g.execute()
        assert "d1" in result.nodes
        assert "d2" in result.nodes
        assert "join" in result.nodes
        assert result.output_of("join") == "one+two"
        assert result.layers_executed >= 2

    def test_replan_can_reference_prior_nodes(self):
        g = PipelineGraph()

        def root(_i):
            return 5

        def spawner(inputs):
            base = inputs["root"]
            return Replan(
                new_nodes=[
                    PipelineNode(
                        id="follower",
                        run=lambda i, b=base: b * 2,
                    ),
                ]
            )

        g.add(PipelineNode(id="root", run=root))
        g.add(PipelineNode(id="spawner", run=spawner, depends_on=["root"]))

        result = g.execute()
        assert result.output_of("follower") == 10


# ===========================================
# Callbacks
# ===========================================


class TestCallbacks:
    def test_on_node_callbacks_fire(self):
        starts = []
        finishes = []

        g = PipelineGraph()
        g.add(PipelineNode(id="a", run=lambda i: 1))
        g.add(PipelineNode(id="b", run=lambda i: 2, depends_on=["a"]))

        g.execute(
            on_node_start=lambda n: starts.append(n.id),
            on_node_finish=lambda r: finishes.append((r.node_id, r.status)),
        )

        assert set(starts) == {"a", "b"}
        assert ("a", "success") in finishes
        assert ("b", "success") in finishes


# ===========================================
# GraphResult helpers
# ===========================================


class TestGraphResult:
    def test_succeeded_helper(self):
        g = PipelineGraph()
        g.add(PipelineNode(id="ok", run=lambda i: 1))

        def bad(_i):
            raise ValueError("nope")

        g.add(PipelineNode(id="bad", run=bad))

        result = g.execute()
        assert result.succeeded("ok")
        assert not result.succeeded("bad")
        assert not result.all_succeeded()


class TestReplanValidation:
    def test_replan_with_unknown_dep_raises(self):
        """A replan that references a nonexistent node must raise a clear
        error, not silently mark nodes failed."""
        import pytest

        from core.pipeline_graph import PipelineGraph, PipelineNode, Replan

        g = PipelineGraph()
        g.add(
            PipelineNode(
                id="root",
                run=lambda _inputs: Replan(
                    new_nodes=[
                        PipelineNode(id="child", run=lambda i: 1, depends_on=["ghost"]),
                    ]
                ),
            )
        )
        with pytest.raises(ValueError, match="unknown node"):
            g.execute()
