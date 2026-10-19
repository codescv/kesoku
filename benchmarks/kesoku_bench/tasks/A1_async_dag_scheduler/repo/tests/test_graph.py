import pytest
from dagrun import DAG, CycleError, DuplicateNodeError, UnknownNodeError


async def _noop() -> None:
    """Do nothing."""


def _dag(edges: dict[str, list[str]]) -> DAG:
    """Build a DAG from ``{node: [deps]}``."""
    dag = DAG()
    for name, deps in edges.items():
        dag.add(name, _noop, deps=deps)
    return dag


def test_topo_order_is_deterministic():
    """Ready nodes come out in insertion order."""
    dag = _dag({"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"], "e": []})
    assert dag.topo_order() == ["a", "e", "b", "c", "d"]


def test_topo_order_respects_dependencies():
    """Every node appears after all of its dependencies."""
    dag = _dag({"z": ["y"], "y": ["x"], "x": [], "w": ["x", "z"]})
    order = dag.topo_order()
    for name in dag:
        for dep in dag.deps(name):
            assert order.index(dep) < order.index(name)


def test_cycle_detected():
    """A cycle is reported with its members in execution order."""
    dag = _dag({"a": ["c"], "b": ["a"], "c": ["b"], "d": []})
    with pytest.raises(CycleError) as info:
        dag.validate()
    cycle = info.value.cycle
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a", "b", "c"}


def test_self_dependency_is_a_cycle():
    """A node cannot depend on itself."""
    with pytest.raises(CycleError):
        _dag({"a": ["a"]})


def test_unknown_dependency():
    """Dependencies must exist."""
    dag = _dag({"a": ["ghost"]})
    with pytest.raises(UnknownNodeError):
        dag.topo_order()


def test_duplicate_node():
    """Node names are unique."""
    dag = _dag({"a": []})
    with pytest.raises(DuplicateNodeError):
        dag.add("a", _noop)


def test_successors_roots_leaves_descendants():
    """Graph navigation helpers."""
    dag = _dag({"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]})
    assert dag.successors("a") == ["b", "c"]
    assert dag.roots() == ["a"]
    assert dag.leaves() == ["d"]
    assert dag.descendants("a") == {"b", "c", "d"}
    assert dag.deps("d") == ("b", "c")
    assert "d" in dag and len(dag) == 4
