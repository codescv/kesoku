"""Holdout verification tests for A2_sql_query_planner."""

from minisql.ast import ColumnRef
from minisql.executor import execute_sql
from minisql.optimizer import extract_column_refs
from minisql.parser import parse_sql
from minisql.planner import FilterNode, PlanNode, SubqueryNode, build_plan


def _collect_filter_nodes(node: PlanNode) -> list[FilterNode]:
    """Recursively collect all FilterNode instances in a PlanNode subtree."""
    found: list[FilterNode] = []
    if isinstance(node, FilterNode):
        found.append(node)
    child = getattr(node, "child", None)
    if isinstance(child, PlanNode):
        found.extend(_collect_filter_nodes(child))
    return found


def test_alias_shadowing_numeric() -> None:
    """Query 1: `u.id > 3` on `(SELECT tier AS id, id AS uid FROM users) u` resolves `id` -> `tier`."""
    sql = """
    SELECT u.id, u.uid
    FROM (SELECT tier AS id, id AS uid FROM users) u
    WHERE u.id > 3
    """
    rows = execute_sql(sql)
    assert rows == [
        {"id": 5, "uid": 1},
        {"id": 4, "uid": 3},
        {"id": 4, "uid": 6},
    ]


def test_alias_shadowing_string_swap() -> None:
    """Query 2: Two shadowed column aliases (`name AS id, dept AS name`) resolve to output schema."""
    sql = """
    SELECT u.id, u.name
    FROM (SELECT name AS id, dept AS name FROM users) u
    WHERE u.id = 'Carol' AND u.name = 'eng'
    """
    rows = execute_sql(sql)
    assert rows == [
        {"id": "Carol", "name": "eng"},
    ]


def test_limit_subquery_blocks_pushdown() -> None:
    """Query 3: Outer WHERE above a `LIMIT 3` subquery must filter after LIMIT, not before."""
    sql = """
    SELECT sub.id, sub.name, sub.score
    FROM (SELECT id, name, score FROM users LIMIT 3) sub
    WHERE sub.score >= 85
    """
    rows = execute_sql(sql)
    assert rows == [
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 3, "name": "Carol", "score": 88},
    ]


def test_ordered_limit_subquery_blocks_pushdown() -> None:
    """Query 4: Outer WHERE above an `ORDER BY ... LIMIT 3` subquery must not push below LIMIT."""
    sql = """
    SELECT top_u.name, top_u.tier
    FROM (SELECT name, tier, score FROM users ORDER BY score ASC LIMIT 3) top_u
    WHERE top_u.tier >= 3
    """
    rows = execute_sql(sql)
    assert rows == [
        {"name": "Eve", "tier": 3},
    ]


def test_group_by_blocks_aggregate_pushdown() -> None:
    """Query 5: Outer WHERE on aggregate column `g.cnt >= 2` must stay above GROUP BY."""
    sql = """
    SELECT g.dept, g.cnt
    FROM (SELECT dept, COUNT(*) AS cnt FROM users GROUP BY dept) g
    WHERE g.cnt >= 2
    """
    rows = execute_sql(sql)
    assert rows == [
        {"dept": "eng", "cnt": 3},
        {"dept": "sales", "cnt": 2},
    ]


def test_combined_alias_shadowing_and_group_by() -> None:
    """Query 6: Combined alias shadowing (`dept AS id, COUNT(*) AS dept`) and GROUP BY."""
    sql = """
    SELECT g.id, g.dept
    FROM (SELECT dept AS id, COUNT(*) AS dept FROM users GROUP BY dept) g
    WHERE g.id = 'eng' AND g.dept >= 2
    """
    rows = execute_sql(sql)
    assert rows == [
        {"id": "eng", "dept": 3},
    ]


def test_pushdown_still_happens_when_safe() -> None:
    """Ensure predicate pushdown still occurs for safe queries and rewrites shadowed aliases."""
    sql = """
    SELECT u.id, u.uid
    FROM (SELECT tier AS id, id AS uid FROM users) u
    WHERE u.id > 3
    """
    plan = build_plan(parse_sql(sql), optimize=True)

    # Find the SubqueryNode and verify there is no outer FilterNode above it,
    # while a FilterNode on `tier` exists inside the subquery.
    curr: PlanNode | None = plan
    outer_filters: list[FilterNode] = []
    subquery_node: SubqueryNode | None = None
    while curr is not None:
        if isinstance(curr, FilterNode):
            outer_filters.append(curr)
        if isinstance(curr, SubqueryNode):
            subquery_node = curr
            break
        curr = getattr(curr, "child", None)

    assert subquery_node is not None
    assert outer_filters == [], "Safe predicate should be pushed into subquery, not kept outside"

    inner_filters = _collect_filter_nodes(subquery_node.child)
    assert len(inner_filters) == 1, "Expected pushed FilterNode inside SubqueryNode"
    pushed_cols = [c for c in extract_column_refs(inner_filters[0].predicate) if isinstance(c, ColumnRef)]
    assert [c.name for c in pushed_cols] == ["tier"]
