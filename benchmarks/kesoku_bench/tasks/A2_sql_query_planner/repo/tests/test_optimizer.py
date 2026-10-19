"""Visible unit tests for minisql query optimization and execution."""

from minisql.ast import BinaryOp, ColumnRef, Literal, SubquerySource
from minisql.executor import execute_sql
from minisql.optimizer import optimize_query
from minisql.parser import parse_sql


def test_simple_subquery_predicate_pushdown() -> None:
    """Verify safe predicate pushdown into a simple non-shadowing derived subquery."""
    sql = """
    SELECT u.user_id, u.user_name
    FROM (SELECT id AS user_id, name AS user_name, dept FROM users) u
    WHERE u.dept = 'ops'
    """
    stmt = parse_sql(sql)
    optimized = optimize_query(stmt)

    assert optimized.where is None
    assert isinstance(optimized.source, SubquerySource)
    assert optimized.source.query.where == BinaryOp(
        op="=",
        left=ColumnRef(table=None, name="dept"),
        right=Literal(value="ops"),
    )

    rows = execute_sql(sql)
    assert rows == [{"user_id": 6, "user_name": "Frank"}]


def test_aliased_column_pushdown_non_shadowing() -> None:
    """Verify pushing a predicate that references a unique alias in the subquery."""
    sql = """
    SELECT u.user_id, u.points
    FROM (SELECT id AS user_id, score AS points FROM users) u
    WHERE u.points >= 90
    """
    rows = execute_sql(sql)
    assert rows == [
        {"user_id": 1, "points": 95},
        {"user_id": 6, "points": 91},
    ]


def test_base_table_group_by_and_order_by() -> None:
    """Verify executing a direct GROUP BY + ORDER BY query without subqueries."""
    sql = "SELECT dept, COUNT(*) AS cnt FROM users GROUP BY dept ORDER BY cnt DESC;"
    rows = execute_sql(sql)
    assert rows == [
        {"dept": "eng", "cnt": 3},
        {"dept": "sales", "cnt": 2},
        {"dept": "ops", "cnt": 1},
    ]
