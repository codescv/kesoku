"""Visible unit tests for the minisql lexer and parser."""

from minisql.ast import (
    AggCall,
    BinaryOp,
    ColumnRef,
    Literal,
    SelectStmt,
    SubquerySource,
    TableRef,
)
from minisql.parser import parse_sql, parse_sql_script


def test_parse_simple_select_with_where_and_limit() -> None:
    """Verify parsing a flat SELECT with aliases, WHERE comparison, ORDER BY, and LIMIT."""
    sql = "SELECT id AS user_id, name FROM users WHERE score >= 80 ORDER BY score DESC LIMIT 5;"
    stmt = parse_sql(sql)

    assert isinstance(stmt, SelectStmt)
    assert len(stmt.items) == 2
    assert stmt.items[0].alias == "user_id"
    assert stmt.items[0].expr == ColumnRef(table=None, name="id")
    assert stmt.items[1].output_name == "name"
    assert stmt.source == TableRef(name="users", alias=None)
    assert stmt.where == BinaryOp(
        op=">=",
        left=ColumnRef(table=None, name="score"),
        right=Literal(value=80),
    )
    assert len(stmt.order_by) == 1
    assert stmt.order_by[0].expr == ColumnRef(table=None, name="score")
    assert stmt.order_by[0].ascending is False
    assert stmt.limit == 5


def test_parse_subquery_and_group_by_count() -> None:
    """Verify parsing a derived subquery with GROUP BY and COUNT(*)."""
    sql = """
    SELECT g.dept, g.cnt
    FROM (SELECT dept, COUNT(*) AS cnt FROM users GROUP BY dept) g
    WHERE g.dept = 'eng'
    """
    stmt = parse_sql(sql)

    assert isinstance(stmt.source, SubquerySource)
    assert stmt.source.alias == "g"
    inner = stmt.source.query
    assert len(inner.group_by) == 1
    assert inner.group_by[0] == ColumnRef(table=None, name="dept")
    assert inner.items[1].alias == "cnt"
    assert inner.items[1].expr == AggCall(func="COUNT", arg=None, star=True)


def test_parse_sql_script_multiple_statements() -> None:
    """Verify splitting and parsing multiple statements separated by semicolons."""
    script = """
    -- First query
    SELECT id FROM users WHERE id = 1;
    -- Second query
    SELECT name FROM users LIMIT 2;
    """
    stmts = parse_sql_script(script)
    assert len(stmts) == 2
    assert stmts[0].limit is None
    assert stmts[1].limit == 2
