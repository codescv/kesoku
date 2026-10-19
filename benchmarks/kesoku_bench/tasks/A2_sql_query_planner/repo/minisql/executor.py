"""In-memory query executor and default sample catalog for minisql."""

from typing import Any

from minisql.ast import (
    AggCall,
    BinaryOp,
    ColumnRef,
    Expr,
    Literal,
    SelectItem,
    UnaryOp,
)
from minisql.parser import parse_sql
from minisql.planner import (
    AggregateNode,
    FilterNode,
    LimitNode,
    PlanNode,
    ProjectNode,
    ScanNode,
    SortNode,
    SubqueryNode,
    build_plan,
)

Row = dict[str, Any]
Catalog = dict[str, list[Row]]


def load_default_catalog() -> Catalog:
    """Return a fresh copy of the standard in-memory catalog with the `users` table."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "dept": "eng", "tier": 5, "score": 95},
            {"id": 2, "name": "Bob", "dept": "sales", "tier": 2, "score": 70},
            {"id": 3, "name": "Carol", "dept": "eng", "tier": 4, "score": 88},
            {"id": 4, "name": "Dan", "dept": "sales", "tier": 1, "score": 60},
            {"id": 5, "name": "Eve", "dept": "eng", "tier": 3, "score": 82},
            {"id": 6, "name": "Frank", "dept": "ops", "tier": 4, "score": 91},
        ]
    }


def _lookup_column(row: Row, col: ColumnRef) -> Any:
    """Look up `col` in `row`, supporting both qualified (`u.id`) and unqualified (`id`) keys."""
    if col.table:
        qualified = f"{col.table}.{col.name}"
        if qualified in row:
            return row[qualified]
    if col.name in row:
        return row[col.name]
    # Fallback: match any `<alias>.<name>` suffix if unqualified lookup missed.
    suffix = f".{col.name}"
    for key, val in row.items():
        if key.endswith(suffix):
            return val
    return None


def _compare_values(op: str, left: Any, right: Any) -> bool:
    """Evaluate a SQL comparison operator between `left` and `right` safely."""
    if left is None or right is None:
        return False
    # If comparing incompatible scalar types (e.g. int vs str after a bad rewrite),
    # try numeric coercion; if either value is non-numeric, the comparison is False.
    if type(left) is not type(right) and not (
        isinstance(left, (int, float)) and isinstance(right, (int, float))
    ):
        try:
            left = float(left)
            right = float(right)
        except (ValueError, TypeError):
            return False

    if op == "=":
        return bool(left == right)
    if op == "!=":
        return bool(left != right)
    if op == "<":
        return bool(left < right)
    if op == "<=":
        return bool(left <= right)
    if op == ">":
        return bool(left > right)
    if op == ">=":
        return bool(left >= right)
    raise ValueError(f"Unsupported comparison operator: {op!r}")


def eval_expr(expr: Expr, row: Row, group_rows: list[Row] | None = None) -> Any:
    """Evaluate `expr` against a single `row` and optional `group_rows` (for aggregates)."""
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, ColumnRef):
        return _lookup_column(row, expr)

    if isinstance(expr, UnaryOp):
        val = eval_expr(expr.operand, row, group_rows)
        if expr.op == "NOT":
            return not bool(val)
        raise ValueError(f"Unsupported unary operator: {expr.op!r}")

    if isinstance(expr, BinaryOp):
        if expr.op == "AND":
            return bool(eval_expr(expr.left, row, group_rows)) and bool(
                eval_expr(expr.right, row, group_rows)
            )
        if expr.op == "OR":
            return bool(eval_expr(expr.left, row, group_rows)) or bool(
                eval_expr(expr.right, row, group_rows)
            )
        left_val = eval_expr(expr.left, row, group_rows)
        right_val = eval_expr(expr.right, row, group_rows)
        return _compare_values(expr.op, left_val, right_val)

    if isinstance(expr, AggCall):
        rows_for_agg = group_rows if group_rows is not None else [row]
        func = expr.func.upper()
        if func == "COUNT":
            if expr.star or expr.arg is None:
                return len(rows_for_agg)
            return sum(1 for r in rows_for_agg if eval_expr(expr.arg, r) is not None)
        if expr.arg is None:
            raise ValueError(f"Aggregate {func} requires an argument")
        vals = [eval_expr(expr.arg, r) for r in rows_for_agg]
        non_null = [v for v in vals if v is not None]
        if not non_null:
            return 0 if func == "SUM" else None
        if func == "SUM":
            return sum(non_null)
        if func == "MIN":
            return min(non_null)
        if func == "MAX":
            return max(non_null)
        raise ValueError(f"Unsupported aggregate function: {func!r}")

    raise TypeError(f"Unsupported expression node: {type(expr)!r}")


def _project_row(items: list[SelectItem], row: Row, group_rows: list[Row] | None = None) -> Row:
    """Evaluate each SelectItem in `items` and return a newly projected row dict."""
    out: Row = {}
    for item in items:
        out[item.output_name] = eval_expr(item.expr, row, group_rows)
    return out


def execute_plan(node: PlanNode, catalog: Catalog) -> list[Row]:
    """Execute a physical `PlanNode` tree against `catalog` and return the output rows."""
    if isinstance(node, ScanNode):
        raw_rows = catalog.get(node.table_name)
        if raw_rows is None:
            raise KeyError(f"Unknown table {node.table_name!r} in catalog")
        return [dict(r) for r in raw_rows]

    if isinstance(node, SubqueryNode):
        inner_rows = execute_plan(node.child, catalog)
        return [dict(r) for r in inner_rows]

    if isinstance(node, FilterNode):
        child_rows = execute_plan(node.child, catalog)
        return [r for r in child_rows if bool(eval_expr(node.predicate, r))]

    if isinstance(node, ProjectNode):
        child_rows = execute_plan(node.child, catalog)
        # Preserve underlying columns alongside projected aliases on intermediate nodes
        # only if not the final projection? No: SQL projection exposes only projected items,
        # while SortNode if present may inspect either projected output or source row.
        projected: list[Row] = []
        for r in child_rows:
            proj = _project_row(node.items, r)
            # Attach hidden source keys for downstream ORDER BY on non-projected columns,
            # stripped at top-level execute_sql.
            for k, v in r.items():
                if k not in proj:
                    proj[f"__src__.{k}"] = v
            projected.append(proj)
        return projected

    if isinstance(node, AggregateNode):
        child_rows = execute_plan(node.child, catalog)
        groups: dict[tuple[Any, ...], list[Row]] = {}
        for r in child_rows:
            key = tuple(_lookup_column(r, ColumnRef(None, col_name)) for col_name in node.group_by)
            groups.setdefault(key, []).append(r)
        agg_rows: list[Row] = []
        for group_list in groups.values():
            representative = group_list[0]
            agg_rows.append(_project_row(node.items, representative, group_list))
        return agg_rows

    if isinstance(node, SortNode):
        child_rows = execute_plan(node.child, catalog)
        result = list(child_rows)
        # Apply sort keys from right to left (Python's timsort is stable).
        for spec in reversed(node.order_by):
            def _sort_key(r: Row, col: ColumnRef = spec.expr) -> Any:
                val = _lookup_column(r, col)
                if val is None and f"__src__.{col.name}" in r:
                    val = r[f"__src__.{col.name}"]
                return val

            result.sort(key=_sort_key, reverse=not spec.ascending)
        return result

    if isinstance(node, LimitNode):
        child_rows = execute_plan(node.child, catalog)
        return child_rows[: node.limit]

    raise TypeError(f"Unsupported plan node: {type(node)!r}")


def execute_sql(
    sql: str,
    catalog: Catalog | None = None,
    *,
    optimize: bool = True,
) -> list[Row]:
    """Parse, plan, and execute `sql` against `catalog` (defaults to `load_default_catalog()`)."""
    active_catalog = catalog if catalog is not None else load_default_catalog()
    stmt = parse_sql(sql)
    plan = build_plan(stmt, optimize=optimize)
    raw_rows = execute_plan(plan, active_catalog)
    return [
        {k: v for k, v in row.items() if not k.startswith("__src__.")}
        for row in raw_rows
    ]
