"""Query optimizer for minisql: boolean simplification and subquery predicate pushdown."""

from dataclasses import replace

from minisql.ast import (
    AggCall,
    BinaryOp,
    ColumnRef,
    Expr,
    Literal,
    SelectStmt,
    SubquerySource,
    UnaryOp,
)


def _build_subquery_schema(subquery: SelectStmt) -> dict[str, Expr]:
    """Build a lookup map from column names exposed by `subquery` to inner expressions.

    Maps each projected item in the subquery so outer predicates referencing
    `sub_alias.<col>` can be rewritten in terms of the subquery's internal columns.
    """
    schema: dict[str, Expr] = {}
    # Register explicit output aliases declared in the subquery's SELECT list.
    for item in subquery.items:
        if item.alias is not None:
            schema[item.alias] = item.expr

    # Also register underlying ColumnRef names so unaliased pass-through columns resolve.
    for item in subquery.items:
        if isinstance(item.expr, ColumnRef):
            schema[item.expr.name] = ColumnRef(table=item.expr.table, name=item.expr.name)

    return schema


def _resolve_column_in_subquery(
    col: ColumnRef,
    subquery_alias: str,
    schema: dict[str, Expr],
) -> Expr | None:
    """Resolve an outer ColumnRef against the subquery's schema map.

    Returns the mapped inner expression if the column belongs to `subquery_alias`
    (or is unqualified) and exists in `schema`.
    """
    if col.table is not None and col.table != subquery_alias:
        return None
    if col.name in schema:
        return schema[col.name]
    return ColumnRef(table=None, name=col.name)


def split_conjuncts(expr: Expr | None) -> list[Expr]:
    """Flatten a top-level tree of `AND` BinaryOps into a list of conjunct expressions."""
    if expr is None:
        return []
    if isinstance(expr, BinaryOp) and expr.op == "AND":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def combine_conjuncts(conjuncts: list[Expr]) -> Expr | None:
    """Combine a list of conjunct expressions into a left-deep `AND` tree, or None if empty."""
    if not conjuncts:
        return None
    result = conjuncts[0]
    for item in conjuncts[1:]:
        result = BinaryOp(op="AND", left=result, right=item)
    return result


def extract_column_refs(expr: Expr) -> list[ColumnRef]:
    """Collect all ColumnRef nodes appearing anywhere inside `expr` in traversal order."""
    refs: list[ColumnRef] = []
    if isinstance(expr, ColumnRef):
        refs.append(expr)
    elif isinstance(expr, BinaryOp):
        refs.extend(extract_column_refs(expr.left))
        refs.extend(extract_column_refs(expr.right))
    elif isinstance(expr, UnaryOp):
        refs.extend(extract_column_refs(expr.operand))
    elif isinstance(expr, AggCall) and expr.arg is not None:
        refs.extend(extract_column_refs(expr.arg))
    return refs


def contains_aggregate(expr: Expr) -> bool:
    """Return True if `expr` contains any AggCall node (`COUNT`, `SUM`, `MIN`, `MAX`)."""
    if isinstance(expr, AggCall):
        return True
    if isinstance(expr, BinaryOp):
        return contains_aggregate(expr.left) or contains_aggregate(expr.right)
    if isinstance(expr, UnaryOp):
        return contains_aggregate(expr.operand)
    return False


def simplify_expr(expr: Expr) -> Expr:
    """Perform lightweight constant folding and double-negation elimination on `expr`."""
    if isinstance(expr, UnaryOp):
        inner = simplify_expr(expr.operand)
        if expr.op == "NOT" and isinstance(inner, UnaryOp) and inner.op == "NOT":
            return inner.operand
        return UnaryOp(op=expr.op, operand=inner)

    if isinstance(expr, BinaryOp):
        left = simplify_expr(expr.left)
        right = simplify_expr(expr.right)
        if expr.op == "AND":
            if isinstance(left, Literal) and left.value is True:
                return right
            if isinstance(right, Literal) and right.value is True:
                return left
        return BinaryOp(op=expr.op, left=left, right=right)

    return expr


def _rewrite_expr_for_subquery(
    expr: Expr,
    subquery_alias: str,
    schema: dict[str, Expr],
) -> Expr | None:
    """Rewrite all ColumnRefs in `expr` from outer subquery references to inner expressions.

    Returns None if any referenced column cannot be mapped into the subquery scope.
    """
    if isinstance(expr, Literal):
        return expr

    if isinstance(expr, ColumnRef):
        return _resolve_column_in_subquery(expr, subquery_alias, schema)

    if isinstance(expr, UnaryOp):
        rewritten_op = _rewrite_expr_for_subquery(expr.operand, subquery_alias, schema)
        if rewritten_op is None:
            return None
        return UnaryOp(op=expr.op, operand=rewritten_op)

    if isinstance(expr, BinaryOp):
        left = _rewrite_expr_for_subquery(expr.left, subquery_alias, schema)
        right = _rewrite_expr_for_subquery(expr.right, subquery_alias, schema)
        if left is None or right is None:
            return None
        return BinaryOp(op=expr.op, left=left, right=right)

    if isinstance(expr, AggCall):
        return expr

    return None


def _push_predicates(stmt: SelectStmt) -> SelectStmt:
    """Recursively push outer WHERE predicates down into derived-table subqueries."""
    if not isinstance(stmt.source, SubquerySource):
        if stmt.where is not None:
            return replace(stmt, where=simplify_expr(stmt.where))
        return stmt

    sub_source = stmt.source
    subquery = sub_source.query
    sub_alias = sub_source.alias

    if stmt.where is None:
        optimized_inner = _push_predicates(subquery)
        return replace(stmt, source=SubquerySource(query=optimized_inner, alias=sub_alias))

    schema = _build_subquery_schema(subquery)
    outer_conjuncts = split_conjuncts(simplify_expr(stmt.where))

    pushed_conjuncts: list[Expr] = []
    remaining_outer: list[Expr] = []

    for conj in outer_conjuncts:
        rewritten = _rewrite_expr_for_subquery(conj, sub_alias, schema)
        if rewritten is not None:
            pushed_conjuncts.append(rewritten)
        else:
            remaining_outer.append(conj)

    existing_inner = split_conjuncts(subquery.where)
    merged_inner_where = combine_conjuncts(existing_inner + pushed_conjuncts)
    updated_inner = replace(subquery, where=merged_inner_where)

    # Recursively optimize nested subqueries if the inner source is also a subquery.
    optimized_inner = _push_predicates(updated_inner)
    return replace(
        stmt,
        source=SubquerySource(query=optimized_inner, alias=sub_alias),
        where=combine_conjuncts(remaining_outer),
    )


def optimize_query(stmt: SelectStmt) -> SelectStmt:
    """Apply all logical optimization passes to `stmt` and return the rewritten AST."""
    return _push_predicates(stmt)
