"""Logical and physical query planner for minisql."""

from dataclasses import dataclass

from minisql.ast import (
    Expr,
    OrderByItem,
    SelectItem,
    SelectStmt,
    SubquerySource,
    TableRef,
)
from minisql.optimizer import optimize_query


class PlanNode:
    """Base class for query execution plan nodes."""

    def format_tree(self, indent: int = 0) -> str:
        """Render a human-readable multi-line string of the plan subtree."""
        raise NotImplementedError


@dataclass(frozen=True)
class ScanNode(PlanNode):
    """Full scan of an in-memory base table in the catalog."""

    table_name: str
    alias: str

    def format_tree(self, indent: int = 0) -> str:
        """Format this scan node as an indented string."""
        prefix = "  " * indent
        return f"{prefix}Scan({self.table_name} AS {self.alias})"


@dataclass(frozen=True)
class SubqueryNode(PlanNode):
    """Materializes a derived subquery under `alias`."""

    child: PlanNode
    alias: str

    def format_tree(self, indent: int = 0) -> str:
        """Format this subquery node and its child subtree."""
        prefix = "  " * indent
        return f"{prefix}Subquery(alias={self.alias})\n{self.child.format_tree(indent + 1)}"


@dataclass(frozen=True)
class FilterNode(PlanNode):
    """Row-level filter evaluating `predicate` on each incoming row."""

    child: PlanNode
    predicate: Expr

    def format_tree(self, indent: int = 0) -> str:
        """Format this filter node and its child subtree."""
        prefix = "  " * indent
        return f"{prefix}Filter({self.predicate})\n{self.child.format_tree(indent + 1)}"


@dataclass(frozen=True)
class AggregateNode(PlanNode):
    """Groups incoming rows by `group_by` and evaluates `items` per group."""

    child: PlanNode
    group_by: list[str]
    items: list[SelectItem]

    def format_tree(self, indent: int = 0) -> str:
        """Format this aggregate node and its child subtree."""
        prefix = "  " * indent
        keys = ", ".join(self.group_by)
        return f"{prefix}Aggregate(keys=[{keys}])\n{self.child.format_tree(indent + 1)}"


@dataclass(frozen=True)
class ProjectNode(PlanNode):
    """Projects a list of `SelectItem` expressions onto output column names."""

    child: PlanNode
    items: list[SelectItem]

    def format_tree(self, indent: int = 0) -> str:
        """Format this projection node and its child subtree."""
        prefix = "  " * indent
        cols = ", ".join(item.output_name for item in self.items)
        return f"{prefix}Project([{cols}])\n{self.child.format_tree(indent + 1)}"


@dataclass(frozen=True)
class SortNode(PlanNode):
    """Sorts incoming rows according to `order_by` specifications."""

    child: PlanNode
    order_by: list[OrderByItem]

    def format_tree(self, indent: int = 0) -> str:
        """Format this sort node and its child subtree."""
        prefix = "  " * indent
        spec = ", ".join(
            f"{item.expr.format_sql()} {'ASC' if item.ascending else 'DESC'}"
            for item in self.order_by
        )
        return f"{prefix}Sort([{spec}])\n{self.child.format_tree(indent + 1)}"


@dataclass(frozen=True)
class LimitNode(PlanNode):
    """Truncates the incoming stream to at most `limit` rows."""

    child: PlanNode
    limit: int

    def format_tree(self, indent: int = 0) -> str:
        """Format this limit node and its child subtree."""
        prefix = "  " * indent
        return f"{prefix}Limit({self.limit})\n{self.child.format_tree(indent + 1)}"


def _build_unoptimized_plan(stmt: SelectStmt) -> PlanNode:
    """Convert a SelectStmt directly into a tree of PlanNodes without rewrites."""
    if isinstance(stmt.source, TableRef):
        node: PlanNode = ScanNode(
            table_name=stmt.source.name,
            alias=stmt.source.effective_name,
        )
    elif isinstance(stmt.source, SubquerySource):
        inner_plan = _build_unoptimized_plan(stmt.source.query)
        node = SubqueryNode(child=inner_plan, alias=stmt.source.alias)
    else:
        raise TypeError(f"Unsupported table source: {type(stmt.source)!r}")

    if stmt.where is not None:
        node = FilterNode(child=node, predicate=stmt.where)

    if stmt.group_by:
        group_cols = [col.name for col in stmt.group_by]
        node = AggregateNode(child=node, group_by=group_cols, items=stmt.items)
    else:
        node = ProjectNode(child=node, items=stmt.items)

    if stmt.order_by:
        node = SortNode(child=node, order_by=stmt.order_by)

    if stmt.limit is not None:
        node = LimitNode(child=node, limit=stmt.limit)

    return node


def build_plan(stmt: SelectStmt, *, optimize: bool = True) -> PlanNode:
    """Optimize `stmt` (when `optimize=True`) and construct its execution PlanNode tree."""
    target = optimize_query(stmt) if optimize else stmt
    return _build_unoptimized_plan(target)
