"""Abstract syntax tree definitions for minisql."""

from dataclasses import dataclass, field


class Expr:
    """Base class for SQL scalar and aggregate expressions."""


@dataclass(frozen=True)
class ColumnRef(Expr):
    """Reference to a column, optionally qualified by a table or subquery alias."""

    table: str | None
    name: str

    def format_sql(self) -> str:
        """Render the column reference back into SQL notation."""
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


@dataclass(frozen=True)
class Literal(Expr):
    """Literal constant value (int, float, str, or None)."""

    value: int | float | str | None


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary operation expression (`AND`, `OR`, `=`, `!=`, `<`, `<=`, `>`, `>=`)."""

    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation expression (such as `NOT`)."""

    op: str
    operand: Expr


@dataclass(frozen=True)
class AggCall(Expr):
    """Aggregate function call (`COUNT(*)`, `COUNT(col)`, `SUM(col)`, `MIN(col)`, `MAX(col)`)."""

    func: str
    arg: Expr | None = None
    star: bool = False


@dataclass(frozen=True)
class SelectItem:
    """A single projected expression in a SELECT list with an optional `AS` alias."""

    expr: Expr
    alias: str | None = None

    @property
    def output_name(self) -> str:
        """Return the externally visible column name produced by this SELECT item."""
        if self.alias is not None:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.name
        if isinstance(self.expr, AggCall):
            return self.expr.func.lower()
        return "expr"


class TableSource:
    """Base class for FROM clause table sources."""


@dataclass(frozen=True)
class TableRef(TableSource):
    """Reference to a base table in the catalog with an optional alias."""

    name: str
    alias: str | None = None

    @property
    def effective_name(self) -> str:
        """Return the alias if present, otherwise the base table name."""
        return self.alias if self.alias is not None else self.name


@dataclass(frozen=True)
class OrderByItem:
    """An item in an ORDER BY clause."""

    expr: ColumnRef
    ascending: bool = True


@dataclass(frozen=True)
class SelectStmt:
    """Parsed representation of a SQL SELECT statement."""

    items: list[SelectItem]
    source: TableSource
    where: Expr | None = None
    group_by: list[ColumnRef] = field(default_factory=list)
    order_by: list[OrderByItem] = field(default_factory=list)
    limit: int | None = None


@dataclass(frozen=True)
class SubquerySource(TableSource):
    """A derived table (`FROM (SELECT ...) alias`) with a required alias."""

    query: SelectStmt
    alias: str
