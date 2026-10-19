"""Recursive-descent SQL parser for minisql."""

from minisql.ast import (
    AggCall,
    BinaryOp,
    ColumnRef,
    Expr,
    Literal,
    OrderByItem,
    SelectItem,
    SelectStmt,
    SubquerySource,
    TableRef,
    TableSource,
    UnaryOp,
)
from minisql.lexer import Token, TokenKind, tokenize

AGG_TOKEN_KINDS: frozenset[TokenKind] = frozenset(
    {
        TokenKind.COUNT,
        TokenKind.SUM,
        TokenKind.MIN,
        TokenKind.MAX,
    }
)


class ParseError(ValueError):
    """Raised when a SQL statement violates minisql grammar rules."""


class Parser:
    """Stateful token-stream parser for minisql SELECT queries."""

    def __init__(self, tokens: list[Token]) -> None:
        """Initialize the parser with a tokenized SQL stream."""
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind is not TokenKind.EOF:
            self._pos += 1
        return tok

    def _match(self, *kinds: TokenKind) -> Token | None:
        if self._peek().kind in kinds:
            return self._advance()
        return None

    def _expect(self, kind: TokenKind, message: str | None = None) -> Token:
        """Consume the next token if it matches `kind`, otherwise raise ParseError."""
        tok = self._peek()
        if tok.kind is not kind:
            detail = message or f"Expected {kind}, got {tok.kind} ({tok.text!r})"
            raise ParseError(f"{detail} at position {tok.pos}")
        return self._advance()

    def parse_statement(self) -> SelectStmt:
        """Parse a complete top-level SELECT statement and optional trailing semicolon."""
        stmt = self._parse_select_stmt()
        self._match(TokenKind.SEMICOLON)
        if self._peek().kind is not TokenKind.EOF:
            tok = self._peek()
            raise ParseError(f"Unexpected token {tok.text!r} at position {tok.pos}")
        return stmt

    def _parse_select_stmt(self) -> SelectStmt:
        """Parse `SELECT <items> FROM <source> [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT n]`."""
        self._expect(TokenKind.SELECT)
        items = self._parse_select_list()
        self._expect(TokenKind.FROM)
        source = self._parse_table_source()

        where_expr: Expr | None = None
        if self._match(TokenKind.WHERE):
            where_expr = self._parse_or_expr()

        group_by: list[ColumnRef] = []
        if self._match(TokenKind.GROUP):
            self._expect(TokenKind.BY)
            group_by.append(self._parse_column_ref())
            while self._match(TokenKind.COMMA):
                group_by.append(self._parse_column_ref())

        order_by: list[OrderByItem] = []
        if self._match(TokenKind.ORDER):
            self._expect(TokenKind.BY)
            order_by.append(self._parse_order_by_item())
            while self._match(TokenKind.COMMA):
                order_by.append(self._parse_order_by_item())

        limit_val: int | None = None
        if self._match(TokenKind.LIMIT):
            num_tok = self._expect(TokenKind.NUMBER, "Expected integer after LIMIT")
            limit_val = int(num_tok.text)

        return SelectStmt(
            items=items,
            source=source,
            where=where_expr,
            group_by=group_by,
            order_by=order_by,
            limit=limit_val,
        )

    def _parse_select_list(self) -> list[SelectItem]:
        """Parse a comma-separated list of SELECT projection items."""
        items: list[SelectItem] = [self._parse_select_item()]
        while self._match(TokenKind.COMMA):
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> SelectItem:
        """Parse a single projection item `<expr> [AS <alias>]`."""
        expr = self._parse_primary_expr()
        alias: str | None = None
        if self._match(TokenKind.AS):
            alias_tok = self._expect(TokenKind.IDENT, "Expected column alias after AS")
            alias = alias_tok.text
        elif self._peek().kind is TokenKind.IDENT:
            alias = self._advance().text
        return SelectItem(expr=expr, alias=alias)

    def _parse_table_source(self) -> TableSource:
        """Parse either a base table reference or a parenthesized subquery with an alias."""
        if self._match(TokenKind.LPAREN):
            subquery = self._parse_select_stmt()
            self._expect(TokenKind.RPAREN)
            self._match(TokenKind.AS)
            alias_tok = self._expect(TokenKind.IDENT, "Subquery in FROM requires an alias")
            return SubquerySource(query=subquery, alias=alias_tok.text)

        name_tok = self._expect(TokenKind.IDENT, "Expected table name in FROM clause")
        alias: str | None = None
        if self._match(TokenKind.AS):
            alias = self._expect(TokenKind.IDENT, "Expected table alias after AS").text
        elif self._peek().kind is TokenKind.IDENT:
            alias = self._advance().text
        return TableRef(name=name_tok.text, alias=alias)

    def _parse_order_by_item(self) -> OrderByItem:
        """Parse a single `col [ASC | DESC]` item in an ORDER BY clause."""
        col = self._parse_column_ref()
        ascending = True
        if self._match(TokenKind.ASC):
            ascending = True
        elif self._match(TokenKind.DESC):
            ascending = False
        return OrderByItem(expr=col, ascending=ascending)

    def _parse_or_expr(self) -> Expr:
        """Parse left-associative OR boolean expressions."""
        left = self._parse_and_expr()
        while self._match(TokenKind.OR):
            right = self._parse_and_expr()
            left = BinaryOp(op="OR", left=left, right=right)
        return left

    def _parse_and_expr(self) -> Expr:
        """Parse left-associative AND boolean expressions."""
        left = self._parse_not_expr()
        while self._match(TokenKind.AND):
            right = self._parse_not_expr()
            left = BinaryOp(op="AND", left=left, right=right)
        return left

    def _parse_not_expr(self) -> Expr:
        """Parse optional unary NOT expressions."""
        if self._match(TokenKind.NOT):
            return UnaryOp(op="NOT", operand=self._parse_not_expr())
        return self._parse_comparison_expr()

    def _parse_comparison_expr(self) -> Expr:
        """Parse binary comparison expressions (`=`, `!=`, `<`, `<=`, `>`, `>=`)."""
        left = self._parse_primary_expr()
        op_tok = self._match(TokenKind.OP)
        if op_tok is not None:
            right = self._parse_primary_expr()
            return BinaryOp(op=op_tok.text, left=left, right=right)
        return left

    def _parse_primary_expr(self) -> Expr:
        """Parse literals, aggregate calls, parenthesized expressions, or column references."""
        tok = self._peek()

        if tok.kind is TokenKind.LPAREN:
            self._advance()
            inner = self._parse_or_expr()
            self._expect(TokenKind.RPAREN)
            return inner

        if tok.kind in AGG_TOKEN_KINDS:
            func_tok = self._advance()
            self._expect(TokenKind.LPAREN)
            if self._match(TokenKind.STAR):
                self._expect(TokenKind.RPAREN)
                return AggCall(func=func_tok.text, arg=None, star=True)
            arg_expr = self._parse_primary_expr()
            self._expect(TokenKind.RPAREN)
            return AggCall(func=func_tok.text, arg=arg_expr, star=False)

        if tok.kind is TokenKind.NUMBER:
            self._advance()
            if "." in tok.text:
                return Literal(value=float(tok.text))
            return Literal(value=int(tok.text))

        if tok.kind is TokenKind.STRING:
            self._advance()
            return Literal(value=tok.text)

        if tok.kind is TokenKind.IDENT:
            return self._parse_column_ref()

        raise ParseError(f"Unexpected token {tok.text!r} ({tok.kind}) at position {tok.pos}")

    def _parse_column_ref(self) -> ColumnRef:
        """Parse `ident` or `table_alias.ident` column reference."""
        first = self._expect(TokenKind.IDENT, "Expected column or table identifier")
        if self._match(TokenKind.DOT):
            second = self._expect(TokenKind.IDENT, "Expected column name after '.'")
            return ColumnRef(table=first.text, name=second.text)
        return ColumnRef(table=None, name=first.text)


def parse_sql(sql: str) -> SelectStmt:
    """Tokenize and parse a SQL SELECT statement string into a SelectStmt AST."""
    tokens = tokenize(sql)
    return Parser(tokens).parse_statement()


def parse_sql_script(sql_script: str) -> list[SelectStmt]:
    """Parse multiple semicolon-delimited SQL statements from a script string."""
    tokens = tokenize(sql_script)
    stmts: list[SelectStmt] = []
    # Split token stream on semicolons while keeping EOF at the end of each statement.
    current: list[Token] = []
    for tok in tokens:
        if tok.kind is TokenKind.SEMICOLON:
            if current:
                current.append(Token(TokenKind.EOF, "", tok.pos))
                stmts.append(Parser(current).parse_statement())
                current = []
        elif tok.kind is TokenKind.EOF:
            if current:
                current.append(tok)
                stmts.append(Parser(current).parse_statement())
                current = []
        else:
            current.append(tok)
    return stmts
