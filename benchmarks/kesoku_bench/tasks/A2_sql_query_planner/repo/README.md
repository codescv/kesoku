# minisql

`minisql` is a lightweight, pure-stdlib in-memory SQL query engine supporting `SELECT` projections with aliases, derived-table subqueries (`FROM (SELECT ...) alias`), `WHERE` predicates (`AND`, `OR`, comparisons), `GROUP BY` with `COUNT`, `ORDER BY`, and `LIMIT`.

## Architecture

- `minisql/lexer.py`: Tokenizer converting SQL text into a stream of `Token` objects.
- `minisql/ast.py`: AST node dataclasses (`SelectStmt`, `SelectItem`, `TableRef`, `SubquerySource`, `ColumnRef`, `Literal`, `BinaryOp`, `UnaryOp`, `AggCall`, `OrderByItem`).
- `minisql/parser.py`: Recursive-descent parser turning token streams into `SelectStmt` ASTs.
- `minisql/optimizer.py`: Query rewrite passes, including predicate pushdown into derived subqueries and column alias resolution.
- `minisql/planner.py`: Logical-to-physical plan builder (`ScanNode`, `SubqueryNode`, `FilterNode`, `AggregateNode`, `ProjectNode`, `SortNode`, `LimitNode`).
- `minisql/executor.py`: In-memory query executor and sample catalog (`load_default_catalog()`).

## Running Tests & Queries

Run the test suite with:

```bash
uv run pytest
```

See `queries/README.md` and `queries/regressions.sql` for sample queries against the default `users` table.
