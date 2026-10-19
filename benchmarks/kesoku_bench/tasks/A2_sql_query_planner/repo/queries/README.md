# Documented Queries & Expected Results

All queries below execute against the default catalog returned by `minisql.executor.load_default_catalog()`.

## Default `users` Table

| `id` | `name` | `dept` | `tier` | `score` |
|---:|---|---|---:|---:|
| 1 | `"Alice"` | `"eng"` | 5 | 95 |
| 2 | `"Bob"` | `"sales"` | 2 | 70 |
| 3 | `"Carol"` | `"eng"` | 4 | 88 |
| 4 | `"Dan"` | `"sales"` | 1 | 60 |
| 5 | `"Eve"` | `"eng"` | 3 | 82 |
| 6 | `"Frank"` | `"ops"` | 4 | 91 |

---

## Documented Queries

### 1. Subquery Alias Shadowing (Numeric)

```sql
SELECT u.id, u.uid
FROM (SELECT tier AS id, id AS uid FROM users) u
WHERE u.id > 3;
```

In the subquery `u`, output column `id` is an alias for `users.tier`, while `uid` is an alias for `users.id`. Filtering on `u.id > 3` must filter on the subquery's output column `id` (`users.tier > 3`), returning:

```python
[
    {"id": 5, "uid": 1},
    {"id": 4, "uid": 3},
    {"id": 4, "uid": 6},
]
```

### 2. Subquery Alias Shadowing (String Swap)

```sql
SELECT u.id, u.name
FROM (SELECT name AS id, dept AS name FROM users) u
WHERE u.id = 'Carol' AND u.name = 'eng';
```

Expected rows:

```python
[
    {"id": "Carol", "name": "eng"},
]
```

### 3. Outer Filter Above `LIMIT` Subquery

```sql
SELECT sub.id, sub.name, sub.score
FROM (SELECT id, name, score FROM users LIMIT 3) sub
WHERE sub.score >= 85;
```

The inner query selects the first 3 rows of `users` (`Alice` with 95, `Bob` with 70, `Carol` with 88). The outer `WHERE sub.score >= 85` filters those 3 rows, returning 2 rows (pushing the filter below `LIMIT 3` would wrongly include `Frank` with 91):

```python
[
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 3, "name": "Carol", "score": 88},
]
```

### 4. Outer Filter Above Ordered `LIMIT` Subquery

```sql
SELECT top_u.name, top_u.tier
FROM (SELECT name, tier, score FROM users ORDER BY score ASC LIMIT 3) top_u
WHERE top_u.tier >= 3;
```

The inner query takes the 3 lowest-scoring users (`Dan` score 60 tier 1, `Bob` score 70 tier 2, `Eve` score 82 tier 3). Filtering those 3 rows by `top_u.tier >= 3` returns:

```python
[
    {"name": "Eve", "tier": 3},
]
```

### 5. Outer Filter Above `GROUP BY` Subquery

```sql
SELECT g.dept, g.cnt
FROM (SELECT dept, COUNT(*) AS cnt FROM users GROUP BY dept) g
WHERE g.cnt >= 2;
```

The inner query groups by `dept` (`eng`: 3, `sales`: 2, `ops`: 1). The outer filter keeps groups with `cnt >= 2`:

```python
[
    {"dept": "eng", "cnt": 3},
    {"dept": "sales", "cnt": 2},
]
```

### 6. Combined Alias Shadowing and `GROUP BY`

```sql
SELECT g.id, g.dept
FROM (SELECT dept AS id, COUNT(*) AS dept FROM users GROUP BY dept) g
WHERE g.id = 'eng' AND g.dept >= 2;
```

Expected rows:

```python
[
    {"id": "eng", "dept": 3},
]
```
