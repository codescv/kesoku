-- Regression Query 1: Filtering on an aliased column in a derived subquery
-- where the subquery swaps/shadows column names from the base table.
SELECT u.id, u.uid
FROM (SELECT tier AS id, id AS uid FROM users) u
WHERE u.id > 3;

-- Regression Query 2: Filtering the output of a LIMIT subquery.
SELECT sub.id, sub.name, sub.score
FROM (SELECT id, name, score FROM users LIMIT 3) sub
WHERE sub.score >= 85;

-- Regression Query 3: Filtering the output of a GROUP BY subquery.
SELECT g.dept, g.cnt
FROM (SELECT dept, COUNT(*) AS cnt FROM users GROUP BY dept) g
WHERE g.cnt >= 2;
