-- ============================================================
-- Sesión 08 · Bloque 5.1
-- Queries de monitorización del estado de los índices
-- ============================================================

-- Query canónica: estado de todos los índices sobre chunks.
--   · idx_scan      → cuántas veces se ha usado el índice.
--   · last_idx_scan → cuándo fue la última vez (Postgres 16+).
SELECT
    indexrelname AS index_name,
    idx_scan AS scans,
    last_idx_scan AS last_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY idx_scan DESC;

-- Filas vivas vs muertas y última actualización de estadísticas.
-- Muchas muertas → VACUUM; last_analyze antiguo tras una carga → ANALYZE.
SELECT
    relname,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- Operator class de cada índice vectorial (para cazar el antipatrón del
-- Bloque 2.3 cuando idx_scan = 0).
SELECT
    i.indexrelname,
    op.opcname AS operator_class
FROM pg_stat_user_indexes i
JOIN pg_index pgi ON pgi.indexrelid = i.indexrelid
JOIN pg_opclass op ON op.oid = ANY (pgi.indclass::oid[])
WHERE i.relname = 'chunks'
  AND (op.opcname LIKE 'vector%' OR op.opcname LIKE 'halfvec%');
