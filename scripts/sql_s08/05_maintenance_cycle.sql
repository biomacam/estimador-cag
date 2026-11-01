-- ============================================================
-- Sesión 08 · Bloque 5.2
-- Ciclo de mantenimiento: ANALYZE, VACUUM, REINDEX CONCURRENTLY
-- ============================================================
--
-- Contexto: se acaban de insertar ~100 chunks sintéticos con
-- insert_synthetic_chunks_s08.py para simular crecimiento del corpus.
--
-- REINDEX ... CONCURRENTLY y VACUUM no pueden ir dentro de una transacción.
-- En psql (autocommit) no hay problema, tanto interactivo como redirigiendo
-- el archivo completo.

-- Estado previo.
SELECT n_live_tup, n_dead_tup, last_analyze, last_vacuum
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- Paso 1 — ANALYZE: recalcula estadísticas del planner. Tras cargas grandes, siempre.
ANALYZE chunks;

SELECT n_live_tup, n_dead_tup, last_analyze
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- Paso 2 — VACUUM ANALYZE: además recupera espacio de filas muertas.
VACUUM ANALYZE chunks;

-- Paso 3 — REINDEX CONCURRENTLY: reconstruye el HNSW sin bloquear queries
-- (índice paralelo + swap atómico; cuesta el doble de espacio temporal).
REINDEX INDEX CONCURRENTLY chunks_embedding_idx;

-- Verificar que el índice sigue activo tras el swap.
SELECT
    indexrelname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    idx_scan
FROM pg_stat_user_indexes
WHERE relname = 'chunks';

-- Cadencias orientativas en producción:
--   · ANALYZE   → autovacuum; manual tras cargas masivas.
--   · VACUUM    → semanal (o autovacuum bien configurado).
--   · REINDEX   → mensual, o ante degradación de recall/latencia
--                 (el grafo HNSW se degrada con muchos updates/deletes).
