-- ============================================================
-- Sesión 08 · Reversión post-ensayo (→ estado baseline sin índice)
-- ============================================================
--
-- Devuelve la BBDD al punto de arranque: chunks reales + sintéticos del
-- pre-flight, SIN índice vectorial y sin overrides persistentes. No re-embebe.
--
-- Qué revierte:
--   · los índices HNSW (vector y halfvec),
--   · los chunks sintéticos insertados después del pre-flight,
--   · el override global de ef_search (ALTER DATABASE),
--   · los contadores de pg_stat para un "antes" limpio.
--
-- Uso (desde la raíz del repo):
--
--   docker compose exec -T postgres psql -U estimator -d estimator \
--     < scripts/sql_s08/99_revert_live_session.sql
--
-- Sintéticos a conservar (default 30000): -v keep_synthetic=50000
-- Idempotente y fail-safe.

\if :{?keep_synthetic}
\else
  \set keep_synthetic 30000
\endif

\echo '== Estado ANTES de revertir =='
SELECT
    (SELECT count(*) FROM chunks)                              AS total_chunks,
    (SELECT count(*) FROM chunks WHERE chunk_type='synthetic') AS synthetic_chunks;
SELECT indexrelname FROM pg_stat_user_indexes WHERE relname='chunks' ORDER BY indexrelname;

-- 1. Borrar los sintéticos posteriores al pre-flight (ids más altos).
--    Si hay <= keep_synthetic, la subconsulta da NULL y no se borra nada.
DELETE FROM chunks
WHERE chunk_type = 'synthetic'
  AND id > (
      SELECT id
      FROM chunks
      WHERE chunk_type = 'synthetic'
      ORDER BY id
      OFFSET (:keep_synthetic - 1)
      LIMIT 1
  );

-- 2. Eliminar los índices vectoriales (incluye uno INVALID por un CONCURRENTLY abortado).
DROP INDEX IF EXISTS chunks_embedding_idx;
DROP INDEX IF EXISTS chunks_embedding_halfvec_idx;

-- 3. Revertir el override global de ef_search (inofensivo si no se fijó).
ALTER DATABASE estimator RESET hnsw.ef_search;

-- 4. Resetear estadísticas y repoblar las del planner.
SELECT pg_stat_reset();
ANALYZE chunks;

\echo '== Estado DESPUÉS de revertir =='
SELECT
    (SELECT count(*) FROM chunks)                              AS total_chunks,
    (SELECT count(*) FROM chunks WHERE chunk_type='synthetic') AS synthetic_chunks,
    (SELECT count(*) FROM documents WHERE document_type='synthetic_test') AS synthetic_docs;
-- Esperado: solo índices relacionales (chunks_pkey, ix_chunks_*); NINGÚN hnsw.
SELECT indexrelname FROM pg_stat_user_indexes WHERE relname='chunks' ORDER BY indexrelname;
