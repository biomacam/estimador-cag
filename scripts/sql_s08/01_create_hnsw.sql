-- ============================================================
-- Sesión 08 · Bloque 2.2
-- Construcción del índice HNSW principal con vector_cosine_ops
-- ============================================================
--
-- Cómo ejecutarlo: psql corre en el contenedor de Postgres, que NO tiene
-- este archivo montado. Desde la raíz del repo:
--
--   docker compose exec -T postgres psql -U estimator -d estimator \
--     < scripts/sql_s08/01_create_hnsw.sql
--
-- o abrir psql interactivo y pegar los bloques uno a uno:
--
--   docker compose exec postgres psql -U estimator -d estimator

-- Estado actual: NO debería haber índice vectorial todavía.
-- Solo veréis los índices relacionales de la migración (B-tree + GIN).
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks';

-- Construir el índice HNSW:
--   · m = 16              → conexiones por nodo del grafo. Subirlo mejora
--                           recall a costa de memoria y tiempo de build.
--   · ef_construction=128 → candidatos al construir (default pgvector: 64).
--                           Mejor grafo a cambio de un build más lento.
--   · vector_cosine_ops   → operator class de COSENO, alineada con el
--                           operador <=> de cosine_search_statement. Si no
--                           lo está, Postgres ignora el índice EN SILENCIO
--                           (Bloque 2.3).
CREATE INDEX chunks_embedding_idx
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Verificar que el índice existe y cuánto ocupa frente a la tabla.
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks';
