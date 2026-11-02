-- ============================================================
-- Sesión 08 · Bloque 4.1
-- Creación del índice halfvec en paralelo al vector original
-- ============================================================
--
-- halfvec almacena cada componente en 16 bits en vez de 32. La columna
-- embedding NO cambia (sigue siendo vector(1536)): solo el ÍNDICE se
-- construye sobre la expresión casteada. Resultado: un índice ~50% más
-- pequeño que, con embeddings normalizados, devuelve resultados equivalentes.

-- Tamaño del índice vector actual, como referencia.
SELECT pg_size_pretty(pg_relation_size('chunks_embedding_idx')) AS vector_index_size;

-- Índice de EXPRESIÓN (doble paréntesis) con halfvec_cosine_ops.
-- Solo lo usarán queries que ordenen por LA MISMA expresión casteada;
-- `embedding <=> ...` sin cast NO lo usa.
CREATE INDEX chunks_embedding_halfvec_idx
ON chunks
USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Tamaños lado a lado. Esperado: el halfvec ocupa aproximadamente la mitad.
SELECT
    indexrelname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
  AND indexrelname LIKE '%embedding%';

-- Verificación: esperado "Index Scan using chunks_embedding_halfvec_idx".
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY (embedding::halfvec(1536)) <=> (
    SELECT embedding::halfvec(1536) FROM chunks ORDER BY id LIMIT 1
)
LIMIT 5;
