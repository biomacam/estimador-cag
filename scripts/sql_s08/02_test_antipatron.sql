-- ============================================================
-- Sesión 08 · Bloque 2.3
-- Demostración del antipatrón operador / operator class
-- ============================================================
--
-- El índice está construido con vector_cosine_ops (operador <=>). Si una
-- query ordena por <-> (distancia L2), Postgres NO puede usar el índice... y
-- no avisa: sequential scan sin error ni warning. La defensa es EXPLAIN ANALYZE.
--
-- Como query vector se usa el embedding de un chunk existente (subconsulta
-- escalar): el top-1 será ese mismo chunk a distancia 0.0.

-- Caso 1: operador ALINEADO con el índice (vector_cosine_ops + <=>).
-- Esperado: "Index Scan using chunks_embedding_idx".
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY embedding <=> (SELECT embedding FROM chunks ORDER BY id LIMIT 1)
LIMIT 5;

-- Caso 2: operador DESALINEADO (índice de coseno, query con <-> de L2).
-- Esperado: "Seq Scan on chunks" + Sort. Comparad los "Execution Time".
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY embedding <-> (SELECT embedding FROM chunks ORDER BY id LIMIT 1)
LIMIT 5;

-- idx_scan sube con el Caso 1 pero NO con el Caso 2. Un índice vectorial con
-- idx_scan = 0 en producción es casi siempre este antipatrón.
SELECT indexrelname, idx_scan, last_idx_scan
FROM pg_stat_user_indexes
WHERE relname = 'chunks';

-- ------------------------------------------------------------
-- Variante con un vector externo (la query real del benchmark).
-- measure_baseline_s08.py imprime al final el literal pgvector de la
-- primera query; sustituid <embedding_vector> por ese literal:
--
-- EXPLAIN ANALYZE
-- SELECT id, chunk_type
-- FROM chunks
-- ORDER BY embedding <=> '<embedding_vector>'::vector
-- LIMIT 5;
-- ------------------------------------------------------------
