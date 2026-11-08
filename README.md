# Estimador CAG

## Arrancar con Docker Compose

Requisitos: Docker Desktop con Docker Compose y una API key de OpenAI o Anthropic.

Desde la carpeta del proyecto, crea `.env` si todavía no existe y añade la clave del proveedor:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Arranca los servicios desde la raíz del proyecto:

```powershell
docker compose up -d --build
docker compose ps
```

El `docker-compose.yml` raíz incluye el cliente Rails (`estimator-web/docker-compose.yml`), así que Compose levanta seis servicios en la misma red:

| Servicio | Qué es | Puerto host |
|---|---|---|
| `estimator` | API FastAPI; aplica las migraciones de Alembic al arrancar | 8000 |
| `postgres` | PostgreSQL 16 + pgvector del servicio IA | 5432 |
| `redis` | Redis Stack (caches; RedisInsight en 8001) | 6379, 8001 |
| `streamlit` | Cliente Streamlit | 8501 |
| `estimator-web` | Cliente Rails 8; llama a la API en `http://estimator:8000` | 3000 |
| `web-postgres` | PostgreSQL 16 de Rails | 5433 |

- Cliente Rails: http://localhost:3000 (healthcheck en `/up`)
- Streamlit: http://localhost:8501
- API y Swagger: http://localhost:8000/docs

El primer build de `estimator-web` compila gemas nativas y puede tardar varios minutos. Al arrancar, Rails ejecuta `db:prepare` sobre `web-postgres`. El `.env` de `estimator-web/` es opcional: el compose ya fija la URL de la API y la base de datos.

Para levantar solo una parte, indica los servicios (Compose arranca también sus dependencias):

```powershell
docker compose up -d --build estimator              # solo el servicio IA (+ postgres y redis)
docker compose up -d --build estimator estimator-web  # servicio IA + cliente Rails
```

Arranca siempre desde la raíz: lanzar `docker compose` desde `estimator-web/` crea otro proyecto Compose con volúmenes distintos y sin el servicio `estimator`.

Para consultar PostgreSQL, los logs o ejecutar los tests de Rails:

```powershell
docker compose exec postgres psql -U estimator -d estimator -c "SELECT version();"
docker compose exec web-postgres psql -U postgres estimator_web_development
docker compose logs -f estimator estimator-web
docker compose exec -e RAILS_ENV=test estimator-web bash -c "bin/rails db:prepare && bin/rails test"
```

Detén los servicios con `docker compose down`. Esto conserva los datos de los volúmenes de PostgreSQL.

## Esquema de persistencia

La migración `0002_document_chunks` amplía la migración de ingesta existente (`0001`) con `documents` y `chunks`.

- **Dos tablas:** un documento puede producir muchos chunks. Separarlos evita duplicar la metadata del documento y conserva la integridad referencial; `ON DELETE CASCADE` elimina sus chunks al borrar el documento.
- **Metadata JSONB:** los atributos estables, como tipo de documento o tipo de chunk, son columnas tipadas; los atributos variables, como tags, scope o tecnologías, quedan en JSONB. El índice GIN de `chunks.metadata` permite consultarlos sin migrar el esquema por cada clave nueva. En el ORM el atributo se llama `metadata_json` porque `metadata` está reservado por SQLAlchemy; el nombre SQL sigue siendo `metadata`.
- **`vector(1536)` nullable:** 1536 es la dimensión de `text-embedding-3-small`; cambiarla exige volver a generar los embeddings. `NULL` deja abierta una futura ingesta asíncrona en dos etapas, aunque el flujo actual guarda chunk y embedding en la misma transacción.
- **Distancia cosine:** usamos `cosine_distance` (`<=>`), la convención más extendida en RAG. Para embeddings normalizados, cosine, inner product y L2 producen rankings equivalentes; la elección es semántica y operativa, no una mejora de precisión. Además, `<=>` coincide con la operator class `vector_cosine_ops` prevista para un futuro índice HNSW, evitando que PostgreSQL ignore el índice por incompatibilidad de operador.
- **Sin índice vectorial por ahora:** el corpus de ejemplo tiene cientos de chunks, donde un sequential scan exacto es suficiente y evita el espacio y mantenimiento extra de un índice. Mediremos el corpus y la latencia antes de añadir HNSW y ajustar sus parámetros.

### English full-text search

Migration `0006_chunks_fts_english` replaces the Spanish configuration from
`0005_chunks_fts_spanish` with `english` to match the budget corpus.
The `chunks.content_tsv` column is a
`GENERATED ALWAYS ... STORED` `tsvector` computed from `content`, with the GIN
index `ix_chunks_content_tsv`. PostgreSQL recomputes it on content inserts and
updates; lexical queries use `plainto_tsquery('english', ...)` for matching English
stemming and stop words.

Compose applies the migration through `alembic upgrade head` when the API starts.
It rebuilds the generated column and index for existing rows, preserving chunk
data and embeddings. This takes an exclusive table lock during the migration;
allow for a maintenance window on large datasets. Downgrade restores the Spanish
column and index and should be paired with the previous application version.

### Hybrid retrieval and RRF

`app.rag.retrieval.pipeline.retrieve()` returns one unique best-first chunk ranking.
Hybrid mode retrieves up to `RETRIEVAL_RECALL_TOP_K` candidates from each branch:
cosine-distance vector search and English keyword search using `content_tsv @@
plainto_tsquery('english', query_text)`, ordered by `ts_rank_cd` descending.
Both branches apply the same sector, year and chunk-type filters; the distance
threshold applies only to the vector branch.

Reciprocal Rank Fusion sums `1 / (RRF_K + rank)` for each chunk across the two
rankings, with positions starting at 1. `RRF_K` is a positive environment setting
(default 60). Scores are not mixed directly; shared chunks appear only once.
Branch ties and fused-score ties break by ascending chunk id for reproducibility.
Without reranking, the fused list is cut to `top_k`; with reranking, the first
`RETRIEVAL_RECALL_TOP_K` fused candidates are rescored and cut to `RERANK_TOP_N`.
Lexical-only chunks retain the existing display distance sentinel `1.0`; this is
not a measured cosine distance and is not used by RRF or the reranker.

Both `POST /v1/retrieval/search` (retrieval API key) and
`POST /v1/estimate/stages/retrieve` (estimate API key) accept explicit switches:

| Configuration | `search_mode` | `rerank` |
|---|---|---|
| A: vector baseline | `"vector"` | `false` |
| B: hybrid RRF | `"hybrid"` | `false` |
| C: vector + reranking | `"vector"` | `true` |
| D: hybrid RRF + reranking | `"hybrid"` | `true` |

Send both switches to avoid runtime defaults affecting an experiment. For example:

```powershell
$headers = @{ "X-API-Key" = "<your-retrieval-api-key>" }
$body = @{
    query_text = "presupuestos de desarrollo de comercio electrónico"
    search_mode = "hybrid"
    rerank = $false
    top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/retrieval/search" `
    -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

When omitted, switches resolve from the Redis runtime override and then settings.
The stage endpoint now uses the same pipeline and precedence as the search
endpoint; send `"search_mode": "vector", "rerank": false` for its old vector-only
behavior. The legacy `POST /search` remains vector-only.

`uv run python scripts/eval_retrieval_s10.py` runs all four configurations through
the shared pipeline. It requires the migrated PostgreSQL database, an ingested
corpus, an embedding-provider key and available cross-encoder weights; model
downloads happen on first use. All runs use the same `RRF_K` and recall limit.

### Cross-encoder: recall 50, rerank 5

The shared retrieval pipeline resolves the lazy `CrossEncoderReranker` only when
reranking is enabled and candidates exist. It scores up to 50 `(query, content)`
pairs in a worker thread, then keeps the five highest-scoring chunks. For hybrid
search, RRF first merges both recall lists; only the first 50 fused candidates
reach the cross-encoder. Tied scores retain the recall order.

Defaults in `.env`:

```dotenv
RETRIEVAL_RECALL_TOP_K=50
RERANK_TOP_N=5
RERANKER_ENABLED=true
```

Set `"rerank": true` or `"rerank": false` in either retrieval endpoint to override
the toggle for one request. To change subsequent retrievals without editing code
or restarting the API, update the Redis-backed runtime toggle:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/config/retrieval" `
    -Method Put -ContentType "application/json" -Body '{"rerank":false}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/config/retrieval" `
    -Method Put -ContentType "application/json" -Body '{"rerank":true}'
```

Explicit request switches take precedence. `{"rerank":null}` clears the runtime
override and restores the environment default. The full transcript pipeline also
reads this toggle, except when replaying a cached idempotent result.

Disabled reranking does not load or run the cross-encoder and returns `top_k`
chunks in recall order. Enabled reranking returns up to `RERANK_TOP_N`, or fewer
when recall finds fewer candidates. First use may download model weights; model
loading/inference failures raise a retrieval error (HTTP 502), never silently
fall back to the original ranking. Invalid score counts or non-finite scores are
rejected explicitly.

### Golden set and measured comparison

`evals/golden_retrieval.json` contains five manually reviewed project descriptions
and relevant budget ids, annotated by requested functionality rather than sector
alone. `scripts/eval_retrieval_s10.py` evaluates A/B/C/D with real embeddings and
cross-encoder inference. It saves the comparison in `evals/retrieval_results.md`
and individual measured rankings, latencies, settings and corpus fingerprints in
`evals/retrieval_results.json`.

Run against a dedicated PostgreSQL database with migrations applied and all 15
base budgets ingested (52 budget-component chunks). The evaluator rejects missing,
unembedded or additional budget-component chunks rather than publishing a
misleading comparison:

```powershell
uv run python scripts/eval_retrieval_s10.py --output evals/retrieval_results.json
```

Precision@5 is relevant retrieved chunks divided by five, averaged over the five
queries. A chunk inherits its parent's manually annotated relevance; different
chunks from the same budget count separately. Q2/Q3/Q4 have fewer than five
relevant chunks, which limits their maximum scores. Each query/configuration has
one discarded warm-up and three measured runs. Latency includes retrieval and
reranking but excludes query embedding, initial model loading and HTTP overhead.
The shipped dataset, queries and current full-text configuration are English.
Lexical match counts are recorded to diagnose restrictive keyword matching.
`evals/retrieval_results.json` and `.md` contain measurements rerun with English
full-text search after migration `0006`, using an isolated database, real
embeddings and the locally cached cross-encoder. The lexical branch still finds
no matches for these long project descriptions because `plainto_tsquery` requires
all non-stop-word terms to occur in a single chunk.

## Sesión 08 — Indexación vectorial y operación

La migración deja `chunks` **sin índice vectorial** a propósito: el sequential scan es el baseline contra el que se mide el impacto del índice. Los índices HNSW (`vector` y `halfvec`) se crean en vivo con los snippets de `scripts/sql_s08/`; IVFFlat queda fuera de alcance.

### Scripts Python (`scripts/*_s08.py`)

Se ejecutan con `docker compose run --rm estimator python scripts/<script>`. Reutilizan la configuración, la sesión async, el embedder y la misma query que `/search` (`cosine_search_statement`). `s08_common.py` es un módulo compartido.

| Script | Qué hace |
|---|---|
| `measure_baseline_s08.py` | Latencia SQL de las queries del benchmark (warm-up + 2 mediciones). Ejecutar antes y después de crear el índice. Imprime el literal pgvector de la primera query. |
| `sweep_ef_search_s08.py` | Barre `hnsw.ef_search` en [10..200] y mide latencia y recall contra la verdad de fondo (sequential scan forzado). Marca la recomendación con ★. |
| `compare_indexes_s08.py` | Compara el índice `vector` y el `halfvec` (forzados por expresión): top-5, overlap y latencias. |
| `report_index_sizes_s08.py` | Tipo, tamaño, `idx_scan` y último uso de cada índice de `chunks`. |
| `insert_synthetic_chunks_s08.py` | Inserta chunks sintéticos con embeddings reales (default 100; `30000` para que el baseline sea medible). Limpieza: `DELETE FROM documents WHERE document_type = 'synthetic_test';` |

### Snippets SQL (`scripts/sql_s08/`)

```powershell
# Archivo completo (desde la raíz del repo):
Get-Content scripts/sql_s08/01_create_hnsw.sql | docker compose exec -T postgres psql -U estimator -d estimator
# Interactivo (pegar bloques):
docker compose exec postgres psql -U estimator -d estimator
```

| Orden | Snippet | Qué hace |
|---|---|---|
| 1 | `01_create_hnsw.sql` | Índice HNSW (`vector_cosine_ops`, m=16, ef_construction=128) |
| 2 | `02_test_antipatron.sql` | Antipatrón silencioso `<=>` vs `<->` con `EXPLAIN ANALYZE` |
| 3 | `03_create_halfvec.sql` | Índice halfvec sobre `(embedding::halfvec(1536))` |
| 4 | `04_monitoring_queries.sql` | Monitorización con `pg_stat_user_indexes` |
| 5 | `05_maintenance_cycle.sql` | ANALYZE → VACUUM → REINDEX CONCURRENTLY |
| 99 | `99_revert_live_session.sql` | Vuelve al baseline sin índice vectorial |

Secuencia típica: `insert_synthetic_chunks_s08.py 30000` → `measure_baseline_s08.py` (seq scan) → `01_create_hnsw.sql` → `measure_baseline_s08.py` (HNSW) → `sweep_ef_search_s08.py`.

Si un índice vectorial tiene `idx_scan = 0` tras servir queries semánticas, casi seguro el operador de la query no coincide con la operator class del índice (p. ej. `<->` contra `vector_cosine_ops`).
