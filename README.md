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

