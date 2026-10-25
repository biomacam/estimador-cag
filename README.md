# Estimador CAG

## Arrancar con Docker Compose

Requisitos: Docker Desktop con Docker Compose y una API key de OpenAI o Anthropic.

Desde la carpeta del proyecto, crea `.env` si todavía no existe y añade la clave del proveedor:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Arranca los servicios:

```powershell
docker compose up -d --build
docker compose ps
```

Compose levanta `estimator`, `postgres` con pgvector, `redis` y `streamlit`. El servicio `estimator` aplica automáticamente las migraciones de Alembic al arrancar.

- Streamlit: http://localhost:8501
- API y Swagger: http://localhost:8000/docs

Para consultar la versión de PostgreSQL o los logs:

```powershell
docker compose exec postgres psql -U estimator -d estimator -c "SELECT version();"
docker compose logs -f estimator
```

Detén los servicios con `docker compose down`. Esto conserva los datos del volumen de PostgreSQL.

## Esquema de persistencia

La migración `0002_document_chunks` amplía la migración de ingesta existente (`0001`) con `documents` y `chunks`.

- **Dos tablas:** un documento puede producir muchos chunks. Separarlos evita duplicar la metadata del documento y conserva la integridad referencial; `ON DELETE CASCADE` elimina sus chunks al borrar el documento.
- **Metadata JSONB:** los atributos estables, como tipo de documento o tipo de chunk, son columnas tipadas; los atributos variables, como tags, scope o tecnologías, quedan en JSONB. El índice GIN de `chunks.metadata` permite consultarlos sin migrar el esquema por cada clave nueva. En el ORM el atributo se llama `metadata_json` porque `metadata` está reservado por SQLAlchemy; el nombre SQL sigue siendo `metadata`.
- **`vector(1536)` nullable:** 1536 es la dimensión de `text-embedding-3-small`; cambiarla exige volver a generar los embeddings. `NULL` deja abierta una futura ingesta asíncrona en dos etapas, aunque el flujo actual guarda chunk y embedding en la misma transacción.
- **Distancia cosine:** usamos `cosine_distance` (`<=>`), la convención más extendida en RAG. Para embeddings normalizados, cosine, inner product y L2 producen rankings equivalentes; la elección es semántica y operativa, no una mejora de precisión. Además, `<=>` coincide con la operator class `vector_cosine_ops` prevista para un futuro índice HNSW, evitando que PostgreSQL ignore el índice por incompatibilidad de operador.
- **Sin índice vectorial por ahora:** el corpus de ejemplo tiene cientos de chunks, donde un sequential scan exacto es suficiente y evita el espacio y mantenimiento extra de un índice. Mediremos el corpus y la latencia antes de añadir HNSW y ajustar sus parámetros.

