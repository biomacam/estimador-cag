# Arquitectura personal: cierre de Sesion 08

Tres vistas de menor a mayor complejidad. Se omite Streamlit.

**Alcance:** verde = implementado en `estimador-cag`; gris = cliente Rails existente en el proyecto vecino `ai-engineering/estimator-web`, mostrado solo como contexto de las dos primeras capas; amarillo = punto donde termina el flujo implementado. Las flechas entre Rails y la API representan el contrato HTTP, no una comprobacion de despliegue conjunto.

Los nombres solicitados son logicos: **`ingest/` corresponde a `app/ingestion/`** y **`storage/` corresponde a `app/persistence/`**. `embedding_pipeline/` existe como `app/embedding_pipeline/`. No se inventan directorios ni componentes pendientes.

## 1. Vista general: tres capas

```mermaid
flowchart LR
    F["Frontend externo<br/>Rails: vistas + Hotwire"]
    B["Backend de negocio externo<br/>Rails: controladores, contratos y persistencia"]
    IA["Servicio IA: estimador-cag<br/>FastAPI<br/>Estimacion, ingesta, embeddings y busqueda"]
    FIN["FIN IMPLEMENTADO S08<br/>Chunks persistidos + busqueda top-k por HTTP"]

    F -->|HTTP| B
    B -->|Contrato HTTP del cliente EstimatorAi| IA
    IA --> FIN

    classDef externo fill:#eeeeee,stroke:#666666,color:#222222;
    classDef implementado fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef limite fill:#fef3c7,stroke:#b45309,stroke-width:3px,color:#78350f;
    class F,B externo;
    class IA implementado;
    class FIN limite;
```

## 2. Vista modular del servicio IA

```mermaid
flowchart TB
    F["Frontend externo<br/>Rails: vistas + Hotwire"]
    B["Backend de negocio externo<br/>Rails + clientes EstimatorAi"]
    F --> B

    subgraph IA["Servicio IA: implementado en estimador-cag"]
        API["HTTP: app/main.py<br/>routers/ + embedding_pipeline/router.py"]
        EST["services/ + prompts/ + context/ + schemas/<br/>Estimacion estructurada, SSE y Actor-Critic-Boss"]
        SES["sessions/ + attachments.py<br/>Historial, resumen, anclas, metadata y PDF/DOCX"]
        GUARD["guardrails/<br/>Moderacion, inyeccion, PII y filtro de salida"]
        CACHE["cache/<br/>Cache exacta y cache semantica"]
        ING["ingest/ = app/ingestion/<br/>Catalogo, loaders, parsers, cleaning y pii"]
        EMB["embedding_pipeline/<br/>Chunking, comparacion, embeddings e ingesta vectorial"]
        ST["storage/ = app/persistence/<br/>SQLAlchemy, modelos y repositorios"]
        SEARCH["embedding_pipeline/search.py<br/>Embedding de consulta + distancia coseno"]
        FIN["FIN S08: POST /search<br/>Devuelve chunks top-k, metadata y distancia"]

        API --> EST
        API --> SES
        SES --> EST
        EST --> GUARD
        EST --> CACHE
        API --> ING
        ING -->|Estado y recuento del job| ST
        API --> EMB
        EMB -->|Documents + chunks con vectores| ST
        API --> SEARCH
        SEARCH -->|Consulta chunks| ST
        SEARCH --> FIN
    end

    PG[("PostgreSQL + pgvector")]
    REDIS[("Redis Stack")]
    LLM["Proveedor LLM<br/>OpenAI / Anthropic"]
    OPENAI["OpenAI embeddings<br/>text-embedding-3-small"]

    B -->|HTTP| API
    ST --> PG
    CACHE --> REDIS
    EST --> LLM
    EMB --> OPENAI
    SEARCH --> OPENAI

    classDef externo fill:#eeeeee,stroke:#666666,color:#222222;
    classDef implementado fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef limite fill:#fef3c7,stroke:#b45309,stroke-width:3px,color:#78350f;
    class F,B,LLM,OPENAI externo;
    class API,EST,SES,GUARD,CACHE,ING,EMB,ST,SEARCH,PG,REDIS implementado;
    class FIN limite;
    style IA fill:#f0fdf4,stroke:#15803d,stroke-width:2px
```

La ingesta documental registra jobs y produce documentos en memoria; no llama automaticamente a la ingesta vectorial. La limpieza y la seudonimizacion existen como utilidades, pero no se representan como pasos automaticos del orquestador. La busqueda de chunks tampoco se conecta al prompt de estimacion: son caminos independientes en el codigo actual.

## 3. Endpoints y flujos internos

```mermaid
flowchart TB
    F["Frontend externo: Rails / Hotwire"]
    B["Backend de negocio externo: Rails<br/>Clientes HTTP EstimatorAi"]
    F --> B

    subgraph IA["Servicio IA: frontera implementada al cierre de S08"]
        HTTP["FastAPI: app/main.py"]

        subgraph API["Endpoints HTTP registrados hoy"]
            E["POST /api/v1/estimate<br/>POST /api/v1/estimate/stream"]
            S["POST /sessions<br/>GET /sessions/{session_id}<br/>POST /sessions/{session_id}/estimate<br/>POST /sessions/{session_id}/estimate-acb"]
            I["POST /api/v1/ingestion/runs<br/>GET /api/v1/ingestion/jobs/{job_id}"]
            V["POST /embeddings/ingest"]
            C["POST /embeddings/compare"]
            Q["POST /search"]
            H["GET /health<br/>GET /docs + GET /redoc<br/>GET /openapi.json"]
        end

        HTTP --> E
        HTTP --> S
        HTTP --> I
        HTTP --> V
        HTTP --> C
        HTTP --> Q
        HTTP --> H

        GENERACION["services/llm_service.py + llm_wrapper.py<br/>Guardrails, prompts CAG y esquemas validados<br/>Cache exacta/semantica en estimacion simple"]
        CONV["sessions/ + attachments.py<br/>SessionStore en memoria: historial y metadata<br/>Conversacion + Actor-Critic-Boss en estimate-acb"]
        E --> GENERACION
        S --> CONV
        CONV --> GENERACION

        subgraph ING["ingest/ = app/ingestion/"]
            CAT["Catalogo: fuente incluida"]
            PARSE["BackgroundTask: ingest_source<br/>FileSystemLoader + ParserRegistry"]
            DOC["Document en memoria<br/>Fin del procesamiento documental"]
            CAT --> PARSE --> DOC
        end
        I --> CAT

        subgraph EMB["embedding_pipeline/"]
            CHUNK["ingest_document<br/>JSONStructuralChunker"]
            EMBED["OpenAIEmbedder<br/>Vectores de 1536 dimensiones"]
            COMP["ChunkingComparator<br/>Estrategias + estadisticas + top-k en memoria"]
            QUERY["search_chunks<br/>embed_one de la consulta"]
            SQL["cosine_search_statement<br/>ORDER BY distancia coseno, LIMIT k"]
            CHUNK --> EMBED
            QUERY --> SQL
        end
        V --> CHUNK
        C --> COMP
        Q --> QUERY

        subgraph STORAGE["storage/ = app/persistence/"]
            JOBS["JobsRepository<br/>ingestion_jobs: estado y documents_count"]
            MAP["MappingsRepository<br/>pseudonym_mappings"]
            ROWS["DocumentRow + ChunkRow<br/>documents + chunks + metadata JSONB<br/>embedding vector(1536)"]
        end
        PARSE -->|Actualiza el job| JOBS
        I -->|Consulta el job por id| JOBS
        PII["ingestion/pii/<br/>Seudonimizacion con Presidio"] --> MAP
        EMBED -->|Transaccion atomica; duplicados: 409| ROWS
        SQL -->|Lee chunks persistidos| ROWS
        SQL --> FIN["FIN IMPLEMENTADO S08<br/>SearchResponse: top-k, contenido, metadata,<br/>distancia y tiempo de busqueda"]
    end

    PG[("PostgreSQL + pgvector")]
    REDIS[("Redis Stack: caches")]
    PROVIDER["Proveedores externos<br/>LLM y OpenAI embeddings"]
    OPS["scripts/*_s08.py + scripts/sql_s08/<br/>Baseline, HNSW vector/halfvec, ef_search,<br/>recall, tamanos y mantenimiento"]

    B -->|Contrato HTTP| HTTP
    JOBS --> PG
    MAP --> PG
    ROWS --> PG
    GENERACION --> REDIS
    GENERACION --> PROVIDER
    EMBED --> PROVIDER
    COMP --> PROVIDER
    QUERY --> PROVIDER
    OPS -->|Mide y administra indices| PG

    classDef externo fill:#eeeeee,stroke:#666666,color:#222222;
    classDef implementado fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef limite fill:#fef3c7,stroke:#b45309,stroke-width:3px,color:#78350f;
    class F,B,PROVIDER externo;
    class HTTP,E,S,I,V,C,Q,H,GENERACION,CONV,CAT,PARSE,CHUNK,EMBED,COMP,QUERY,SQL,JOBS,MAP,ROWS,PII,PG,REDIS,OPS implementado;
    class FIN,DOC limite;
    style IA fill:#f0fdf4,stroke:#15803d,stroke-width:2px
```

**Limite operativo S08:** las migraciones dejan el baseline sin indice vectorial. Hay scripts implementados para crear y comparar HNSW `vector` y `halfvec`; no se afirma que esos indices esten creados en una base de datos concreta. `/search` ejecuta la consulta por distancia coseno sobre los vectores persistidos.

**Superficie auxiliar:** `app/main.py` tambien monta `/static` si existe `app/static/`; sirve recursos estaticos, no un endpoint de negocio. Swagger, ReDoc y OpenAPI documentan la API existente.

### Fuentes verificables

- [Registro de routers y health](app/main.py).
- [Estimaciones](app/routers/estimations.py) y [sesiones](app/routers/sessions.py).
- [Ingesta HTTP](app/routers/ingestion.py) y [orquestador documental](app/ingestion/orchestrator.py).
- [Endpoints de embeddings](app/embedding_pipeline/router.py) e [ingesta vectorial](app/embedding_pipeline/ingestion.py).
- [Endpoint de busqueda](app/routers/search.py) y [consulta por coseno](app/embedding_pipeline/search.py).
- [Modelos de almacenamiento](app/persistence/models.py) y [alcance operativo S08](README.md).