# Session 11 — Citación verificable + baseline RAGAS

Entregable del pre-work de la Sesión 11. Cubre las dos partes del enunciado:

1. **Citación verificable a nivel de línea** (Parte 1).
2. **Baseline de calidad de generación con RAGAS** sobre el golden set extendido (Parte 2).

Todo el código (schema, prompt, verificación, harness) está en inglés; estas notas van en español.

## Nota breve: lo que más chirría de los números

`faithfulness` (0.49) y `context_recall` (0.35) son flojos aunque hay 0 citas colgantes y
`context_precision` ≈ 0.99: la citación por línea es real, pero gruesa (el generador descompone
cada componente histórico en subtareas y reparte horas), y es probable (hipótesis, no verificada)
que por eso el juez no pueda atribuir cada cifra derivada a la del `<source>`. `context_recall` es
0.000 en Q1 y Q2 y mejor en Q3–Q5 (0.38–0.71), y en Q4 y Q5 el total sale en **horas** (840 y 600)
en vez de engineer-days (~62 y ~76), una inconsistencia de unidades que el baseline deja al
descubierto.

---

## Qué se cambió (Parte 1)

- **Schema** (`app/rag/schemas.py`):
  - `SourceReference{chunk_id, document_id, evidence}` — la cita verificable de una línea.
  - `TaskItem` (cada línea de estimación) lleva `grounded: bool` y
    `sources: list[SourceReference]` (antes `list[int]`). Un `model_validator` impone la regla de
    integridad: `grounded=True` ⇒ ≥1 fuente; `grounded=False` ⇒ sin fuentes y sin horas inventadas
    (`engineer_days` pasa a ser opcional).
  - `CitationReport` / `LineCitation` — la salida del verificador.
- **Ensamblado de contexto** (`app/rag/context_assembler.py`): cada `<source>` expone también
  `document_id` (el `budget_id` del chunk).
- **Prompt de generación** (`app/rag/prompt_builder.py`): atribución obligatoria por línea —
  `chunk_id` exacto del `<source>`, `document_id` y `evidence` **verbatim** (no parafraseado);
  `grounded=false` cuando no hay soporte (sin horas a ojo).
- **Verificación post-generación** (`app/rag/validation.py::verify_citations`): recorre cada línea y
  comprueba que todo `chunk_id` citado esté en el conjunto de chunks recuperados. Distingue líneas
  *grounded*, *dangling* (id inventado) e *insufficient* (sin datos). Se integra en el orquestador
  (`estimator.py`: un reintento correctivo y bajada a `confidence=low` si no se repara; log
  `citation_report` por `request_id`) y en `/v1/estimate/stages/generate` (campos `citation_report`
  y `fabricated_source_ids`).

### Reporte de verificación de citaciones (criterios de aceptación)

Generado por `scripts/demo_verify_citations_s11.py` (offline, sin red): una estimación grounded con
**una citación colgante plantada a propósito** (`chunk_id=999`, nunca recuperado) y una línea sin
datos suficientes.

```
=== Citation verification report ===
lines: 4  grounded: 2  dangling: 1  insufficient: 1
verified citations: 2
dangling citations: ['999']

module                   line                     status        cited -> dangling
----------------------------------------------------------------------------------------
Authentication & SCA     OAuth 2.0 backend        grounded      ['101'] -> -
PSD2 & Open Banking      Open banking connectors  grounded      ['102'] -> -
Ledger                   Transaction ledger       dangling      ['999'] -> ['999']
Reporting                Regulatory reporting     insufficient  [] -> -

ACCEPTANCE: PASS
```

- **grounded=True con fuente real**: las 2 primeras líneas citan chunks 101/102 que sí estaban en
  el contexto.
- **citación colgante detectada**: "Transaction ledger" cita 999 → marcada `dangling`.
- **sin datos suficientes**: "Regulatory reporting" no se rellena con horas; se marca `insufficient`.

Sobre estimaciones **reales** del pipeline, el informe por consulta está en
`evals/ragas_baseline_s11.json` (campo `citation_report`) y resumido más abajo.

---

## Baseline RAGAS (Parte 2)

### Qué hay

- **Golden set** — `evals/golden_generation_s11.json`: extiende las 5 consultas Q1–Q5 del golden
  set de la Sesión 10 (`golden_retrieval.json`) con un `transcript` y un `ground_truth` (estimación
  de referencia en engineer-days ≈ horas/8, derivada de los presupuestos relevantes de
  `data/budgets_sample.json`). `query` y `relevant_budget_ids` son idénticos a los de la Sesión 10.
- **Recogida** — `scripts/eval_ragas_s11.py`: por cada consulta ejecuta el pipeline real
  (reformulación → embedding → recuperación híbrida + rerank → ensamblado → generación) y registra
  las cuatro entradas de RAGAS:

  | Entrada RAGAS | Origen |
  |---|---|
  | `question` | `query` del golden set (la petición de estimación) |
  | `answer` | la estimación generada, renderizada como texto |
  | `contexts` | los chunks recuperados que se pasaron al generador |
  | `ground_truth` | `ground_truth` del golden set |

  También guarda el `citation_report` (`verify_citations`) de cada estimación real.
- **Puntuación** — `scripts/score_ragas_s11.py`: calcula `faithfulness`, `answer_relevancy`,
  `context_precision` y `context_recall`. Juez: `gpt-4o-mini`; embeddings:
  `text-embedding-3-small` (ambos configurables en el golden set). Usa `OPENAI_API_KEY`.
- **Resultado** — `evals/ragas_baseline_s11.json` (tabla + métricas por consulta + informe de citas).

## Cómo reproducirlo

`ragas` importa módulos de Vertex de `langchain_community` que la versión del proyecto ya no
expone, así que se puntúa en un venv aislado (el scorer registra un stub de Vertex que nunca se
usa con juez OpenAI). Flujo en dos pasos, con el stack arriba (`docker compose up -d`) y el
corpus ingerido:

```bash
# 1) Recoger muestras con el pipeline real, dentro del contenedor (tiene BD, clave y reranker).
#    - scripts/ está montado; evals/ no, de ahí --golden.
#    - gpt-5 con razonamiento alto tarda ~2 min por consulta: LLM_TIMEOUT=30 (por defecto) se
#      queda corto, se sobreescribe solo para este run.
docker exec estimator mkdir -p /tmp/evals_s11
docker cp evals/golden_generation_s11.json estimator:/tmp/evals_s11/
docker exec -e LLM_TIMEOUT=900 -e LLM_RETRIES=1 estimator python scripts/eval_ragas_s11.py \
  --golden /tmp/evals_s11/golden_generation_s11.json --collect-only /tmp/samples.json
docker cp estimator:/tmp/samples.json samples.json

# 2) Puntuar en un venv aislado con ragas (OPENAI_API_KEY en el entorno)
uv venv /ruta/ragas-venv --python 3.12
uv pip install --python /ruta/ragas-venv/bin/python ragas datasets langchain-openai langchain-community
/ruta/ragas-venv/bin/python scripts/score_ragas_s11.py samples.json --out evals/ragas_baseline_s11.json
```

## Tabla de métricas (4 métricas × 5 consultas + promedio)

Run real: juez `gpt-4o-mini`, embeddings `text-embedding-3-small`, ragas 0.4.4, 5 chunks de contexto
por consulta (`RERANK_TOP_N`).

| query | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Q1 | 0.585 | 0.111 | 1.000 | 0.000 |
| Q2 | 0.417 | 0.040 | 1.000 | 0.000 |
| Q3 | 0.613 | 0.155 | 1.000 | 0.714 |
| Q4 | 0.407 | 0.214 | 0.950 | 0.667 |
| Q5 | 0.433 | 0.098 | 1.000 | 0.375 |
| **average** | 0.491 | 0.124 | 0.990 | 0.351 |

### Verificación de citas sobre las estimaciones reales

Salida de `verify_citations` sobre cada estimación generada:

| query | líneas | grounded | dangling | insufficient | citas verificadas | total (días) | referencia |
|---|---|---|---|---|---|---|---|
| Q1 | 42 | 25 | 0 | 17 | 29 | 61 | ~60 |
| Q2 | 27 | 27 | 0 | 0 | 31 | 58 | ~59 |
| Q3 | 38 | 22 | 0 | 16 | 26 | 56 | ~70 |
| Q4 | 39 | 33 | 0 | 6 | 38 | 840 | ~62 |
| Q5 | 35 | 26 | 0 | 9 | 30 | 600 | ~76 |
| **total** | **181** | **133** | **0** | **48** | **154** | | |

**Citas colgantes: 0/181 líneas.** Las 48 líneas sin soporte se marcan `insufficient` (sin horas
inventadas) en lugar de rellenarse.

## Hallazgos

- **`context_precision` ≈ 1.0 con `context_recall` irregular (0.35).** La recuperación trae los
  presupuestos relevantes, pero el juez no atribuye bien el `ground_truth` al contexto en Q1 y Q2
  (0.000). Hipótesis (no verificada con el juez): el `ground_truth` está en **engineer-days** y el
  corpus en **horas**, así que las cifras no casan línea a línea.
- **Anomalía de unidades en la generación (medida, no hipótesis):** en Q4 (840) y Q5 (600) el total
  está en escala de **horas**, no de engineer-days (referencia ~62 y ~76). En Q5 coincide exactamente
  con las 600 h de BUD-2024-003, el presupuesto principal; en Q4 (840) no coincide con las 500 h de
  BUD-2024-013, pero también es incompatible con días (no he analizado de dónde sale la cifra).
  Q1–Q3 sí estiman en días. Conviene fijar la unidad en el prompt.
- **`answer_relevancy` bajo (0.12).** Hipótesis: la "pregunta" es un *brief* declarativo y la
  "respuesta" una tabla módulos→tareas, formato poco adecuado para una métrica que regenera
  preguntas a partir de la respuesta; sería un artefacto de medición más que señal de calidad.
- **`faithfulness` 0.49 pese a citación correcta.** El generador descompone cada componente
  histórico en subtareas y reparte horas; la cita por línea existe (0 colgantes), pero el juez no
  puede atribuir 1:1 cada cifra derivada a la cifra del `<source>`. La citación verificable separa
  "la cita existe" de "la cifra se deduce de la cita".
