# Retrieval evaluation: precision@5 and latency

Latency includes retrieval and reranking, but excludes embeddings, model loading and HTTP. Each query/configuration has one discarded warm-up and three measured runs.

Relevance is budget-level, precision is chunk-level: multiple chunks from one relevant budget count individually. Q2/Q3 have only four relevant chunks and Q4 only three, so their maximum precision@5 is 0.8/0.8/0.6.

The shipped corpus, queries and full-text configuration are English.

| Config | Search | Reranking | Precision@5 | Latency (ms) |
| --- | --- | --- | --- | --- |
| A | Vector | No | 0.84 | 7.6 |
| B | Hybrid | No | 0.84 | 8.3 |
| C | Vector | Yes | 0.80 | 579.0 |
| D | Hybrid | Yes | 0.80 | 586.8 |

## Per-query precision@5

| Query | A | B | C | D |
| --- | --- | --- | --- | --- |
| Q1 | 1.00 | 1.00 | 0.80 | 0.80 |
| Q2 | 0.80 | 0.80 | 0.80 | 0.80 |
| Q3 | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4 | 0.60 | 0.60 | 0.60 | 0.60 |
| Q5 | 1.00 | 1.00 | 1.00 | 1.00 |

## Golden set annotations

| Query | Project | Relevant budgets |
| --- | --- | --- |
| Q1 | Mobile banking application with OAuth2 authentication, PSD2 open banking compliance, strong customer authentication and a transaction ledger. | BUD-2024-001, BUD-2024-003 |
| Q2 | Headless e-commerce storefront with a product catalog, shopping cart checkout and personalized product recommendations. | BUD-2024-005 |
| Q3 | Telemedicine platform for patient appointment scheduling and video consultations with HL7/FHIR integration and clinical records. | BUD-2024-009 |
| Q4 | Industrial IoT platform ingesting sensor telemetry for predictive maintenance and real-time equipment monitoring in a factory. | BUD-2024-013 |
| Q5 | Real-time payments gateway with fraud detection, transaction reconciliation and a double-entry ledger. | BUD-2024-003, BUD-2024-001 |

## Lexical branch diagnostics

| Query | Matching chunks |
| --- | --- |
| Q1 | 0 |
| Q2 | 0 |
| Q3 | 0 |
| Q4 | 0 |
| Q5 | 0 |

No lexical matches were found. Hybrid rankings therefore equal vector rankings before reranking. plainto_tsquery requires every non-stop-word term in each long project description to match in a single chunk.

## Conclusions

For the current project, we would use configuration A (vector search without reranking): it achieves the highest measured precision@5 (0.84, tied with B) with the lowest mean retrieval latency (7.6 ms). Hybrid search adds no relevant results because its lexical branch finds no matches, while reranking reduces precision to 0.80 and raises latency to roughly 580 ms, about 76 times the baseline. Although half a second might be acceptable within an LLM-based estimation workflow, there is no measured relevance gain to justify that cost here. This recommendation is provisional: five queries over a small corpus are limited evidence, and budget-level annotations do not measure the usefulness of individual components or final estimate quality. We would revisit hybrid matching and reranking after evaluating a larger, more varied golden set.
