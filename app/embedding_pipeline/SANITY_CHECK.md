# Embedding sanity check

Model: `text-embedding-3-small` (1536 dimensions, default dimension).
Generated with `scripts/compare.py`.
Run date: 2026-10-18.

| Pair | Text A | Text B | Cosine | Expectation |
|------|--------|--------|--------|-------------|
| A — near | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.5957** | high (> 0.6) |
| B — unrelated | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.1920** | low (< 0.4) |
| C — generic | "Backend services" | "API development" | **0.5407** | no fixed expectation |

## Commentary

A scores 0.5957, just below the orientative 0.6 threshold despite the shared authentication and banking concepts.  
B scores 0.1920, as expected for unrelated topics.  
C reaches 0.5407, close to A: short, generic phrases can appear semantically close while carrying little distinguishing context, making it a useful discussion point for the live session.
