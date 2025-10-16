# Memory Models

Jarvis works with two flavours of memory services. Both satisfy the `MemoryStore` interface consumed by the orchestrator.

## Primary Store: `MemoryStore`

- **Engine**: PostgreSQL with pgvector extension, configured via `POSTGRES_DSN`.
- **Pooling**: `POSTGRES_POOL_SIZE` (default 10) controls async connection pooling.
- **Usage**: Orchestrator calls `recall` → `guard` → `write` in sequence, allowing for moderation or dedup logic in the guards.
- **Tip**: Run `Documentation/Postgres_pgvector_Setup.md` to bootstrap the schema if you have not already.

## Null Store: `NullMemoryStore`

- Activated automatically when the primary DSN fails during bootstrap.
- Methods no-op but maintain the same async interface so the rest of the pipeline keeps flowing.
- Helpful for demos where you do not want to spin up PostgreSQL.

## Persona Metadata

- Personas are not stored in the DB—they live in memory via the new `PersonaManager`.
- If you want persistent persona preferences per user, extend the memory store to add a persona table keyed by device or user ID, then load it before `PersonaManager.initialize()`.

## Best Practices

- Keep `guard` logic fast; it sits on the critical path after every LLM response.
- Use PG bloat checks periodically if you store long transcripts or audio references.
- For high availability, wrap the `MemoryStore` constructor in retry logic or supply a secondary DSN via environment variables.
