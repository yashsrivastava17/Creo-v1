# Postgres + pgvector Quick Start (Local Docker)

1. **Run the container**

   ```bash
   docker run --name jarvis-pg \
     -e POSTGRES_USER=jarvis \
     -e POSTGRES_PASSWORD=jarvis \
     -e POSTGRES_DB=jarvis \
     -p 5432:5432 \
     ankane/pgvector:latest
   ```

   This image ships with the `pgvector` extension pre-installed and matches the DSN in `.env.example`.

2. **Create the extension** (first run only)

   ```bash
   psql -h localhost -U jarvis -d jarvis -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Point the app at the database**

   Ensure `.env` contains `POSTGRES_DSN=postgresql+asyncpg://jarvis:jarvis@localhost:5432/jarvis` and keep `POSTGRES_POOL_SIZE` modest (e.g., 5–10) when running on a laptop.

4. **Verify connectivity**

   Start the voice agent (`uvicorn jarvis.main:app --reload`) and watch the logs for `memory` table creation. The ORM automatically applies the schema on startup.

5. **Maintenance**

   - `docker stop jarvis-pg` to pause
   - `docker start jarvis-pg` to resume
   - `docker logs jarvis-pg` if you need visibility into Postgres output

If you prefer a native install, install Postgres 15+, `CREATE EXTENSION vector`, and reuse the same DSN pointing to `localhost`.
