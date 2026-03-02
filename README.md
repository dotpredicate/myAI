myAI

Like Open WebUI but better suited for me.

Goals
- integrate Ollama for running local models
- create a great RAG / DB for AI
- define agent workflows
- add secure permissions and policy model

Guidelines:
- keep it simple
- just works (tm)
- secure and private

Simplifactions for now:
- web and desktop UI only, no mobile, minimal styling
- just use postgres
- single user, no log-in
- no cloud integrations
- no caching, etc. use in-memory structures

Code style:
- use python types

Migrations
- All database changes live under the `migrations/` directory. Each file is a plain `.sql` file named with a sequential number (e.g. `000_create_users.sql`, `001_add_index.sql`).
- At startup, `app.py` automatically scans the `migrations/` folder, executes any SQL files that have not yet been applied, and records the execution in the `migrations` table.
- To add a new migration, create a new sequential SQL file and restart the application (or trigger `init_database()`), and it will be applied automatically.
- No external migration tool or script is required.

Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. python app.py

Server runs at http://127.0.0.1:5000
