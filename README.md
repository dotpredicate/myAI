myAI

Like Open WebUI but better suited for me.

Goals
- integrate Ollama for running local models
- create a great RAG / DB for AI
- define agent workflows
- add secure permissions and policy model

Guidelines
- keep it simple
- just works (tm)
- secure and private
- just use postgres


Simplifications for now
- web UI only, no mobile, minimal styling
- single user, no log-in
- no cloud integrations
- monolithic back-end, single instance
- no caching like Redis, prefer in-memory structures
- Ollama's /api/chat requires aggregating thinking, content, and tool_calls into a single message, which prevents interleaving of reasoning and actions

Code style
- use Python types

Migrations
- at startup, the module scans the `migrations/` folder for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table.

Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. python app.py

Server runs at http://127.0.0.1:5000
