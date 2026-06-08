myAI

Like Open WebUI but better suited for me.

## Project description

### Goals
- Private and secure place for local model deployment
- Personal cloud - a centralized place for your data
- A great search engine
- Simple and powerful Agent creation

### Guidelines
- Secure and private
- Just works (tm)
- Keep it simple
- Just use postgres
- Move fast and break things

## Technical info

Project structure:
- `./app.py` - FastAPI routes and application entrypoint
- `./conversation.py` - conversation state management and orchestration
- `./domain.py` - conversation model
- `./index.py` - search indexing and vector search logic
- `./database.py` - database connection and utility functions
- `./documents.py` - document processing and management
- `./system.py` - system-level utilities
- `./tools.py` - tool execution logic
- `./log_config.py` - logging configuration
- `./inference/` - AI protocol handling (stateless), split into submodules
- `./migrations/` - folder with migration scripts
- `./static/` - static resources deployed by the server, web front-end
- `./doc/` - design and documentation
  - `conversation.md`
  - `repositories.md`
  - `inference_providers.md`
  - `agents.md`

### Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. fastapi dev

Server runs at http://127.0.0.1:8000

### Migrations
At startup, the module scans the `migrations` folder for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table

### Code style
- Use explicit Python typing
- Lint with `mypy .` and `ruff check`
- The use of `style` attribute is forbidden

### Simplifications for now
- the only available inference provider is embedded llama.cpp server
- web UI only, no mobile, minimal styling
- single user, no log-in
- no cloud integrations
- monolithic back-end, single instance
- no caching like Redis, prefer in-memory structures
