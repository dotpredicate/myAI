myAI

Like Open WebUI but better suited for me.

## Project description

### Goals
- integrate llama.cpp for running local models
- create a great RAG / DB / search engine for AI
- simple and powerful agent workflow creation
- add permissions, security policy and a safety model

### Guidelines
- keep it simple
- just works (tm)
- secure and private
- just use postgres

## Technical info

Project structure:
- `./app.py` - FastAPI routes and application entrypoint
- `./conversation.py` - conversation state management and orchestration
- `./inference.py` - AI protocol handling (stateless)
- `./index.py` - search indexing and vector search logic
- `./database.py` - database connection and utility functions
- `./documents.py` - document processing and management
- `./system.py` - system-level utilities
- `./tools.py` - tool execution logic
- `./migrations/` - folder with migration scripts
- `./static/` - static resources deployed by the server, web front-end

### Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. python app.py

Server runs at http://127.0.0.1:5000

### Migrations
At startup, the module scans the `migrations` folder for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table

### Code style
- use Python types

### Simplifications for now
- web UI only, no mobile, minimal styling
- single user, no log-in
- no cloud integrations
- monolithic back-end, single instance
- no caching like Redis, prefer in-memory structures
