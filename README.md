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
- `./migrations` - folder with migration scripts
- `./app.py` - entrypoint to the Python web server
- `./static` - static resources deployed by the server, web front-end

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
- function calls and shell execution aren't secured