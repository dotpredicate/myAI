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
- `./images.py` - image generation routes
- `./migrations/` - folder with migration scripts
- `./frontend/` - web front-end sources (`public/`, `public/ts/`) and JS output (`public/js/`)
- `./dev.sh` - development entrypoint (frontend watch + `fastapi dev`)
- `./doc/` - design and documentation
  - `context.md`
  - `repositories.md`
  - `inference_providers.md`
  - `agents.md`

### Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. npm --prefix frontend install
4. ./dev.sh

`./dev.sh` requires the active virtualenv and `setsid`; Ctrl+C twice forces `KILL`.

Server runs at http://127.0.0.1:8000

### Front-end
`frontend/public/` is served directly by FastAPI; `public/ts/` → `tsc` → `public/js/` (git-ignored). No bundler.

```
./dev.sh                        # build + tsc watch + fastapi dev
npm --prefix frontend run build # one-shot build
npm --prefix frontend run check # type-check
```

`tsc` emits despite type errors; `check` is the gate.

### Migrations
At startup, the module scans the `migrations` folder for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table

### Code style
- Use explicit Python typing
- Use explicit TypeScript typing
- Build UI from `frontend/public/ts/components.ts`; use `el()` only for one-off markup
- The use of `style` attribute is forbidden
- Lint and type-check with `./lint.sh`
- Test with `./test.sh`

### Simplifications for now
- the only available language model model provider is embedded llama.cpp server
- image generation uses an embedded stable-diffusion.cpp server; models are user-registered presets in `sd_cpp_image_models` (recipe + file IDs from `MYAI_SD_MODEL_DIRS`)
- web UI only, no mobile, minimal styling
- no accessibility work: no ARIA attributes or roles, no screen-reader tuning, no keyboard-only flows beyond what the browser gives for free
- single user, no log-in
- no cloud integrations
- monolithic back-end, single instance
- no caching like Redis, prefer in-memory structures
