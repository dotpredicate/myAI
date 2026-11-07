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
- `./scopes.py` - repository scope resolution shared by conversations and search
- `./repositories.py` - repository registry and indexing input
- `./agents.py` - agent configuration and repository access
- `./search/` - indexing and vector search (`__init__.py`, `documents.py`)
- `./database.py` - database connection and utility functions
- `./system.py` - system-level utilities (workspaces, sandbox)
- `./tools.py` - tool execution logic
- `./log_config.py` - logging configuration
- `./inference/` - AI protocol handling (stateless), split into submodules
- `./images.py` - image generation routes
- `./migrations/` - folder with migration scripts
- `./frontend/` - web front-end sources (`public/`, `public/ts/`) and JS output (`public/js/`)
- `./dev.sh` - development entrypoint (frontend watch + `fastapi dev`)
- `./lint.sh` - lint and type-check
- `./test.sh` - test runner
- `./tests/` - unit, integration and e2e tests
- `./doc/` - design and documentation
  - `context.md`
  - `repositories.md`
  - `inference_providers.md`
  - `agents.md`
  - `generation.md`
  - `analysis/image_backends.md`

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
npm --prefix frontend run lint  # type-aware Oxlint
```

`tsc` emits despite type errors; `check` is the gate. `lint` runs Oxlint with the type-aware `typescript/no-unnecessary-condition` rule.

### Search
The index lives in `search/`. `semantic_search` takes an explicit list of resolved repository scopes; an empty list returns no results and never falls back to the whole index. `/api/search` resolves scopes through `scopes.resolve_scopes`: explicit scopes plus every repository the `agent_id` can access. Results carry `cosine_distance` (lower is closer) and are ordered closest first.

### Migrations
At startup, the module scans the `migrations` folder for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table

### Code style
- Use explicit Python typing
- Use explicit TypeScript typing
- Build UI from `frontend/public/ts/components.ts`; use `el()` only for one-off markup
- The use of `style` attribute is forbidden
- Lint and type-check with `./lint.sh` (ruff, mypy, tsc, Oxlint)
- Test with `./test.sh`

### Simplifications for now
- the only available language model provider is the embedded llama.cpp server
- image generation uses an embedded stable-diffusion.cpp server; models are user-registered presets in `sd_cpp_image_models` (recipe + file IDs from `MYAI_SD_MODEL_DIRS`)
- web UI only, no mobile, minimal styling
- no accessibility work: no ARIA attributes or roles, no screen-reader tuning, no keyboard-only flows beyond what the browser gives for free
- single user, no log-in
- no cloud integrations
- monolithic back-end, single instance
- no caching like Redis, prefer in-memory structures
