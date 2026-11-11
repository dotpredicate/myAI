myAI

## Project description

### Goals
- Private and secure place to use AI
- Personal cloud - a centralized place for data
- A great search engine
- Simple and powerful Agent creation
- Scales with model capability

### Guidelines
- Keep it simple
- Just use SQLite
- Move fast and break things

## Technical info

Project structure:
- `./app.py` - FastAPI routes and application entrypoint
- `./myai-server` - server launcher
- `./conversation.py` - conversation state management and orchestration
- `./domain.py` - conversation model
- `./scopes.py` - repository scopes of a workspace, used by conversations
- `./repositories.py` - repository registry and indexing input
- `./agents.py` - agent configuration and optional model preference
- `./workspaces.py` - workspace sources and path resolution
- `./search/` - indexing and vector search
- `./database.py` - SQLite connection and utility functions
- `./paths.py` - `MYAI_HOME`, database, model space and sandbox paths
- `./system.py` - model spaces, sandboxed shell and sandbox path checks
- `./system_metrics.py` - host CPU, memory and GPU usage sampling
- `./gpu_metrics.py` - ROCm GPU data shared by system and per-process metrics
- `./tools.py` - tool execution logic
- `./log_config.py` - logging configuration
- `./inference/` - AI protocol handling (stateless), split into submodules
- `./images.py` - image generation routes
- `./migrations/` - folder with migration scripts
- `./frontend/` - front-end sources and scripts (one npm project)
  - `./frontend/web/` - browser client, served by FastAPI; the HTTP transport and wire types shared with the TUI live in `web/ts/core/api.ts` (TUI imports it as `#core/api`)
  - `./frontend/tui/` - terminal client (Node, runs from source)
- `./build.sh` - build the front-end
- `./install.sh` - install the terminal client as `myai` and the server as `myai-server`
- `./dev.sh` - development entrypoint (live reload)
- `./lint.sh` - lint and type-check
- `./test.sh` - test runner
- `./tests/` - unit, integration and e2e tests
- `./doc/` - design and documentation
  - `workspaces.md`
  - `repositories.md`
  - `agents.md`
  - `conversations.md`
  - `environment.md`
  - `inference.md`
  - `analysis/image_backends.md`

### Quickstart
1. source .venv/bin/activate
2. pip install -r requirements.txt
3. npm --prefix frontend install
4. ./dev.sh

`./dev.sh` requires the active virtualenv and `setsid`; Ctrl+C twice forces `KILL`.

### Server
`./myai-server` runs the FastAPI app with uvicorn on `127.0.0.1:8000`; `MYAI_PORT` overrides
the port, and `myai-server stop` stops it. `GET /api/info` reports the process id, which
clients use to recognize a running server.

### Front-end
`frontend/web/` is served directly by FastAPI; `frontend/web/ts/` -> `tsc` -> `frontend/web/js/`. No bundler, one `tsc` project; the TUI is not compiled at all (Node runs its `.ts` sources).

### TUI
`frontend/tui/` is a Node client; TypeScript runs from source (no build step, no runtime
dependencies). It starts the server if needed and resolves the current directory
into a workspace.

```sh
node frontend/tui/src/main.ts                 # interactive
node frontend/tui/src/main.ts --dump [conv]   # render one frame without a TTY
```


### Migrations
At startup, the module scans `migrations` for SQL files, sorts them in lexicographic order, executes those that have not yet been applied and records the execution in the `migrations` table.

### Data location
`MYAI_HOME` defaults to `~/.myai`; relative `.env` paths resolve from the repo root.

```text
$MYAI_HOME/
├── db.sqlite
├── spaces/       # durable model space per workspace
└── sandboxes/    # ephemeral space per conversation
```

### Code style
- Use explicit Python typing
- Use explicit TypeScript typing
- When building UI, reuse the shared components (the `components` module)
- The use of `style` attribute is forbidden
- Lint and type-check with `./lint.sh` (ruff, mypy, tsc, Oxlint)
- Test with `./test.sh`

### Simplifications for now
- language model providers: the embedded llama.cpp server and user-configured OpenAI-compatible endpoints
- image generation uses an embedded stable-diffusion.cpp server; models are user-registered presets in `sd_cpp_image_models` (recipe + file IDs from `MYAI_SD_MODEL_DIRS`)
- web UI and a terminal client, no mobile, minimal styling
- no accessibility work: no ARIA attributes or roles, no screen-reader tuning, no keyboard-only flows beyond what the browser gives for free
- single user, no log-in
- no built-in cloud sync; external model providers are opt-in
- monolithic back-end, single server instance
- no caching like Redis, prefer in-memory structures
