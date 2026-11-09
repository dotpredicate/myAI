# myAI TUI

Terminal client for a running myAI instance. Node runs the TypeScript sources
directly (type stripping), so there is no build step and no runtime dependency.

```sh
./install.sh            # from the repo root: installs the `myai` command
myai                    # interactive
myai --dump             # render one frame without a TTY
myai --dump 42
```

Running from source needs nothing else (`node frontend/tui/src/main.ts`); `npm --prefix
frontend install` is only needed for `tsc` and `oxlint`.

Environment:

- `MYAI_URL` — server base URL (default `http://127.0.0.1:8000`)
- `MYAI_PROVIDER`, `MYAI_MODEL` — preselected provider and model

Keys:

- list: `↑↓` select, `enter` open, `n` new, `d` delete, `r` refresh, `q` quit
- new: `↑↓` repository, `space` toggle, `p` provider, `m` model, `enter` start, `esc` back
- chat: text + `enter` send, `↑↓` scroll, `esc` list, `ctrl-c` cancel generation

Layout, styling and the component vocabulary live in `src/h.ts`, `src/render.ts`
and `src/components.ts` and mirror `frontend/web/ts/{dom,components}.ts`: the
same `Row`/`Stack`/`Paragraph`/`Spacer`/`Box`/`Badge` names, rendered as text
cells instead of DOM nodes. Transport and wire types come from the shared
`frontend/web/ts/core/api.ts`, imported as `#core/api` (a Node subpath import
declared in `frontend/package.json`); the transcript reducer is local to `src/`.
