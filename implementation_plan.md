# Implementation Plan

Update the scope/scope selection system so that the front-end sends structured scope objects `{internal_name, security_policy_override}` instead of bare string lists `list[str]`, and the back-end validates, persists, and enforces per-scope security policy overrides — preventing users from escalating privileges beyond the repository's configured security policy.

[Overview]

The chat view currently maintains `activeScopes` as a plain `list[str]` of repository internal names. When `sendPrompt()` builds the request body, it sends `scopes: activeScopes` — a flat string array. The back-end (`app.py::prompt_model()`) ignores the incoming `scopes` entirely in manual mode (sets `scopes = []`). The new design requires:

1. A scope to be `{ internal_name: str, security_policy_override: str | null }`.
2. The scope pill UI to show the effective policy and allow choosing an override ≤ the repo's base policy.
3. The back-end to accept this structured format, persist it in the user message's `scopes` field, and apply the override when resolving repository permissions during tool execution.
4. The agent form's repository access policy selector to enforce the same non-escalation rule.

[Types]

Define a new typed structure for scope objects (used both in JS and Python):

```python
# domain.py addition or as a standalone model:

class ScopeSpec(BaseModel):
    internal_name: str
    security_policy_override: SecurityPolicy | None = None
```

The `Message.scopes` field changes from `list[str]` to `list[str | ScopeSpec]` (backward-compatible: plain strings are treated as `{internal_name: s, security_policy_override: None}`).

```javascript
// Front-end conceptual type:
// ScopeSpec = { internalName: string, securityPolicyOverride: string | null }
```

[Files]

Modify `domain.py` — add `ScopeSpec` model; update `Message.scopes` type annotation to accept both `str` and `ScopeSpec`.

Modify `static/index.html` — the entire scope selection + rendering + send logic.

Modify `app.py` — update `prompt_model()` to accept structured scopes; re-enable scopes for manual mode.

Modify `conversation.py` — update `get_messages_for_continuation()` and `get_scopes_from_last_user_message()` to normalize scopes (convert plain strings to `ScopeSpec`); pass effective policy to tool calls.

Modify `system.py` — update `run_sandboxed_command()` to use `security_policy_override` from scope spec if present.

Modify `agents.py` — add validation in `create_agent()` and `update_agent()` to reject policy escalation.

No new files.

[Functions]

1. **`domain.py` — Add `ScopeSpec` model**  
   Pydantic BaseModel with `internal_name: str` and `security_policy_override: Optional[SecurityPolicy]`.  
   Update `Message.scopes` to accept `list[str | ScopeSpec]`.  
   Add a helper `normalize_scopes(scopes: list) -> list[ScopeSpec]`.

2. **`static/index.html` — Update `renderScopePills()`**  
   Show policy badge next to scope name.  
   Show (default) if no override.

3. **`static/index.html` — Update scope selector**  
   When adding a scope from the dropdown, show a policy override dropdown populated with allowed values (≤ repo's policy). Include a "Use default" option (null).  
   Store each scope as `{internalName, securityPolicyOverride}`.

4. **`static/index.html` — Update `sendPrompt()`**  
   Send `scopes` as `[{internal_name, security_policy_override}]`.

5. **`app.py::prompt_model()`**  
   Parse `scopes` from request body as `list[dict]` (not ignored).  
   Normalize via `normalize_scopes()`.  
   Pass to `prepare_conversation_with_prompt()`.

6. **`conversation.py::prepare_conversation_with_prompt()`**  
   Already accepts `scopes: Optional[List[str]]` — update to accept `Optional[List[ScopeSpec]]` and store in `Message(scopes=scopes)`.

7. **`conversation.py::get_messages_for_continuation()`**  
   Already extracts scopes from user message — update return type to `list[ScopeSpec]`.

8. **`conversation.py::get_scopes_from_last_user_message()`**  
   Return `list[ScopeSpec]`.

9. **`conversation.py::continue_conversation()`**  
   Pass `scopes` (as `list[ScopeSpec]`) to `run_tool_call()`.

10. **`tools.py::run_tool_call()`**  
    Accept `scopes: list[ScopeSpec]`. Pass to downstream functions.

11. **`system.py::run_sandboxed_command()`**  
    For each scope spec, get repo. If `scope.security_policy_override` is set, use that instead of `repo.security_policy` — but only if it's ≤ the base policy (enforced at UI + API level).

12. **`agents.py::create_agent()` / `update_agent()`**  
    Validate each `repository_access` entry: if `security_policy_override` is provided, ensure it is ≤ the repo's `security_policy` (i.e., not an escalation). Return `400` if invalid.

[Dependencies]

No new packages. Only internal type changes.

[Testing]

Manual testing:
- Open chat, add a scope with policy override, send a prompt — verify scopes are sent correctly and persisted.
- Add a scope with a higher policy than the repo allows — verify UI prevents it (dropdown filtering) and API rejects it if bypassed.
- Use agent with repo access override — verify no regression.
- Verify scope pills show correct policy badge.

[Implementation Order]

1. Add `ScopeSpec` model to `domain.py` + `normalize_scopes()` helper.
2. Update `system.py::run_sandboxed_command()` to handle `ScopeSpec`.
3. Update `conversation.py` functions to work with `ScopeSpec`.
4. Update `app.py::prompt_model()` to accept structured scopes.
5. Add agent repo-access escalation validation in `agents.py`.
6. Update `static/index.html` scope selection, pills, and send logic.
7. Test.