#!/usr/bin/env node
// myAI TUI: workspaces, conversations, streaming, tool-call approvals.
//
//   node src/main.ts                 # interactive
//   node src/main.ts --dump [id]     # render one frame and exit (no TTY needed)
//   MYAI_URL=http://host:port ...    # explicit server; disables autostart

import {
  ApiError,
  api,
  openNdjson,
  type AgentConfig,
  type ConversationDetails,
  type ConversationSummary,
  type CreateConversationRequest,
  type ProviderModel,
  type ResolveWorkspaceResponse,
  type StreamEvent,
  type ToolCallDecisionResponse,
  type WorkspaceConfig,
  type WorkspaceList,
} from './wire.ts';
import { Badge, Box, Heading, Hint, KeyHint, Line, Paragraph, Row, Spacer, Stack, Text } from './components.ts';
import { applyStreamEvent, applyToolResult, findTool, transcriptFromMessages, truncate, type TranscriptItem } from './transcript.ts';
import { Screen, layout, paint, type Line as RenderedLine } from './render.ts';
import { startInput, type InputHandle, type Key } from './input.ts';
import { handlePickerKey, openPicker, pickerFrame, type Picker } from './picker.ts';
import { modelChoices, type PickItem } from '#core/models';
import { resolveServer } from './service.ts';
import { type VNode } from './h.ts';

let BASE = '';

function url(path: string): string {
  return `${BASE}${path}`;
}

/* -------------------------------------------------------------------- model */

type View = 'list' | 'new' | 'chat';

interface State {
  view: View;
  workspace: WorkspaceConfig | null;

  conversations: ConversationSummary[];
  selectedConversation: number;
  scroll: number;

  agents: AgentConfig[];
  agentIndex: number;

  providers: string[];
  providerIndex: number;
  models: ProviderModel[];
  modelIndex: number;
  modelsLoading: boolean;

  conversationId: number | null;
  title: string | null;
  items: TranscriptItem[];
  buffer: string;
  busy: boolean;
  blockingId: number | null;
  lastSequence: number;
  status: string;
  error: string;
}

const state: State = {
  view: 'list',
  workspace: null,
  conversations: [],
  selectedConversation: 0,
  scroll: 0,
  agents: [],
  agentIndex: 0,
  providers: [],
  providerIndex: 0,
  models: [],
  modelIndex: 0,
  modelsLoading: false,
  conversationId: null,
  title: null,
  items: [],
  buffer: '',
  busy: false,
  blockingId: null,
  lastSequence: 0,
  status: '',
  error: '',
};

const screen = new Screen(process.stdout);
let input: InputHandle | null = null;
let streamToken = 0;
let modelsRequest = 0;
let nextMessageStartsNew = false;
const dumpMode = process.argv.includes('--dump');

if (process.argv.includes('--help') || process.argv.includes('-h')) {
  process.stdout.write(
    [
      'myai — myAI terminal client',
      '',
      'Usage: myai [--dump [conversation-id]] [--help]',
      '',
      '  --dump [id]   render one frame and exit (works without a TTY)',
      '  --help        show this help',
      '',
      'Environment:',
      '  MYAI_URL        explicit server base URL; disables autostart',
      '  MYAI_PORT       port for the local server (default 8000)',
      '  MYAI_PROVIDER   preselected provider key',
      '  MYAI_MODEL      preselected model id',
      '',
    ].join('\n'),
  );
  process.exit(0);
}

function columns(): number {
  return Math.max(20, process.stdout.columns || 100);
}

function rows(): number {
  return Math.max(3, process.stdout.rows || 30);
}

/* ----------------------------------------------------------------- renders */

function renderItems(items: TranscriptItem[]): VNode[] {
  return items.map((item) => {
    switch (item.kind) {
      case 'user':
        return Box({ title: 'you' }, Paragraph(Text(item.text)));
      case 'assistant':
        return Paragraph(Text(item.text || ' '));
      case 'thinking':
        return Paragraph(Hint(item.text));
      case 'tool':
        return Box(
          { title: `tool: ${item.name}` },
          Paragraph(Hint(item.parameters)),
          item.proposal === null ? null : Paragraph(Hint(truncate(item.proposal, 800))),
          item.result === null ? null : Paragraph(Text(`result: ${truncate(item.result, 800)}`)),
        );
      case 'error':
        return Paragraph(Text(item.text, { color: 'red' }));
      case 'notice':
      default:
        return Paragraph(Hint(item.text));
    }
  });
}

function footer(hints: VNode[]): RenderedLine[] {
  const width = columns();
  return layout(Row(...hints, Spacer(), Hint(state.status)), width);
}

function listScreen(): string[] {
  const width = columns();
  const height = rows();
  const body: VNode[] = [
    Row(Heading('myAI'), Spacer(), Hint(state.workspace ? `workspace: ${state.workspace.name}` : 'no workspace'), Text('  '), Hint(BASE)),
    Line(),
  ];
  if (state.workspace) {
    body.push(Paragraph(Hint(`sources: ${state.workspace.sources.map((source) => source.internal_name).join(', ') || '(none)'}`)));
    body.push(Line());
  }
  if (state.conversations.length === 0) body.push(Paragraph(Hint('no conversations yet — press n')));
  state.conversations.forEach((conversation, index) => {
    const selected = index === state.selectedConversation;
    const title = conversation.title?.trim() || 'untitled';
    body.push(
      Row(
        Text(selected ? '❯ ' : '  ', { reverse: selected }),
        Text(`#${conversation.id} `, { dim: true }),
        Text(title, { reverse: selected }),
        Spacer(),
        Text(conversation.created_at.slice(0, 19).replace('T', ' '), { dim: true }),
      ),
    );
  });
  body.push(
    Line(),
    Row(
      KeyHint('enter', 'open'),
      Text('  '),
      KeyHint('n', 'new'),
      Text('  '),
      KeyHint('d', 'delete'),
      Text('  '),
      KeyHint('r', 'refresh'),
      Text('  '),
      KeyHint('q', 'quit'),
    ),
  );
  if (state.error) body.push(Paragraph(Text(state.error, { color: 'red' })));

  return paint(layout(Stack(...body), width), width, height);
}

function selectedAgent(): AgentConfig | null {
  return state.agents[state.agentIndex - 1] ?? null;
}

function newScreen(): string[] {
  const width = columns();
  const height = rows();
  const provider = state.providers[state.providerIndex] ?? '—';
  const model = state.models[state.modelIndex]?.id ?? (state.modelsLoading ? 'loading…' : '—');
  const agent = selectedAgent();
  const agentLabel = agent ? `${agent.display_name}${agent.provider_key ? ` (${agent.model_id ?? 'model'})` : ' (model per chat)'}` : 'none';
  const body: VNode[] = [
    Row(Heading('new conversation'), Spacer(), Hint(state.workspace ? state.workspace.name : '')),
    Line(),
    Row(Text('agent    ', { dim: true }), Text(agentLabel), Text('  '), KeyHint('a', 'cycle')),
    Row(Text('provider ', { dim: true }), Text(provider), Text('  '), KeyHint('p', 'choose')),
    Row(Text('model    ', { dim: true }), Text(model), Text('  '), KeyHint('m', 'choose')),
    Line(),
    Paragraph(Hint('sources (fixed by workspace):')),
  ];
  for (const source of state.workspace?.sources ?? []) {
    body.push(Paragraph(Text(`${source.internal_name}  ${source.path}`, { dim: true })));
  }
  body.push(
    Line(),
    Row(KeyHint('enter', 'start'), Text('  '), KeyHint('esc', 'back')),
  );
  if (state.error) body.push(Paragraph(Text(state.error, { color: 'red' })));

  return paint(layout(Stack(...body), width), width, height);
}

function chatScreen(): string[] {
  const width = columns();
  const height = rows();
  const transcriptHeight = Math.max(3, height - 3);

  const header = layout(
    Row(
      Heading(state.title?.trim() || `conversation #${state.conversationId ?? 'new'}`),
      Spacer(),
      state.workspace ? Hint(state.workspace.name) : null,
      state.blockingId === null ? null : Badge('approval needed'),
      state.busy ? Badge('generating') : null,
    ),
    width,
  );

  const items = renderItems(state.items);
  const tool: VNode[] = items;
  if (state.blockingId !== null) {
    const pending = findTool(state.items, state.blockingId);
    tool.push(
      Box(
        { title: 'approval required' },
        Paragraph(Text(`tool: ${pending?.name ?? 'unknown'}`, { bold: true })),
        pending === null ? Paragraph(Hint('this tool call is not in the transcript')) : Paragraph(Hint(pending.parameters)),
        Paragraph(KeyHint('y', 'approve and continue'), Text('  '), KeyHint('n', 'reject and continue')),
      ),
    );
  }
  const lines = layout(Stack(Paragraph(Text(' ')), ...tool), width);
  const maxScroll = Math.max(0, lines.length - transcriptHeight);
  if (state.scroll > maxScroll) state.scroll = maxScroll;
  const end = lines.length - state.scroll;
  const start = Math.max(0, end - transcriptHeight);
  const window = lines.slice(start, end);

  const inputRow = layout(
    Row(
      Text(state.blockingId === null ? '› ' : '! ', { bold: true, color: state.blockingId === null ? 'cyan' : 'yellow' }),
      Text(state.buffer, {}),
      Text(state.busy ? '▌' : ''),
    ),
    width,
  );

  const hints: VNode[] = state.blockingId !== null
    ? [KeyHint('y/n', 'decide')]
    : [KeyHint('enter', 'send'), Text('  '), KeyHint('↑↓', 'scroll'), Text('  '), KeyHint('alt-m', 'model'), Text('  '), KeyHint('alt-p', 'provider'), Text('  '), KeyHint('ctrl-c', 'cancel'), Text('  '), KeyHint('esc', 'list')];

  const out = [
    ...paint(header, width, 1),
    ...paint(window, width, transcriptHeight),
    ...paint(inputRow, width, 1),
    ...paint(footer(hints), width, 1),
  ];
  return out;
}

let picker: { view: Picker; choose: (value: string) => void } | null = null;

function openPickerOverlay(title: string, items: PickItem[], onChoose: (value: string) => void): void {
  picker = { view: openPicker(title, items), choose: onChoose };
  render();
}

function closePicker(): void {
  picker = null;
  render();
}

function openProviderPicker(): void {
  openPickerOverlay('provider', state.providers.map((key) => ({ value: key, label: key })), (value) => {
    const index = state.providers.indexOf(value);
    if (index >= 0 && index !== state.providerIndex) {
      state.providerIndex = index;
      state.status = `provider: ${value}`;
      fire(loadModels());
    }
  });
}

function openModelPicker(): void {
  if (state.models.length === 0) {
    state.status = state.modelsLoading ? 'models are loading' : 'no models for this provider';
    render();
    return;
  }
  const provider = state.providers[state.providerIndex] ?? '—';
  openPickerOverlay(`model · ${provider}`, modelChoices(state.models), (value) => {
    const index = state.models.findIndex((model) => model.id === value);
    if (index >= 0) {
      state.modelIndex = index;
      state.status = `model: ${value}`;
    }
  });
}

function currentFrame(): string[] {
  if (picker) return pickerFrame(picker.view, columns(), rows());
  if (state.view === 'list') return listScreen();
  if (state.view === 'new') return newScreen();
  return chatScreen();
}

let frameQueued = false;
function render(): void {
  if (dumpMode) return;
  if (frameQueued) return;
  frameQueued = true;
  queueMicrotask(() => {
    frameQueued = false;
    screen.draw(currentFrame());
  });
}

/* ------------------------------------------------------------------- data */

function rebuildList(): void {
  if (state.selectedConversation >= state.conversations.length) {
    state.selectedConversation = Math.max(0, state.conversations.length - 1);
  }
}

async function loadConversations(): Promise<void> {
  if (!state.workspace) return;
  state.conversations = await api.get<ConversationSummary[]>(
    url(`/api/conversations?workspace_id=${state.workspace.id}`),
  );
  rebuildList();
  render();
}

async function loadAgents(): Promise<void> {
  const data = await api.get<{ agents: AgentConfig[] }>(url('/api/agents'));
  state.agents = data.agents;
  state.agentIndex = Math.min(state.agentIndex, state.agents.length);
  render();
}

async function loadProviders(): Promise<void> {
  const data = await api.get<{ providers: Array<{ key: string }> }>(url('/api/providers'));
  state.providers = data.providers.map((provider) => provider.key);
  const preferred = process.env.MYAI_PROVIDER ?? state.providers[0] ?? '';
  state.providerIndex = Math.max(0, state.providers.indexOf(preferred));
  await loadModels();
}

async function loadModels(preferredModel?: string): Promise<void> {
  const request = ++modelsRequest;
  const provider = state.providers[state.providerIndex];
  state.models = [];
  state.modelIndex = 0;
  if (!provider) return render();
  state.modelsLoading = true;
  render();
  try {
    const data = await api.get<{ models: ProviderModel[] }>(url(`/api/providers/${provider}/models`));
    if (request !== modelsRequest) return;
    state.models = data.models;
    const preferred = preferredModel ?? process.env.MYAI_MODEL ?? '';
    const index = data.models.findIndex((model) => model.id === preferred);
    state.modelIndex = index >= 0 ? index : 0;
  } catch (error) {
    if (request === modelsRequest) state.error = describe(error);
  } finally {
    if (request === modelsRequest) {
      state.modelsLoading = false;
      render();
    }
  }
}

async function resolveWorkspace(): Promise<void> {
  try {
    const result = await api.post<ResolveWorkspaceResponse>(url('/api/workspaces/resolve'), { path: process.cwd() });
    state.workspace = result.workspace;
    state.status = result.created ? `workspace ${result.workspace.name} utworzony` : '';
  } catch {
    const data = await api.get<WorkspaceList>(url('/api/workspaces'));
    const workspace = data.workspaces[0];
    if (!workspace) throw new Error('no workspace available');
    state.workspace = workspace;
    state.status = `workspace: ${workspace.name}`;
  }
}

async function selectAgent(agentIndex: number): Promise<void> {
  state.agentIndex = agentIndex;
  const agent = selectedAgent();
  if (agent?.provider_key) {
    const index = state.providers.indexOf(agent.provider_key);
    if (index >= 0) state.providerIndex = index;
    await loadModels(agent.model_id ?? undefined);
  }
  render();
}

/* ------------------------------------------------------------- transcript */

function applyEvent(event: StreamEvent): void {
  if (event.type === 'message' && event.role === 'assistant' && nextMessageStartsNew) {
    state.items.push({ kind: 'assistant', text: '' });
    nextMessageStartsNew = false;
  }
  if (event.type === 'finalized') nextMessageStartsNew = false;
  if (event.type === 'approval_request') {
    state.blockingId = event.original_message_id;
    state.status = 'approval required';
    return;
  }
  if (event.type === 'finalized') state.lastSequence = Math.max(state.lastSequence, event.sequence_id);
  applyStreamEvent(state.items, event);
}

/* ------------------------------------------------------------------ flows */

function fire(task: Promise<unknown>): void {
  void task.catch((error: unknown) => {
    state.error = describe(error);
    render();
  });
}

function approvalPending(): boolean {
  return state.blockingId !== null;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

async function consumeStream(
  stream: Awaited<ReturnType<typeof openNdjson<StreamEvent>>>,
  token: number,
): Promise<void> {
  for await (const event of stream.events) {
    if (token !== streamToken) return;
    applyEvent(event);
    render();
  }
}

function applyDetails(details: ConversationDetails): void {
  state.conversationId = details.id;
  state.title = details.title;
  state.items = transcriptFromMessages(details.messages);
  state.blockingId = details.blocking_message_id;
  state.lastSequence = details.messages.reduce((max, message) => Math.max(max, message.sequence_id), 0);
}

async function follow(conversationId: number, token: number): Promise<void> {
  const stream = await openNdjson<StreamEvent>(
    url(`/api/conversations/${conversationId}/stream?after_sequence_id=${state.lastSequence}`),
  );
  nextMessageStartsNew = true;
  await consumeStream(stream, token);
  nextMessageStartsNew = false;
}

async function resume(conversationId: number, token: number): Promise<void> {
  if (token !== streamToken) return;
  state.status = 'reconnecting…';
  render();
  const details = await api.get<ConversationDetails>(url(`/api/conversations/${conversationId}`));
  if (token !== streamToken) return;
  applyDetails(details);
  render();
  await follow(conversationId, token);
  if (token !== streamToken) return;
  state.status = approvalPending() ? 'approval required' : 'done';
}

async function ensureConversation(): Promise<number> {
  if (state.conversationId !== null) return state.conversationId;
  if (!state.workspace) throw new Error('no workspace');
  const agent = selectedAgent();
  const body: CreateConversationRequest = {
    workspace_id: state.workspace.id,
    agent_id: agent?.internal_name ?? null,
  };
  const details = await api.post<ConversationDetails>(url('/api/conversations'), body);
  state.conversationId = details.id;
  state.title = details.title;
  return details.id;
}

async function sendPrompt(prompt: string): Promise<void> {
  if (!prompt.trim() || state.busy) return;
  const provider = state.providers[state.providerIndex];
  const model = state.models[state.modelIndex]?.id;
  if (!provider || !model) {
    state.error = 'wybierz provider i model (alt-p / alt-m)';
    render();
    return;
  }
  const token = ++streamToken;
  nextMessageStartsNew = false;
  const optimistic: TranscriptItem = { kind: 'user', text: prompt };
  state.busy = true;
  state.error = '';
  state.blockingId = null;
  state.buffer = '';
  state.scroll = 0;
  state.items.push(optimistic);
  state.status = 'sending…';
  render();

  try {
    const conversationId = await ensureConversation();
    const stream = await openNdjson<StreamEvent>(url(`/api/conversations/${conversationId}/prompt`), {
      method: 'POST',
      body: { prompt, provider_key: provider, model_id: model, inference_config: {} },
    });
    if (token !== streamToken) return;
    state.status = 'generating…';
    await consumeStream(stream, token);
    if (token !== streamToken) return;
    state.status = approvalPending() ? 'approval required' : 'done';
  } catch (error) {
    if (token !== streamToken) return;
    const rejected = error instanceof ApiError && error.status >= 400 && error.status < 500;
    if (rejected) {
      if (state.items[state.items.length - 1] === optimistic) state.items.pop();
      state.buffer = prompt;
    }
    if (error instanceof ApiError && error.status === 403) {
      state.blockingId = error.blockingMessageId;
      state.status = 'approval required';
    } else if (rejected || state.conversationId === null) {
      state.error = describe(error);
      state.status = 'error';
    } else {
      state.error = describe(error);
      try {
        await resume(state.conversationId, token);
      } catch (resumeError) {
        state.error = describe(resumeError);
        state.status = 'error';
      }
    }
  } finally {
    if (token === streamToken) {
      state.busy = false;
      render();
    }
  }
}

async function decideAndContinue(decision: 'approve' | 'reject'): Promise<void> {
  if (state.conversationId === null || state.blockingId === null || state.busy) return;
  const conversationId = state.conversationId;
  const messageId = state.blockingId;
  const token = ++streamToken;
  state.busy = true;
  state.status = `${decision}…`;
  render();

  try {
    const response = await api.post<ToolCallDecisionResponse>(
      url(`/api/conversations/${conversationId}/tool_calls/${messageId}/decide`),
      { decision },
    );
    if (token !== streamToken) return;
    state.blockingId = null;
    if (response.result !== null) applyToolResult(state.items, messageId, response.result);
    state.status = 'continuing…';
    render();
  } catch (error) {
    if (token !== streamToken) return;
    state.error = describe(error);
    try {
      await resume(conversationId, token);
    } catch (resumeError) {
      state.error = describe(resumeError);
      state.status = 'error';
    }
    return;
  }

  try {
    const stream = await openNdjson<StreamEvent>(url(`/api/conversations/${conversationId}/continue`), {
      method: 'POST',
    });
    await consumeStream(stream, token);
    if (token !== streamToken) return;
    state.status = approvalPending() ? 'approval required' : 'done';
  } catch (error) {
    if (token !== streamToken) return;
    state.error = describe(error);
    try {
      await resume(conversationId, token);
    } catch (resumeError) {
      state.error = describe(resumeError);
      state.status = 'error';
    }
  } finally {
    if (token === streamToken) {
      state.busy = false;
      render();
    }
  }
}

async function openConversation(id: number): Promise<void> {
  const token = ++streamToken;
  state.error = '';
  state.busy = true;
  state.status = 'loading…';
  render();
  try {
    const details = await api.get<ConversationDetails>(url(`/api/conversations/${id}`));
    if (token !== streamToken) return;
    applyDetails(details);
    if (details.provider_key) {
      const providerIndex = state.providers.indexOf(details.provider_key);
      if (providerIndex >= 0) {
        state.providerIndex = providerIndex;
        await loadModels(details.model_id ?? undefined);
      }
    }
    if (details.agent_id) {
      const agentIndex = state.agents.findIndex((agent) => agent.internal_name === details.agent_id) + 1;
      if (agentIndex > 0) state.agentIndex = agentIndex;
    }
    state.scroll = 0;
    state.view = 'chat';
    if (details.blocking_message_id === null) {
      state.status = 'streaming…';
      render();
      await follow(id, token);
      if (token !== streamToken) return;
      state.status = approvalPending() ? 'approval required' : '';
    } else {
      state.status = 'approval required';
    }
  } catch (error) {
    if (token !== streamToken) return;
    state.error = describe(error);
  } finally {
    if (token === streamToken) {
      state.busy = false;
      render();
    }
  }
}

async function deleteSelected(): Promise<void> {
  const conversation = state.conversations[state.selectedConversation];
  if (!conversation) return;
  try {
    await api.del(url(`/api/conversations/${conversation.id}`));
    await loadConversations();
  } catch (error) {
    state.error = describe(error);
    render();
  }
}

async function cancelGeneration(): Promise<void> {
  if (state.conversationId === null) return;
  streamToken += 1;
  state.busy = false;
  try {
    await api.post(url(`/api/conversations/${state.conversationId}/cancel`));
    state.status = 'cancelled';
  } catch (error) {
    state.status = describe(error);
  }
  render();
}

/* --------------------------------------------------------------- keyboard */

function onKey(key: Key): void {
  if (picker) {
    if (key.name === 'ctrl-c') {
      closePicker();
      return;
    }
    const result = handlePickerKey(picker.view, key);
    if (result.kind === 'cancelled') {
      closePicker();
      return;
    }
    if (result.kind === 'chosen') {
      const choose = picker.choose;
      closePicker();
      choose(result.value);
      return;
    }
    render();
    return;
  }
  if (state.view === 'list') return onListKey(key);
  if (state.view === 'new') return onNewKey(key);
  return onChatKey(key);
}

function onListKey(key: Key): void {
  switch (key.name) {
    case 'ctrl-c':
      shutdown(0);
      return;
    case 'char':
      if (key.char === 'q') shutdown(0);
      if (key.char === 'n') {
        state.view = 'new';
        state.error = '';
        fire(loadProviders());
        fire(loadAgents());
      }
      if (key.char === 'd') fire(deleteSelected());
      if (key.char === 'r') fire(loadConversations());
      return;
    case 'up':
      state.selectedConversation = Math.max(0, state.selectedConversation - 1);
      break;
    case 'down':
      state.selectedConversation = Math.min(state.conversations.length - 1, state.selectedConversation + 1);
      break;
    case 'enter': {
      const conversation = state.conversations[state.selectedConversation];
      if (conversation) fire(openConversation(conversation.id));
      return;
    }
    default:
      return;
  }
  render();
}

function onNewKey(key: Key): void {
  switch (key.name) {
    case 'ctrl-c':
      shutdown(0);
      return;
    case 'esc':
      state.view = 'list';
      state.error = '';
      break;
    case 'enter':
      if (!state.workspace) return;
      if (state.models.length === 0) {
        state.error = 'wybierz model (m)';
        render();
        return;
      }
      streamToken += 1;
      state.conversationId = null;
      state.title = null;
      state.items = [];
      state.buffer = '';
      state.blockingId = null;
      state.view = 'chat';
      state.status = '';
      break;
    case 'char':
      if (key.char === 'p') openProviderPicker();
      if (key.char === 'm') openModelPicker();
      if (key.char === 'a') {
        const next = state.agentIndex + 1 > state.agents.length ? 0 : state.agentIndex + 1;
        fire(selectAgent(next));
      }
      break;
    default:
      return;
  }
  render();
}

function onChatKey(key: Key): void {
  if (key.name === 'ctrl-c') {
    if (state.busy) fire(cancelGeneration());
    else {
      state.view = 'list';
      fire(loadConversations());
    }
    return;
  }
  if (state.blockingId !== null) {
    if (key.name === 'char' && key.char === 'y') fire(decideAndContinue('approve'));
    if (key.name === 'char' && key.char === 'n') fire(decideAndContinue('reject'));
    if (key.name === 'esc') {
      state.view = 'list';
      fire(loadConversations());
    }
    return;
  }
  switch (key.name) {
    case 'esc':
      state.view = 'list';
      fire(loadConversations());
      return;
    case 'enter': {
      const prompt = state.buffer;
      state.buffer = '';
      if (prompt.trim()) fire(sendPrompt(prompt));
      return;
    }
    case 'backspace':
      state.buffer = Array.from(state.buffer).slice(0, -1).join('');
      break;
    case 'up':
      state.scroll += 1;
      break;
    case 'down':
      state.scroll = Math.max(0, state.scroll - 1);
      break;
    case 'char':
      state.buffer += key.char;
      break;
    case 'alt':
      if (key.char === 'm') openModelPicker();
      if (key.char === 'p') openProviderPicker();
      break;
    default:
      return;
  }
  render();
}

/* ------------------------------------------------------------------- main */

function shutdown(code = 0): void {
  input?.stop();
  screen.leave();
  process.exit(code);
}

async function main(): Promise<void> {
  BASE = await resolveServer();
  await resolveWorkspace();

  if (dumpMode) {
    const idArgument = process.argv.find((argument) => /^\d+$/.test(argument));
    await loadConversations();
    if (idArgument) await openConversation(Number(idArgument));
    process.stdout.write(`${currentFrame().join('\n')}\n`);
    return;
  }

  screen.enter();
  process.on('exit', () => screen.leave());
  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));
  process.stdout.on('resize', () => render());

  input = startInput(onKey);
  try {
    await Promise.all([loadConversations(), loadProviders(), loadAgents()]);
  } catch (error) {
    state.error = describe(error);
  }
  render();
  await new Promise<never>(() => {});
}

void main().catch((error: unknown) => {
  screen.leave();
  process.stderr.write(`${describe(error)}\n`);
  process.exit(1);
});
