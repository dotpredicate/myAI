import { el, clear, labeledField, readString, writeValue, conversationLabel, messageOf, type MountOptions, type Route } from '../ui.js';
import { ApiError, api, openNdjson } from '../core/api.js';
import type {
  AgentConfig,
  AgentList,
  ConversationDetails,
  CreateConversationRequest,
  NdjsonStream,
  ProviderInfo,
  ProviderList,
  ProviderModel,
  ProviderModelList,
  StreamEvent,
  ToolCallDecisionResponse,
  WorkspaceConfig,
  WorkspaceList,
} from '../core/api.js';
import { Timeline, type ToolBlock } from './timeline.js';
import { Row, Grid, Heading, InlineHint, Button, TextArea, Select, Option, Spacer } from '../components.js';

export function mount(route: Route, _options: MountOptions = {}) {
  const root = el('section', { class: 'chat-screen' });
  const title = Heading(1, 'Conversation');
  const top = el('header', { class: 'chat-top' }, title);
  const history = el('div', { class: 'chat-history' });

  const workspaceSelect = Select({});
  const workspaceField = labeledField('Workspace', workspaceSelect, { hint: 'Sources and writes come from the workspace.' });
  const agentSelect = Select({});
  const agentField = labeledField('Agent', agentSelect, { hint: 'Optional instructions.' });
  const providerSelect = Select({});
  const providerField = labeledField('Provider', providerSelect);
  const modelSelect = Select({});
  const modelField = labeledField('Model', modelSelect);
  const selectRow = Grid({ columns: 2 }, workspaceField.wrapper, agentField.wrapper, providerField.wrapper, modelField.wrapper);

  const textarea = TextArea({ rows: 3, placeholder: 'Ask anything…' });
  const sendButton = Button('Send', { variant: 'primary' });
  const stopButton = Button('Stop', { variant: 'quiet' });
  const liveStatus = InlineHint();
  const alert = el('div', { class: 'notice notice--error row', hidden: true });
  const composer = el('div', { class: 'composer' },
    alert,
    selectRow,
    textarea,
    Row(liveStatus, Spacer(), stopButton, sendButton),
  );

  root.appendChild(top);
  root.appendChild(history);
  root.appendChild(composer);

  let destroyed = false;
  let conversationId: number | null = route.id || null;
  let details: ConversationDetails | null = null;
  let workspaces: WorkspaceConfig[] = [];
  let agents: AgentConfig[] = [];
  let providers: ProviderInfo[] = [];
  let models: ProviderModel[] = [];
  let workspaceId: number | null = null;
  let agentKey = readString('chat.agent', '');
  let providerKey = readString('chat.provider', '');
  let modelId = readString('chat.model', '');
  let generating = false;
  let streamToken = 0;
  let controller: AbortController | null = null;

  const timeline = new Timeline(history, {
    onBlockingChange: (id) => {
      updateComposerState();
      if (id) announce('Assistant is waiting for approval.');
    },
    onGenerationError: (message) => {
      showError(message);
    },
    onApprove: (id, _block: ToolBlock) => decide(id, 'approve', ''),
    onReject: (id, comment: string) => decide(id, 'reject', comment),
  });

  function announce(message: string): void {
    liveStatus.textContent = message;
  }

  function showError(message: string): void {
    alert.textContent = message;
    alert.hidden = !message;
  }

  function updateTitle(): void {
    title.textContent = conversationId
      ? (details ? conversationLabel(details) : `Conversation #${conversationId}`)
      : 'New conversation';
  }

  function selectedAgent(): AgentConfig | null {
    return agents.find((agent) => agent.internal_name === agentKey) ?? null;
  }

  function updateComposerState(): void {
    const blocking = timeline.blockingId !== null;
    const hasWorkspace = workspaceId !== null;
    const canSend = hasWorkspace && !generating && !blocking && Boolean(providerKey && modelId) && textarea.value.trim().length > 0;
    sendButton.disabled = !canSend;
    stopButton.disabled = !generating;
    textarea.disabled = generating || blocking;
    // A conversation is bound to its workspace when it is created.
    workspaceSelect.disabled = generating || blocking || conversationId !== null;
    agentSelect.disabled = generating || blocking;
    providerSelect.disabled = generating || blocking;
    modelSelect.disabled = generating || blocking;
  }

  function fillWorkspaces(): void {
    clear(workspaceSelect);
    workspaceSelect.appendChild(Option('', 'Select a workspace'));
    for (const workspace of workspaces) {
      workspaceSelect.appendChild(Option(String(workspace.id), workspace.name));
    }
    const stored = readString('chat.workspace', '');
    const preferred = workspaces.find((workspace) => String(workspace.id) === stored) ?? workspaces[0];
    workspaceId = preferred?.id ?? null;
    workspaceSelect.value = preferred ? String(preferred.id) : '';
  }

  function fillAgents(): void {
    clear(agentSelect);
    agentSelect.appendChild(Option('', 'No agent'));
    for (const agent of agents) {
      agentSelect.appendChild(Option(agent.internal_name, agent.display_name || agent.internal_name));
    }
    if (agentKey && !agents.some((agent) => agent.internal_name === agentKey)) agentKey = '';
    agentSelect.value = agentKey;
  }

  function fillProviders(): void {
    clear(providerSelect);
    providerSelect.appendChild(Option('', 'Select a provider'));
    for (const provider of providers) {
      providerSelect.appendChild(Option(provider.key, provider.display_name || provider.key));
    }
    if (providerKey && !providers.some((provider) => provider.key === providerKey)) providerKey = '';
    providerSelect.value = providerKey;
  }

  function fillModels(): void {
    clear(modelSelect);
    modelSelect.appendChild(Option('', 'Select a model'));
    for (const model of models) modelSelect.appendChild(Option(model.id, model.name || model.id));
    if (modelId && !models.some((model) => model.id === modelId)) modelId = '';
    modelSelect.value = modelId;
  }

  async function loadModels(preferred?: string): Promise<void> {
    models = [];
    modelId = '';
    fillModels();
    if (!providerKey) return;
    try {
      const data = await api.get<ProviderModelList>(`/api/providers/${encodeURIComponent(providerKey)}/models`);
      if (destroyed) return;
      models = Array.isArray(data.models) ? data.models : [];
      modelId = preferred && models.some((model) => model.id === preferred)
        ? preferred
        : (models[0]?.id ?? '');
      fillModels();
      updateComposerState();
    } catch (error) {
      showError(messageOf(error));
    }
  }

  async function loadWorkspaces(): Promise<void> {
    try {
      const data = await api.get<WorkspaceList>('/api/workspaces');
      if (destroyed) return;
      workspaces = Array.isArray(data.workspaces) ? data.workspaces : [];
      fillWorkspaces();
      updateComposerState();
    } catch (error) {
      showError(messageOf(error));
    }
  }

  async function loadAgents(): Promise<void> {
    try {
      const data = await api.get<AgentList>('/api/agents');
      if (destroyed) return;
      agents = Array.isArray(data.agents) ? data.agents : [];
      fillAgents();
      updateComposerState();
    } catch (error) {
      showError(messageOf(error));
    }
  }

  async function loadProviders(): Promise<void> {
    try {
      const data = await api.get<ProviderList>('/api/providers');
      if (destroyed) return;
      providers = Array.isArray(data.providers) ? data.providers : [];
      if (!providerKey) providerKey = providers[0]?.key ?? '';
      fillProviders();
      await loadModels(readString('chat.model', ''));
    } catch (error) {
      showError(messageOf(error));
    }
  }

  function setConversationInUrl(id: number): void {
    conversationId = id;
    const hash = `#/chat/${id}`;
    window.history.replaceState(null, '', hash);
    writeValue('lastHash', hash);
  }

  async function attach(stream: NdjsonStream<StreamEvent>, token: number): Promise<void> {
    for await (const event of stream.events) {
      if (destroyed || token !== streamToken) return;
      timeline.handleEvent(event);
      updateComposerState();
    }
  }

  function finish(token: number): void {
    if (token !== streamToken) return;
    timeline.endStream();
    generating = false;
    controller = null;
    updateComposerState();
    announce(timeline.blockingId !== null ? 'Waiting for approval.' : 'Response complete.');
  }

  async function sendPrompt(): Promise<void> {
    const prompt = textarea.value.trim();
    if (!prompt || generating || workspaceId === null) return;
    if (!providerKey || !modelId) {
      showError('Choose a provider and model.');
      return;
    }
    showError('');
    const token = ++streamToken;
    generating = true;
    updateComposerState();
    announce('Sending…');
    timeline.addUserMessage(prompt);
    textarea.value = '';

    try {
      if (conversationId === null) {
        const body: CreateConversationRequest = {
          workspace_id: workspaceId,
          agent_id: agentKey || null,
        };
        const created = await api.post<ConversationDetails>('/api/conversations', body);
        if (destroyed) return;
        conversationId = created.id;
        details = created;
        setConversationInUrl(created.id);
        updateTitle();
      }
      controller = new AbortController();
      const stream = await openNdjson<StreamEvent>(`/api/conversations/${conversationId}/prompt`, {
        method: 'POST',
        body: { prompt, provider_key: providerKey, model_id: modelId, inference_config: {} },
        signal: controller.signal,
      });
      announce('Generating…');
      await attach(stream, token);
      finish(token);
    } catch (error) {
      if (destroyed || token !== streamToken) return;
      generating = false;
      updateComposerState();
      if (error instanceof ApiError && error.status === 403) {
        announce('Waiting for approval.');
        showError('');
      } else {
        showError(messageOf(error));
      }
    }
  }

  async function decide(id: number, decision: 'approve' | 'reject', comment: string): Promise<void> {
    if (conversationId === null || generating) return;
    const token = ++streamToken;
    generating = true;
    updateComposerState();
    try {
      const response = await api.post<ToolCallDecisionResponse>(
        `/api/conversations/${conversationId}/tool_calls/${id}/decide`,
        { decision, comment },
      );
      timeline.applyDecision(id, decision, comment, response.result);
      controller = new AbortController();
      const stream = await openNdjson<StreamEvent>(`/api/conversations/${conversationId}/continue`, {
        method: 'POST',
        signal: controller.signal,
      });
      await attach(stream, token);
      finish(token);
    } catch (error) {
      if (destroyed || token !== streamToken) return;
      generating = false;
      updateComposerState();
      showError(messageOf(error));
    }
  }

  async function loadConversation(): Promise<void> {
    if (!conversationId) return;
    clear(history);
    timeline.reset();
    try {
      const loaded = await api.get<ConversationDetails>(`/api/conversations/${conversationId}`);
      if (destroyed) return;
      details = loaded;
      workspaceId = loaded.workspace_id ?? workspaceId;
      workspaceSelect.value = workspaceId === null ? '' : String(workspaceId);
      if (loaded.agent_id) {
        agentKey = loaded.agent_id;
        agentSelect.value = agentKey;
      }
      if (loaded.provider_key) {
        providerKey = loaded.provider_key;
        providerSelect.value = providerKey;
        await loadModels(loaded.model_id ?? undefined);
      }
      for (const record of loaded.messages) timeline.addStored(record);
      timeline.setBlocking(loaded.blocking_message_id);
      updateTitle();
      updateComposerState();
      if (loaded.blocking_message_id === null) {
        const stream = await openNdjson<StreamEvent>(
          `/api/conversations/${conversationId}/stream?after_sequence_id=${timeline.lastSequence}`,
        );
        await attach(stream, streamToken);
      }
    } catch (error) {
      showError(messageOf(error));
    }
  }

  workspaceSelect.addEventListener('change', () => {
    const selected = Number(workspaceSelect.value);
    workspaceId = Number.isFinite(selected) && selected > 0 ? selected : null;
    writeValue('chat.workspace', workspaceId === null ? '' : String(workspaceId));
    updateComposerState();
  });

  agentSelect.addEventListener('change', () => {
    agentKey = agentSelect.value;
    writeValue('chat.agent', agentKey);
    const agent = selectedAgent();
    if (agent?.provider_key && agent.model_id) {
      providerKey = agent.provider_key;
      providerSelect.value = providerKey;
      writeValue('chat.provider', providerKey);
      void loadModels(agent.model_id);
    }
    updateComposerState();
  });

  providerSelect.addEventListener('change', () => {
    providerKey = providerSelect.value;
    writeValue('chat.provider', providerKey);
    void loadModels();
  });

  modelSelect.addEventListener('change', () => {
    modelId = modelSelect.value;
    writeValue('chat.model', modelId);
    updateComposerState();
  });

  textarea.addEventListener('input', updateComposerState);
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void sendPrompt();
    }
  });
  sendButton.addEventListener('click', () => { void sendPrompt(); });
  stopButton.addEventListener('click', () => {
    if (conversationId !== null) void api.post(`/api/conversations/${conversationId}/cancel`, {}).catch(() => undefined);
    controller?.abort();
    streamToken += 1;
    generating = false;
    updateComposerState();
  });

  if (conversationId !== null) {
    void loadConversation();
  } else {
    timeline.reset();
  }
  void loadWorkspaces();
  void loadAgents();
  void loadProviders();
  updateComposerState();

  return {
    element: root,
    destroy(): void {
      destroyed = true;
      controller?.abort();
    },
    focus(): void {
      textarea.focus();
    },
    newChat(): void {
      conversationId = null;
      details = null;
      timeline.reset();
      updateTitle();
      updateComposerState();
      textarea.focus();
    },
  };
}
