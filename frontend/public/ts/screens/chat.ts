import { el, clear, errorState, spinnerLabel, labeledField, notify, readString, readValue, writeValue, conversationLabel, allowedPolicies, policyLabel, pickDefined, renderParamFields, statusMessage as setStatus, asError, asRecord, asParamValueMap, asPolicy, messageOf, type MountOptions, type ParamValue, type Route } from '../ui.js';
import { ApiError, isAbort, api, openNdjson } from '../api.js';
import type { AgentConfig, AgentList, ConversationDetails, GenerationRequest, InferenceParam, InferenceParamList, NdjsonStream, PromptRequest, ProviderInfo, ProviderList, ProviderModel, ProviderModelList, RepositoryConfig, RepositoryList, SecurityPolicy, StreamEvent, ToolCallDecisionRequest, ToolCallDecisionResponse } from '../api.js';
import { Timeline, type ToolBlock } from './timeline.js';
import { Row, RowEnd, Stack, Grid, Heading, Paragraph, Hint, InlineHint, FieldError, Notice, Button, TextArea, Select, Option, Spacer, Summary } from '../components.js';

function configKey(provider: string, model: string): string {
  return `${provider}::${model}`;
}

function asStringMap(value: Record<string, unknown>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(asRecord(value))) result[key] = String(item);
  return result;
}

interface ChatScope {
  internal_name: string;
  security_policy_override: SecurityPolicy | null;
}

interface ChatState {
  providers: ProviderInfo[];
  paramsCache: Record<string, InferenceParam[]>;
  agents: AgentConfig[];
  repositories: RepositoryConfig[];
  scopes: ChatScope[];
  selectedAgentKey: string;
  currentProvider: string;
  currentModel: string;
  currentParams: InferenceParam[];
  loadingModels: boolean;
  agentModelMissing: boolean;
  generating: boolean;
  blockingId: number | null;
  generationError: boolean;
  manualProvider: string;
  manualModel: string;
  modelsStored: Record<string, string>;
  configs: Record<string, Record<string, ParamValue>>;
  agentsReady: boolean;
  streamToken: number;
  modelToken: number;
  streamController: AbortController | null;
  lastAnnouncedBlocking: number | null;
  stopRequested: boolean;
}

export function mount(route: Route, options: MountOptions = {}) {
  const root = el('section', { class: 'chat-screen' });
  const title = Heading(1, 'New conversation');
  const top = el('header', { class: 'chat-top' }, title);
  const history = el('div', { class: 'chat-history' });
  const jump = Button('Jump to latest', { small: true, class: 'chat-jump', hidden: true });

  const approvalAlert = el('div', { class: 'notice notice--warning row', hidden: true });
  const generationAlert = el('div', { class: 'notice notice--error row', hidden: true });
  const composerAlerts = Stack(approvalAlert, generationAlert);

  const emptyView = Notice({ level: 'state', stack: true }, Heading(2, 'No messages yet'), Paragraph('Type a message below to start.'));

  const scopeRepoSelect = Select({});
  const scopePolicySelect = Select({});
  const scopeAdd = Button('Add');
  const scopePills = Row();
  const scopeStatus = Hint();
  const scopeRepoField = labeledField('Scope repository', scopeRepoSelect);
  const scopePolicyField = labeledField('Permissions', scopePolicySelect);
  const scopeBar = RowEnd(scopeRepoField.wrapper, scopePolicyField.wrapper, scopeAdd);
  const scopeSection = Stack(scopeBar, scopePills, scopeStatus);

  const textarea = TextArea({
    rows: 3,
    placeholder: 'Ask anything…',
  });

  const agentSelect = Select({});
  const providerSelect = Select({});
  const modelSelect = Select({});
  const agentField = labeledField('Agent', agentSelect);
  const providerField = labeledField('Provider', providerSelect);
  const modelField = labeledField('Model', modelSelect);
  const agentStatus = Hint();
  agentField.wrapper.appendChild(agentStatus);
  const selectRow = Grid({ columns: 3 }, agentField.wrapper, providerField.wrapper, modelField.wrapper);

  const settingsDetails = el('details', { hidden: true });
  const settingsSummary = Summary('Model settings');
  const settingsBody = Stack();
  const paramGrid = Grid({});
  const resetSettings = Button('Reset to defaults', { variant: 'quiet', small: true });
  settingsBody.appendChild(paramGrid);
  settingsBody.appendChild(Row(resetSettings));
  settingsDetails.appendChild(settingsSummary);
  settingsDetails.appendChild(settingsBody);
  const modelStatus = Hint();

  const liveStatus = InlineHint();
  const hint = InlineHint('Enter for a new line · Ctrl/Cmd+Enter to send');
  const sendButton = Button('Send', { variant: 'primary' });
  const continueButton = Button('Continue');
  const stopButton = Button('Stop', { variant: 'quiet' });

  const composer = el('div', { class: 'composer' },
    composerAlerts,
    scopeSection,
    textarea,
    selectRow,
    modelStatus,
    settingsDetails,
    Row(hint, liveStatus, Spacer(), stopButton, continueButton, sendButton),
  );

  root.appendChild(top);
  root.appendChild(history);
  root.appendChild(jump);
  root.appendChild(composer);
  history.appendChild(emptyView);

  let destroyed = false;
  let conversationId: number | null = route.id || null;
  let conversationDetails: ConversationDetails | null = null;
  let conversationAvailable = true;

  const timeline = new Timeline(history, {
    onBlockingChange: (id) => {
      state.blockingId = id;
      updateBlockingUI();
      if (id) announce('Assistant is waiting for approval.');
    },
    onGenerationError: (message) => {
      showGenerationError(message);
      notify(message, { level: 'error' });
    },
    onApprove: (id, block) => decide(id, 'approve', '', block),
    onReject: (id, comment, block) => decide(id, 'reject', comment, block),
  });

  const state: ChatState = {
    providers: [],
    paramsCache: {},
    agents: [],
    repositories: [],
    scopes: [],
    selectedAgentKey: '',
    currentProvider: '',
    currentModel: '',
    currentParams: [],
    loadingModels: false,
    agentModelMissing: false,
    generating: false,
    blockingId: null,
    generationError: false,
    manualProvider: readString('chat.provider', ''),
    manualModel: '',
    modelsStored: asStringMap(readValue<Record<string, unknown>>('chat.models', {})),
    configs: asParamValueMap(readValue<Record<string, unknown>>('chat.configs', {})),
    agentsReady: false,
    streamToken: 0,
    modelToken: 0,
    streamController: null,
    lastAnnouncedBlocking: null,
    stopRequested: false,
  };


  let lastAnnouncement = '';

  function announce(message: string): void {
    if (message === lastAnnouncement) return;
    lastAnnouncement = message;
    liveStatus.textContent = message;
  }

  function setEmptyVisible(visible: boolean): void {
    emptyView.hidden = !visible;
  }

  function isNearBottom(): boolean {
    return history.scrollHeight - history.scrollTop - history.clientHeight < 90;
  }

  function updateJump(): void {
    jump.hidden = isNearBottom() || history.scrollHeight <= history.clientHeight + 5;
  }

  function scrollToBottom(force = false): void {
    if (force || isNearBottom()) {
      history.scrollTop = history.scrollHeight;
      updateJump();
    }
  }

  function autoScroll(): void {
    scrollToBottom(false);
    updateJump();
  }

  history.addEventListener('scroll', updateJump);
  jump.addEventListener('click', () => scrollToBottom(true));

  function showGenerationError(message: string): void {
    state.generationError = true;
    clear(generationAlert);
    generationAlert.hidden = false;
    generationAlert.appendChild(Paragraph(message));
    const close = Button('Dismiss', { small: true });
    close.addEventListener('click', () => {
      generationAlert.hidden = true;
      state.generationError = false;
    });
    generationAlert.appendChild(close);
  }

  function clearGenerationError(): void {
    generationAlert.hidden = true;
    state.generationError = false;
  }

  function updateBlockingUI(): void {
    const blocking = state.blockingId !== null;
    approvalAlert.hidden = !blocking;
    if (blocking) {
      clear(approvalAlert);
      approvalAlert.appendChild(Paragraph('Approval required before the conversation can continue.'));
      approvalAlert.appendChild(Button('Go to approval', { variant: 'primary', small: true, onClick: focusApproval }));
    }
    if (blocking && state.lastAnnouncedBlocking !== state.blockingId) {
      state.lastAnnouncedBlocking = state.blockingId;
      announce('Approval required. The message box is paused.');
    }
    if (!blocking) state.lastAnnouncedBlocking = null;
    updateComposerState();
  }

  function focusApproval(): void {
    const block = timeline.getBlock(state.blockingId);
    if (!block) {
      notify('Waiting action is not visible in the history.', { level: 'error' });
      return;
    }
    block.node.open = true;
    block.node.dataset.highlight = 'true';
    block.node.scrollIntoView({ block: 'center', behavior: 'smooth' });
    window.setTimeout(() => {
      delete block.node.dataset.highlight;
    }, 1600);
    const target = block.focusAction() || block.node.querySelector<HTMLElement>('summary');
    target?.focus();
  }

  function hasModel(): boolean {
    return Boolean(state.currentProvider && state.currentModel);
  }

  function updateComposerState(): void {
    const blocking = state.blockingId !== null;
    const busy = state.generating;
    const modelReady = hasModel() && !state.agentModelMissing && !state.loadingModels;
    const canWrite = textarea.value.trim().length > 0;
    sendButton.disabled = !conversationAvailable || busy || blocking || !modelReady || !canWrite;
    continueButton.disabled = !conversationAvailable || busy || blocking || !modelReady || !conversationId;
    stopButton.hidden = !busy;
    stopButton.disabled = !busy || state.stopRequested;
    textarea.readOnly = blocking;
    if (blocking) announce('Approval required. The message box is paused.');
    else if (busy) announce('Assistant is responding…');
  }

  function saveConfigs(): void {
    writeValue('chat.configs', state.configs);
  }

  function currentConfigValues(): Record<string, ParamValue> {
    if (!state.currentProvider || !state.currentModel) return {};
    return { ...state.configs[configKey(state.currentProvider, state.currentModel)] };
  }

  function setConfigValues(values: Record<string, ParamValue>): void {
    const key = configKey(state.currentProvider, state.currentModel);
    if (Object.keys(values).length === 0) delete state.configs[key];
    else state.configs[key] = values;
    saveConfigs();
  }

  function renderSettings(): void {
    const show = !state.selectedAgentKey && state.currentParams.length > 0 && hasModel();
    settingsDetails.hidden = !show;
    if (!show) {
      paramGrid.replaceChildren();
      return;
    }
    const values = { ...currentConfigValues() };
    renderParamFields(paramGrid, state.currentParams, values, (name, value) => {
      if (value === undefined || value === '') delete values[name];
      else values[name] = value;
      setConfigValues(values);
    }, { idPrefix: 'chat-setting' });
  }

  resetSettings.addEventListener('click', () => {
    setConfigValues({});
    renderSettings();
  });

  function fillProviderSelect(): void {
    clear(providerSelect);
    for (const provider of state.providers) {
      providerSelect.appendChild(Option(provider.key, provider.display_name));
    }
  }

  function ensureProviderOption(key: string): void {
    if (!key) return;
    if ([...providerSelect.options].some((option) => option.value === key)) return;
    providerSelect.appendChild(Option(key, key));
  }

  function persistManualSelection(): void {
    writeValue('chat.provider', state.manualProvider);
    state.modelsStored[state.manualProvider] = state.manualModel;
    writeValue('chat.models', state.modelsStored);
  }

  function selectedAgent(): AgentConfig | null {
    return state.agents.find((agent) => agent.internal_name === state.selectedAgentKey) || null;
  }

  function renderAgentState(): void {
    if (!state.agentsReady) return;
    const agent = selectedAgent();
    clear(agentStatus);
    if (!agent) {
      state.agentModelMissing = false;
      return;
    }
    if (state.agentModelMissing) {
      agentStatus.appendChild(FieldError(`Model “${agent.model_id}” is not available for provider “${agent.provider_key}”. Edit the agent or choose another.`));
    }
  }

  function fillAgentSelect(): void {
    const previous = state.selectedAgentKey;
    clear(agentSelect);
    agentSelect.appendChild(Option('', 'None (manual)'));
    for (const agent of state.agents) {
      agentSelect.appendChild(Option(agent.internal_name, `${agent.display_name} (${agent.internal_name})`));
    }
    if (previous && state.agents.some((agent) => agent.internal_name === previous)) {
      agentSelect.value = previous;
    } else if (previous) {
      state.selectedAgentKey = '';
      agentSelect.value = '';
    }
  }

  function applyModelsToSelect(models: ProviderModel[], desired: string): string {
    clear(modelSelect);
    for (const model of models) {
      modelSelect.appendChild(Option(model.id, model.name));
    }
    if (models.length === 0) {
      modelSelect.appendChild(Option('', 'No models available'));
      return '';
    }
    const found = models.find((model) => model.id === desired);
    const chosen = found ? found.id : (models[0]?.id ?? '');
    modelSelect.value = chosen;
    return chosen;
  }

  async function loadModelsFor(providerKey: string, opts: { agent?: boolean; preferModel?: string } = {}): Promise<void> {
    const token = ++state.modelToken;
    state.loadingModels = true;
    state.agentModelMissing = false;
    if (!providerKey) {
      clear(modelSelect);
      modelSelect.appendChild(Option('', 'Select a provider'));
      state.currentModel = '';
      state.loadingModels = false;
      updateComposerState();
      return;
    }
    state.currentProvider = providerKey;
    ensureProviderOption(providerKey);
    providerSelect.value = providerKey;
    setStatus(modelStatus, 'Loading models…', { loading: true });
    modelSelect.disabled = true;
    modelSelect.replaceChildren(Option('', 'Loading models…'));

    try {
      const data = await api.get<ProviderModelList>(`/api/providers/${encodeURIComponent(providerKey)}/models`);
      if (destroyed || token !== state.modelToken) return;
      const models = Array.isArray(data.models) ? data.models : [];
      const agent = opts.agent ? selectedAgent() : null;
      const desired = agent ? agent.model_id : (opts.preferModel ?? state.modelsStored[providerKey] ?? '');
      const chosen = applyModelsToSelect(models, desired);
      state.currentModel = chosen;
      if (agent) {
        state.agentModelMissing = !models.some((model) => model.id === agent.model_id);
        if (state.agentModelMissing) {
          clear(modelSelect);
          modelSelect.appendChild(Option('', `(unavailable) ${agent.model_id}`));
          modelSelect.value = '';
          state.currentModel = '';
        }
      } else {
        state.manualModel = chosen;
        state.manualProvider = providerKey;
        state.modelsStored[providerKey] = chosen;
        persistManualSelection();
      }
      modelSelect.disabled = Boolean(agent) || models.length === 0;
      setStatus(modelStatus, models.length ? '' : 'No models are available for this provider.');
      if (!agent) {
        await loadParamsFor(providerKey, token);
      } else {
        state.currentParams = [];
        renderSettings();
      }
    } catch (error) {
      if (destroyed || token !== state.modelToken) return;
      state.currentModel = '';
      state.agentModelMissing = Boolean(opts.agent);
      clear(modelSelect);
      modelSelect.appendChild(Option('', 'Models unavailable'));
      modelSelect.disabled = true;
      setStatus(modelStatus, messageOf(error) || 'Could not load models.', {
        error: true,
        retry: () => refreshSelection(),
      });
    } finally {
      if (!destroyed && token === state.modelToken) {
        state.loadingModels = false;
        renderAgentState();
        updateComposerState();
        autoScroll();
      }
    }
  }

  async function loadParamsFor(providerKey: string, token = state.modelToken): Promise<void> {
    if (!providerKey) {
      state.currentParams = [];
      renderSettings();
      return;
    }
    try {
      let params = state.paramsCache[providerKey];
      if (!params) {
        params = await api.get<InferenceParamList>(`/api/providers/${encodeURIComponent(providerKey)}/inference-params`);
        state.paramsCache[providerKey] = Array.isArray(params) ? params : [];
      }
      if (destroyed || token !== state.modelToken || state.currentProvider !== providerKey) return;
      state.currentParams = Array.isArray(params) ? params : [];
      setStatus(modelStatus, '');
      renderSettings();
    } catch (error) {
      if (destroyed || token !== state.modelToken || state.currentProvider !== providerKey) return;
      state.currentParams = [];
      renderSettings();
      setStatus(modelStatus, `Could not load model settings: ${messageOf(error)}`, {
        error: true,
        retry: () => loadParamsFor(providerKey, state.modelToken),
      });
    }
  }

  async function refreshSelection(): Promise<void> {
    const agent = selectedAgent();
    if (agent) {
      await loadModelsFor(agent.provider_key, { agent: true });
    } else {
      const provider = state.manualProvider || state.providers[0]?.key || '';
      if (!provider) return;
      state.manualProvider = provider;
      await loadModelsFor(provider, { preferModel: state.manualModel || state.modelsStored[provider] });
    }
  }

  async function loadProviders(): Promise<void> {
    setStatus(modelStatus, 'Loading providers…', { loading: true });
    providerSelect.disabled = true;
    try {
      const data = await api.get<ProviderList>('/api/providers');
      if (destroyed) return;
      state.providers = Array.isArray(data.providers) ? data.providers : [];
      fillProviderSelect();
      if (state.providers.length === 0) {
        setStatus(modelStatus, 'No inference providers are registered.');
        providerSelect.disabled = true;
        return;
      }
      providerSelect.disabled = false;
      if (!state.manualProvider || !state.providers.some((provider) => provider.key === state.manualProvider)) {
        state.manualProvider = state.providers[0]?.key ?? '';
      }
      const agent = selectedAgent();
      await loadModelsFor(agent ? agent.provider_key : state.manualProvider, agent ? { agent: true } : { preferModel: state.modelsStored[state.manualProvider] });
    } catch (error) {
      if (destroyed) return;
      state.providers = [];
      providerSelect.disabled = true;
      setStatus(modelStatus, `Could not load providers: ${messageOf(error)}`, {
        error: true,
        retry: loadProviders,
      });
    }
  }

  async function loadAgents(): Promise<void> {
    try {
      const data = await api.get<AgentList>('/api/agents');
      if (destroyed) return;
      state.agents = Array.isArray(data.agents) ? data.agents : [];
      state.agentsReady = true;
      fillAgentSelect();
      clear(agentStatus);
      agentSelect.disabled = false;
      if (state.selectedAgentKey) {
        const agent = selectedAgent();
        if (agent) {
          await selectAgent(state.selectedAgentKey, { restoreManual: false });
        } else {
          state.selectedAgentKey = '';
          agentSelect.value = '';
          await refreshSelection();
        }
      }
    } catch (error) {
      if (destroyed) return;
      state.agents = [];
      state.agentsReady = false;
      agentSelect.disabled = true;
      clear(agentStatus);
      setStatus(agentStatus, `Could not load agents: ${messageOf(error)}`, { error: true, retry: loadAgents });
    }
  }

  async function loadRepositories(): Promise<void> {
    setStatus(scopeStatus, 'Loading repositories…', { loading: true });
    scopeRepoSelect.disabled = true;
    scopeAdd.disabled = true;
    try {
      const data = await api.get<RepositoryList>('/api/repositories');
      if (destroyed) return;
      state.repositories = Array.isArray(data.repositories) ? data.repositories : [];
      clear(scopeRepoSelect);
      if (state.repositories.length === 0) {
        scopeRepoSelect.appendChild(Option('', 'No repositories'));
        scopeRepoSelect.disabled = true;
        scopeAdd.disabled = true;
        setStatus(scopeStatus, 'Add a repository first to give the assistant access.');
        return;
      }
      for (const repo of state.repositories) {
        scopeRepoSelect.appendChild(Option(repo.internal_name, repo.display_name || repo.internal_name));
      }
      scopeRepoSelect.disabled = false;
      scopeAdd.disabled = false;
      setStatus(scopeStatus, '');
      updateScopePolicies();
    } catch (error) {
      if (destroyed) return;
      state.repositories = [];
      clear(scopeRepoSelect);
      scopeRepoSelect.appendChild(Option('', 'Repositories unavailable'));
      scopeRepoSelect.disabled = true;
      scopeAdd.disabled = true;
      setStatus(scopeStatus, `Could not load repositories: ${messageOf(error)}`, { error: true, retry: loadRepositories });
    }
  }

  function updateScopePolicies(): void {
    const repo = state.repositories.find((item) => item.internal_name === scopeRepoSelect.value);
    clear(scopePolicySelect);
    if (!repo) {
      scopePolicySelect.appendChild(Option('', 'Default'));
      scopePolicySelect.disabled = true;
      return;
    }
    scopePolicySelect.disabled = false;
    scopePolicySelect.appendChild(Option('', `Repository default (${policyLabel(repo.security_policy)})`));
    for (const policy of allowedPolicies(repo.security_policy)) {
      scopePolicySelect.appendChild(Option(policy.value, policy.label));
    }
  }

  function renderScopes(): void {
    clear(scopePills);
    for (const scope of state.scopes) {
      const repo = state.repositories.find((item) => item.internal_name === scope.internal_name);
      const pill = el('span', { class: 'badge', dataset: { scope: scope.internal_name } },
        `${repo?.display_name || scope.internal_name} · ${policyLabel(scope.security_policy_override || repo?.security_policy)}`,
      );
      const remove = el('button', { type: 'button', title: 'Remove scope' }, '×');
      remove.addEventListener('click', () => {
        state.scopes = state.scopes.filter((item) => item !== scope);
        renderScopes();
      });
      pill.appendChild(remove);
      scopePills.appendChild(pill);
    }
  }

  scopeRepoSelect.addEventListener('change', updateScopePolicies);
  scopeAdd.addEventListener('click', () => {
    const repo = state.repositories.find((item) => item.internal_name === scopeRepoSelect.value);
    if (!repo) return;
    const existing = state.scopes.find((scope) => scope.internal_name === repo.internal_name);
    if (existing) {
      const pill = scopePills.querySelector(`[data-scope="${CSS.escape(repo.internal_name)}"]`);
      if (pill) pill.scrollIntoView({ block: 'nearest' });
      setStatus(scopeStatus, 'That repository is already in the scopes.');
      return;
    }
    const override = asPolicy(scopePolicySelect.value);
    state.scopes.push({
      internal_name: repo.internal_name,
      security_policy_override: override,
    });
    renderScopes();
    setStatus(scopeStatus, '');
  });

  agentSelect.addEventListener('change', () => {
    selectAgent(agentSelect.value, { restoreManual: agentSelect.value === '' });
  });

  async function selectAgent(key: string, opts: { restoreManual?: boolean } = {}): Promise<void> {
    const previousAgent = selectedAgent();
    if (opts.restoreManual && previousAgent) {
      // Keep the manual provider/model that were active before the agent was chosen.
      state.manualProvider = state.manualProvider || state.currentProvider;
      state.manualModel = state.manualModel || state.currentModel;
    }
    state.selectedAgentKey = key;
    const agent = selectedAgent();
    if (!agent) {
      state.agentModelMissing = false;
      providerSelect.disabled = false;
      modelSelect.disabled = false;
      const provider = state.manualProvider || state.providers[0]?.key || '';
      if (!provider) {
        updateComposerState();
        return;
      }
      state.manualProvider = provider;
      await loadModelsFor(provider, { preferModel: state.manualModel || state.modelsStored[provider] });
      renderAgentState();
      updateComposerState();
      return;
    }
    if (opts.restoreManual) {
      // The current provider/model are the manual ones; remember them now.
      state.manualProvider = state.currentProvider || state.manualProvider;
      state.manualModel = state.currentModel || state.manualModel;
      persistManualSelection();
    }
    providerSelect.disabled = true;
    modelSelect.disabled = true;
    state.currentParams = [];
    renderSettings();
    renderAgentState();
    await loadModelsFor(agent.provider_key, { agent: true });
    providerSelect.value = agent.provider_key;
    renderAgentState();
    updateComposerState();
  }

  providerSelect.addEventListener('change', async () => {
    if (state.selectedAgentKey) return;
    state.manualProvider = providerSelect.value;
    state.manualModel = state.modelsStored[state.manualProvider] || '';
    persistManualSelection();
    await loadModelsFor(state.manualProvider, { preferModel: state.manualModel });
  });

  modelSelect.addEventListener('change', () => {
    if (state.selectedAgentKey) return;
    state.manualModel = modelSelect.value;
    state.currentModel = modelSelect.value;
    state.modelsStored[state.manualProvider] = state.manualModel;
    persistManualSelection();
    renderSettings();
    updateComposerState();
  });

  textarea.addEventListener('input', () => {
    updateComposerState();
  });
  textarea.addEventListener('click', () => {
    if (state.blockingId !== null) focusApproval();
  });
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (state.blockingId !== null) focusApproval();
      else sendPrompt();
    }
  });

  sendButton.addEventListener('click', sendPrompt);
  continueButton.addEventListener('click', () => continueGeneration());
  stopButton.addEventListener('click', () => { void stopGeneration(); });

  function composerPayload(): GenerationRequest {
    const agent = selectedAgent();
    return {
      agent_id: agent ? agent.internal_name : null,
      provider_key: agent ? agent.provider_key : state.currentProvider || null,
      model_id: agent ? agent.model_id : state.currentModel || null,
      inference_config: agent ? {} : pickDefined(currentConfigValues()),
      scopes: state.scopes.map((scope) => ({
        internal_name: scope.internal_name,
        security_policy_override: scope.security_policy_override,
      })),
    };
  }

  function setConversationInUrl(id: number): void {
    conversationId = id;
    const hash = `#/chat/${id}`;
    window.history.replaceState(null, '', hash);
    writeValue('lastHash', hash);
  }

  function updateHeader(): void {
    title.textContent = conversationId
      ? (conversationDetails ? conversationLabel(conversationDetails) : `Conversation #${conversationId}`)
      : 'New conversation';
  }

  async function loadConversation(): Promise<void> {
    if (!conversationId) return;
    conversationAvailable = false;
    updateComposerState();
    clear(history);
    history.appendChild(spinnerLabel('Loading conversation…'));
    try {
      const details = await api.get<ConversationDetails>(`/api/conversations/${conversationId}`);
      if (destroyed) return;
      conversationDetails = details;
      conversationAvailable = true;
      timeline.reset();
      setEmptyVisible(false);
      for (const record of details.messages) timeline.addStored(record);
      state.blockingId = typeof details.blocking_message_id === 'number' ? details.blocking_message_id : null;
      timeline.setBlocking(state.blockingId);
      updateHeader();
      updateBlockingUI();
      scrollToBottom(true);
      updateComposerState();
      await attachToStream();
    } catch (error) {
      if (destroyed) return;
      conversationAvailable = false;
      clear(history);
      history.appendChild(errorState(messageOf(error) || 'Could not load the conversation.', loadConversation));
      updateComposerState();
    }
  }

  function finishStream(token: number, controller: AbortController): void {
    if (destroyed || token !== state.streamToken) return;
    timeline.endStream();
    state.generating = false;
    updateComposerState();
    if (controller === state.streamController) state.streamController = null;
  }

  function announceCompletion(): void {
    if (state.generationError) return;
    announce(state.blockingId ? 'Waiting for approval.' : 'Response complete.');
  }

  function reportStreamError(error: Error): void {
    const message = `Connection lost: ${error.message}`;
    showGenerationError(message);
    notify(message, { level: 'error' });
  }

  async function attachToStream(): Promise<void> {
    if (!conversationId) return;
    const token = ++state.streamToken;
    const controller = new AbortController();
    state.streamController = controller;
    state.generating = true;
    updateComposerState();
    let opened: NdjsonStream<StreamEvent>;
    try {
      opened = await openNdjson<StreamEvent>(`/api/conversations/${conversationId}/stream?after_sequence_id=${timeline.lastSequence}`, {
        signal: controller.signal,
      });
    } catch (error) {
      if (!isAbort(error)) console.error(error);
      finishStream(token, controller);
      return;
    }
    if (destroyed || token !== state.streamToken) return;
    const result = await consumeStream(opened, controller);
    if (!result) return;
    if (result.error) {
      if (result.sawEvent) reportStreamError(result.error);
      else console.error(result.error);
    } else if (result.sawEvent) {
      announceCompletion();
    }
  }

  async function consumeStream(opened: NdjsonStream<StreamEvent>, controller: AbortController): Promise<{ error: Error | null; sawEvent: boolean } | null> {
    const token = ++state.streamToken;
    let sawEvent = false;
    try {
      for await (const event of opened.events) {
        sawEvent = true;
        timeline.handleEvent(event);
        autoScroll();
      }
      return { error: null, sawEvent };
    } catch (error) {
      if (destroyed || token !== state.streamToken || isAbort(error)) return null;
      return { error: asError(error), sawEvent };
    } finally {
      finishStream(token, controller);
    }
  }

  async function stopGeneration(): Promise<void> {
    if (!conversationId || !state.generating || state.stopRequested) return;
    state.stopRequested = true;
    stopButton.disabled = true;
    try {
      await api.post(`/api/conversations/${conversationId}/cancel`, {});
    } catch (error) {
      state.stopRequested = false;
      if (error instanceof ApiError && error.status === 409) {
        announce('Generation already finished.');
        updateComposerState();
        return;
      }
      notify(messageOf(error) || 'Could not stop generation.', { level: 'error' });
      updateComposerState();
      return;
    }
    const controller = state.streamController;
    if (controller === null) {
      state.stopRequested = false;
      announce('Generation stopped.');
      return;
    }
    controller.abort();
    announce('Stopping generation…');
  }

  async function sendPrompt() {
    if (sendButton.disabled) {
      if (state.blockingId !== null) focusApproval();
      return;
    }
    const text = textarea.value.trim();
    if (!text) return;
    clearGenerationError();
    setEmptyVisible(false);
    const userBlock = timeline.addUserMessage(text);
    const previousText = textarea.value;
    textarea.value = '';
    scrollToBottom(true);
    state.generating = true;
    updateComposerState();

    const payload: PromptRequest = { ...composerPayload(), prompt: text, conversation_id: conversationId };
    const controller = new AbortController();
    state.streamController = controller;

    let opened: NdjsonStream<StreamEvent>;
    try {
      opened = await openNdjson<StreamEvent>('/api/conversations/prompt', {
        method: 'POST',
        body: payload,
        signal: controller.signal,
      });
    } catch (error) {
      if (destroyed) return;
      timeline.removeBlock(userBlock);
      setEmptyVisible(!conversationId);
      textarea.value = previousText;
      state.generating = false;
      state.streamController = null;
      updateComposerState();
      if (error instanceof ApiError && error.status === 403 && error.blockingMessageId) {
        state.blockingId = error.blockingMessageId;
        updateBlockingUI();
        notify(messageOf(error) || 'Action required.', { level: 'error' });
        if (conversationId) await loadConversation();
      } else {
        notify(messageOf(error) || 'Could not send the message.', { level: 'error' });
      }
      return;
    }

    if (!conversationId && opened.conversationId) {
      setConversationInUrl(opened.conversationId);
      options.onConversationsChanged?.();
    }
    if (!conversationDetails && conversationId) {
      conversationDetails = { id: conversationId, title: null, created_at: new Date().toISOString(), blocking_message_id: null, messages: [] };
    }
    updateHeader();
    const result = await consumeStream(opened, controller);
    if (state.stopRequested) {
      state.stopRequested = false;
      announce('Generation stopped.');
      await loadConversation();
      return;
    }
    if (result?.error) reportStreamError(result.error);
    else if (result) announceCompletion();
  }

  async function continueGeneration() {
    if (continueButton.disabled || !conversationId) return;
    clearGenerationError();
    state.generating = true;
    updateComposerState();
    const payload = composerPayload();
    const controller = new AbortController();
    state.streamController = controller;
    let opened: NdjsonStream<StreamEvent>;
    try {
      opened = await openNdjson<StreamEvent>(`/api/conversations/${conversationId}/continue`, {
        method: 'POST',
        body: payload,
        signal: controller.signal,
      });
    } catch (error) {
      if (destroyed) return;
      state.generating = false;
      state.streamController = null;
      updateComposerState();
      if (error instanceof ApiError && error.status === 403 && error.blockingMessageId) {
        state.blockingId = error.blockingMessageId;
        updateBlockingUI();
        await loadConversation();
      } else {
        notify(messageOf(error) || 'Could not continue the conversation.', { level: 'error' });
      }
      return;
    }
    const result = await consumeStream(opened, controller);
    if (state.stopRequested) {
      state.stopRequested = false;
      announce('Generation stopped.');
      await loadConversation();
      return;
    }
    if (result?.error) reportStreamError(result.error);
    else if (result) announceCompletion();
  }

  async function decide(id: number, decision: 'approve' | 'reject', comment: string, _block: ToolBlock): Promise<void> {
    if (!conversationId) return;
    try {
      const payload: ToolCallDecisionRequest = { decision };
      if (decision === 'reject' && comment) payload.comment = comment;
      const result = await api.post<ToolCallDecisionResponse>(`/api/conversations/${conversationId}/tool_calls/${id}/decide`, payload);
      state.blockingId = null;
      timeline.applyDecision(id, decision, comment, result.result);
      updateBlockingUI();
      notify(decision === 'approve' ? 'Action approved.' : 'Action rejected.', { level: 'success' });
      await continueGeneration();
    } catch (error) {
      state.blockingId = id;
      timeline.setBlocking(id);
      updateBlockingUI();
      notify(messageOf(error) || 'Could not record the decision.', { level: 'error' });
    }
  }

  function resetForNewChat() {
    if (state.streamController) state.streamController.abort();
    state.streamToken += 1;
    state.streamController = null;
    conversationId = null;
    conversationDetails = null;
    conversationAvailable = true;
    state.generating = false;
    state.stopRequested = false;
    state.blockingId = null;
    state.generationError = false;
    state.lastAnnouncedBlocking = null;
    state.scopes = [];
    timeline.reset();
    setEmptyVisible(true);
    history.appendChild(emptyView);
    renderScopes();
    clear(approvalAlert);
    approvalAlert.hidden = true;
    clearGenerationError();
    textarea.value = '';
    updateHeader();
    updateBlockingUI();
    updateComposerState();
    announce('');
    textarea.focus();
  }

  if (conversationId) {
    updateHeader();
    loadConversation();
  } else {
    updateHeader();
    setEmptyVisible(true);
  }

  loadProviders();
  loadAgents();
  loadRepositories();
  updateScopePolicies();
  updateComposerState();
  updateJump();
  textarea.focus();

  return {
    element: root,
    newChat: resetForNewChat,
    focus() {
      textarea.focus();
    },
    destroy() {
      destroyed = true;
      state.streamToken += 1;
      state.streamController?.abort();
    },
  };
}
