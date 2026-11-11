import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, navigate, encodeKey, asError, renderParamFields, createPicker, statusMessage as setStatus, validateKey, validateRequired, reportFieldErrors, messageOf, type FieldControl, type MountOptions, type ParamValue, type Route } from '../ui.js';
import { api } from '../core/api.js';
import { modelChoices } from '../core/models.js';
import type { AgentConfig, AgentList, CreateAgentRequest, DeleteResponse, InferenceParam, InferenceParamList, ProviderInfo, ProviderList, ProviderModelList, UpdateAgentRequest } from '../core/api.js';
import { Row, Stack, Grid, Page, Heading, Card, Hint, Notice, Button, LinkButton, Input, TextArea, Select, Option, Spacer, List, ListItem, Form, Summary } from '../components.js';

export function mount(route: Route, _options: MountOptions = {}) {
  const root = Page({});
  const listContainer = el('div', { class: 'stack' });
  const editorContainer = el('div', { class: 'inset stack', id: 'agent-editor' });
  const listHeader = el('header', {},
    Row(Heading(2, 'Agents'), LinkButton('+ New agent', '#/agents/new', { variant: 'primary', small: true })),
  );
  const listCard = Card(listHeader, listContainer);
  root.appendChild(Heading(1, 'Agents'));
  root.appendChild(listCard);

  interface AgentsData {
    agents: AgentConfig[];
    providers: ProviderInfo[];
    providersLoaded: boolean;
  }

  interface AgentEditor {
    key: string;
    onProvidersLoaded(providers: ProviderInfo[]): void;
    onProvidersError(error: Error, retry: () => void): void;
    onAgentsLoaded(agents: AgentConfig[]): void;
  }

  const data: AgentsData = {
    agents: [],
    providers: [],
    providersLoaded: false,
  };

  let editor: AgentEditor | null = null;
  let destroyed = false;
  let currentMode: string | undefined = route.mode;
  let currentKey: string | null = route.key || null;

  function renderList(mode: 'ready' | 'loading' | 'error' = 'ready', message = ''): void {
    clear(listContainer);
    if (mode === 'loading') {
      listContainer.appendChild(loadingState('Loading agents…'));
      return;
    }
    if (mode === 'error') {
      listContainer.appendChild(errorState(message || 'Could not load agents.', loadAgents));
      return;
    }
    if (currentMode === 'new') listContainer.appendChild(editorContainer);
    if (data.agents.length === 0) {
      listContainer.appendChild(emptyState(
        'No agents',
        'Create one to save a model and instructions.',
      ));
      return;
    }
    const list = List();
    for (const agent of data.agents) {
      const selected = currentMode === 'edit' && agent.internal_name === currentKey;
      const open = () => navigate(selected
        ? '#/agents'
        : `#/agents/${encodeKey(agent.internal_name)}`);
      const row = ListItem(el('button', { type: 'button', class: 'collection-row-main', onClick: open },
        el('span', { class: 'collection-row-title' }, agent.display_name || agent.internal_name),
        el('span', { class: 'collection-row-path truncate' }, agent.internal_name),
      ));
      row.dataset.selected = selected ? 'true' : 'false';
      list.appendChild(row);
      if (selected) {
        const detail = el('li', { class: 'list-detail' });
        detail.appendChild(editorContainer);
        list.appendChild(detail);
      }
    }
    listContainer.appendChild(list);
    if (currentMode === 'edit' && currentKey && !data.agents.some((agent) => agent.internal_name === currentKey)) {
      listContainer.appendChild(emptyState(
        'Agent not found',
        `There is no agent with the key “${currentKey}”.`,
        LinkButton('Back to agents', '#/agents', { small: true }),
      ));
    }
  }

  async function loadAgents(): Promise<void> {
    renderList('loading');
    try {
      const response = await api.get<AgentList>('/api/agents');
      if (destroyed) return;
      data.agents = Array.isArray(response.agents) ? response.agents : [];
      renderList();
      editor?.onAgentsLoaded(data.agents);
      if (currentMode === 'edit') {
        const agent = data.agents.find((item) => item.internal_name === currentKey);
        if (agent && (!editor || editor.key !== currentKey)) renderEditor(agent);
      }
    } catch (error) {
      if (destroyed) return;
      renderList('error', messageOf(error));
    }
  }

  async function loadProviders(): Promise<void> {
    try {
      const response = await api.get<ProviderList>('/api/providers');
      if (destroyed) return;
      data.providers = Array.isArray(response.providers) ? response.providers : [];
      data.providersLoaded = true;
      editor?.onProvidersLoaded(data.providers);
    } catch (error) {
      if (destroyed) return;
      data.providersLoaded = false;
      editor?.onProvidersError(asError(error), loadProviders);
    }
  }

  function renderEditor(agent: AgentConfig | null): void {
    clear(editorContainer);
    const isEdit = Boolean(agent);
    const originalKey = agent?.internal_name || '';
    let providerKey = agent?.provider_key || '';
    let modelId = agent?.model_id || '';
    let params: InferenceParam[] = [];
    let config: Record<string, ParamValue> = { ...agent?.inference_config };
    let dirty = false;
    let saving = false;
    let requestToken = 0;
    let providers = data.providers;

    function markDirty(): void {
      dirty = true;
    }

    const heading = Heading(2, isEdit ? `Edit ${agent?.display_name || originalKey}` : 'New agent');
    const formError = Notice({ level: 'error' });

    const nameInput = Input({ name: 'display_name', value: agent?.display_name || '', autocomplete: 'off' });
    const keyInput = Input({
      name: 'internal_name',
      value: originalKey,
      autocomplete: 'off',
      readonly: isEdit,
      placeholder: 'support-agent',
    });
    const descriptionInput = TextArea({ name: 'description', rows: 3, value: agent?.description || '' });
    const instructionsInput = TextArea({ name: 'instructions', rows: 5, value: agent?.instructions || '', placeholder: 'Instructions the model should follow…' });
    const spawnable = new Set<string>(agent?.spawnable_agents || []);
    const spawnableSelect = Select({ multiple: true });
    spawnableSelect.size = 5;
    function renderSpawnable(agents: AgentConfig[]): void {
      clear(spawnableSelect);
      spawnableSelect.disabled = false;
      const candidates = agents.filter((candidate) => candidate.internal_name !== originalKey);
      if (candidates.length === 0) {
        spawnableSelect.appendChild(Option('', 'Create another agent first'));
        spawnableSelect.disabled = true;
        return;
      }
      for (const candidate of candidates) {
        const text = candidate.display_name && candidate.display_name !== candidate.internal_name
          ? `${candidate.display_name} (${candidate.internal_name})`
          : candidate.internal_name;
        const option = Option(candidate.internal_name, text);
        option.selected = spawnable.has(candidate.internal_name);
        spawnableSelect.appendChild(option);
      }
    }
    spawnableSelect.addEventListener('change', () => {
      spawnable.clear();
      for (const option of Array.from(spawnableSelect.selectedOptions)) {
        spawnable.add(option.value);
      }
      markDirty();
    });
    const nameField = labeledField('Name', nameInput);
    const keyField = labeledField('Key', keyInput, {
      hint: isEdit ? 'The key cannot be changed after saving.' : 'Letters, numbers, hyphens and underscores only.',
    });
    const descriptionField = labeledField('Description', descriptionInput);
    const instructionsField = labeledField('Instructions', instructionsInput);
    const spawnableField = labeledField('Sub-agents', spawnableSelect, {
      hint: 'Agents this agent may start with spawn_agent. Ctrl/Cmd-click to select more than one.',
    });

    const providerSelect = Select({ name: 'provider_key' });
    const providerField = labeledField('Provider', providerSelect);
    const modelPicker = createPicker({
      label: 'Model',
      emptyLabel: 'Select a model',
      placeholder: 'Search models…',
      onChange(value) {
        markDirty();
        modelId = value;
        modelPicker.setError('');
        renderSettings();
      },
    });
    const providerStatus = Hint();
    const modelStatus = Hint();
    providerField.wrapper.appendChild(providerStatus);

    const settingsDetails = el('details', { hidden: true });
    const settingsBody = Stack();
    const paramGrid = Grid({});
    const resetSettings = Button('Reset to defaults', { variant: 'quiet', small: true });
    settingsBody.appendChild(paramGrid);
    settingsBody.appendChild(Row(resetSettings));
    settingsDetails.appendChild(Summary('Model settings'));
    settingsDetails.appendChild(settingsBody);

    const saveButton = Button(isEdit ? 'Save' : 'Create agent', { variant: 'primary', submit: true });
    const cancelButton = Button('Cancel');
    const deleteButton = isEdit
      ? Button('Delete', { variant: 'danger' })
      : null;

    const form = Form({ novalidate: true }, Grid({ columns: 2 }, nameField.wrapper, keyField.wrapper), descriptionField.wrapper, instructionsField.wrapper, spawnableField.wrapper, Grid({ columns: 2 }, providerField.wrapper, modelPicker.element), providerStatus, modelStatus, settingsDetails, formError, Row(saveButton, cancelButton, Spacer(), deleteButton));

    editorContainer.appendChild(el('header', {}, heading));
    editorContainer.appendChild(form);

    [nameInput, keyInput, descriptionInput, instructionsInput].forEach((input) => {
      input.addEventListener('input', markDirty);
    });
    providerSelect.addEventListener('change', () => {
      markDirty();
      const previous = modelId;
      providerKey = providerSelect.value;
      modelId = previous;
      loadModelsFor(providerKey, { previousModel: previous, providerChanged: true });
    });
    function setFormError(message: string): void {
      formError.textContent = message || '';
    }

    function fillProviderSelect(): void {
      clear(providerSelect);
      if (!data.providersLoaded) {
        providerSelect.appendChild(Option('', 'Loading providers…'));
        providerSelect.disabled = true;
        return;
      }
      if (providers.length === 0) {
        providerSelect.appendChild(Option('', 'No providers available'));
        providerSelect.disabled = true;
        return;
      }
      providerSelect.disabled = false;
      providerSelect.appendChild(Option('', 'Select a provider'));
      for (const provider of providers) {
        providerSelect.appendChild(Option(provider.key, provider.display_name));
      }
      providerSelect.value = providerKey;
    }

    function renderSettings(): void {
      const visible = params.length > 0 && Boolean(providerKey);
      settingsDetails.hidden = !visible;
      if (!visible) {
        paramGrid.replaceChildren();
        return;
      }
      renderParamFields(paramGrid, params, config, (name, value) => {
        markDirty();
        if (value === undefined || value === '') delete config[name];
        else config[name] = value;
      }, { idPrefix: 'agent-setting' });
    }

    resetSettings.addEventListener('click', () => {
      config = {};
      markDirty();
      renderSettings();
    });

    function filterConfigToParams(): void {
      const names = new Set(params.map((param) => param.name));
      const next: Record<string, ParamValue> = {};
      for (const [name, value] of Object.entries(config)) {
        if (names.has(name)) next[name] = value;
      }
      config = next;
    }

    async function loadModelsFor(key: string, options: { previousModel?: string; providerChanged?: boolean } = {}): Promise<void> {
      const token = ++requestToken;
      modelId = options.previousModel ?? modelId;
      modelPicker.setItems([]);
      modelPicker.setValue('');
      modelPicker.setDisabled(true);
      clear(modelStatus);
      if (!key) {
        modelId = '';
        params = [];
        renderSettings();
        return;
      }
      modelStatus.appendChild(spinnerLabel('Loading models…'));
      try {
        const response = await api.get<ProviderModelList>(`/api/providers/${encodeURIComponent(key)}/models`);
        if (destroyed || token !== requestToken) return;
        const choices = modelChoices(Array.isArray(response.models) ? response.models : []);
        modelPicker.setItems(choices);
        let missing = false;
        if (modelId && choices.some((choice) => choice.value === modelId)) {
          modelPicker.setValue(modelId);
        } else if (modelId) {
          missing = true;
          modelId = '';
          modelPicker.setValue('');
        } else {
          const first = choices[0];
          if (first) {
            modelId = first.value;
            modelPicker.setValue(modelId);
          }
        }
        modelPicker.setDisabled(choices.length === 0);
        modelPicker.setError(missing ? `The previous model is not available from “${key}”. Choose a model.` : '');
        setStatus(modelStatus, missing ? 'Choose a model for the new provider.' : '', { error: missing });
        await loadParamsFor(key, token);
      } catch (error) {
        if (destroyed || token !== requestToken) return;
        modelPicker.setItems([]);
        modelPicker.setValue('');
        modelPicker.setDisabled(true);
        setStatus(modelStatus, messageOf(error) || 'Could not load models.', {
          error: true,
          retry: () => loadModelsFor(key, { previousModel: modelId }),
        });
      }
    }

    async function loadParamsFor(key: string, token: number): Promise<void> {
      try {
        const response = await api.get<InferenceParamList>(`/api/providers/${encodeURIComponent(key)}/inference-params`);
        if (destroyed || token !== requestToken) return;
        params = Array.isArray(response) ? response : [];
        filterConfigToParams();
        clear(modelStatus);
        renderSettings();
      } catch (error) {
        if (destroyed || token !== requestToken) return;
        params = [];
        renderSettings();
        setStatus(modelStatus, `Could not load model settings: ${messageOf(error)}`, { error: true });
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (saving) return;
      nameField.setError('');
      keyField.setError('');
      providerField.setError('');
      modelPicker.setError('');
      setFormError('');
      const displayName = nameInput.value.trim();
      const internalName = keyInput.value.trim();
      const selectedProvider = providerSelect.value;
      const selectedModel = modelPicker.getValue();
      const errors: FieldControl[] = [];
      validateRequired(nameField, displayName, 'Name is required.', errors);
      if (!isEdit) validateKey(keyField, internalName, errors);
      if (reportFieldErrors(errors)) return;
      const payload: UpdateAgentRequest = {
        display_name: displayName,
        description: descriptionInput.value.trim(),
        instructions: instructionsInput.value.trim() || null,
        spawnable_agents: [...spawnable],
        provider_key: selectedProvider || null,
        model_id: selectedModel || null,
        inference_config: selectedProvider && selectedModel ? config : {},
      };
      saving = true;
      saveButton.disabled = true;
      saveButton.textContent = isEdit ? 'Saving…' : 'Creating…';
      try {
        let saved;
        if (isEdit) {
          saved = await api.put<AgentConfig>(`/api/agents/${encodeKey(originalKey)}`, payload);
        } else {
          const createPayload: CreateAgentRequest = {
            ...payload,
            display_name: displayName,
            provider_key: selectedProvider || null,
            model_id: selectedModel || null,
            internal_name: internalName,
          };
          saved = await api.post<AgentConfig>('/api/agents', createPayload);
        }
        if (destroyed) return;
        dirty = false;
        notify(isEdit ? 'Agent saved.' : 'Agent created.', { level: 'success' });
        if (!isEdit) {
          navigate(`#/agents/${encodeKey(saved.internal_name)}`);
        } else {
          const index = data.agents.findIndex((item) => item.internal_name === saved.internal_name);
          if (index >= 0) data.agents[index] = saved;
          renderList();
          renderEditor(saved);
        }
      } catch (error) {
        if (destroyed) return;
        setFormError(messageOf(error) || 'Could not save the agent.');
        notify(messageOf(error) || 'Could not save the agent.', { level: 'error' });
      } finally {
        if (!destroyed) {
          saving = false;
          saveButton.disabled = false;
          saveButton.textContent = isEdit ? 'Save' : 'Create agent';
        }
      }
    });

    cancelButton.addEventListener('click', async () => {
      if (dirty) {
        const discard = await confirmAction({
          title: 'Discard changes?',
          message: 'The agent form has unsaved changes.',
          confirmLabel: 'Discard changes',
        });
        if (!discard) return;
      }
      navigate('#/agents');
    });

    deleteButton?.addEventListener('click', async () => {
      const confirmed = await confirmAction({
        title: 'Delete agent?',
        message: `“${agent?.display_name || originalKey}” will be removed. This cannot be undone.`,
        confirmLabel: 'Delete agent',
      });
      if (!confirmed) return;
      deleteButton.disabled = true;
      try {
        await api.del<DeleteResponse>(`/api/agents/${encodeKey(originalKey)}`);
        notify('Agent deleted.', { level: 'success' });
        navigate('#/agents');
      } catch (error) {
        deleteButton.disabled = false;
        setFormError(messageOf(error) || 'Could not delete the agent.');
        notify(messageOf(error) || 'Could not delete the agent.', { level: 'error' });
      }
    });

    fillProviderSelect();
    renderSpawnable(data.agents);
    renderSettings();

    if (providerKey) {
      loadModelsFor(providerKey, { previousModel: modelId });
    } else {
      modelPicker.setItems([]);
      modelPicker.setValue('');
      modelPicker.setDisabled(true);
    }

    editor = {
      key: originalKey || '__new__',
      onProvidersLoaded(nextProviders) {
        providers = nextProviders;
        fillProviderSelect();
        const first = providers[0];
        if (!providerKey && first) {
          providerKey = first.key;
          providerSelect.value = providerKey;
          loadModelsFor(providerKey, { previousModel: '' });
        } else if (providerKey) {
          loadModelsFor(providerKey, { previousModel: modelId });
        }
      },
      onProvidersError(error, retry) {
        setStatus(providerStatus, `Could not load providers: ${messageOf(error)}`, { error: true, retry });
      },
      onAgentsLoaded(nextAgents) {
        renderSpawnable(nextAgents);
      },
    };
  }

  if (currentMode === 'new') renderEditor(null);
  loadAgents();
  loadProviders();

  return {
    element: root,
    destroy() {
      destroyed = true;
      editor = null;
    },
  };
}
