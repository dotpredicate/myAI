import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, navigate, encodeKey, asError, renderParamFields, statusMessage as setStatus, validateKey, validateRequired, reportFieldErrors, messageOf, type FieldControl, type MountOptions, type ParamValue, type Route } from '../ui.js';
import { api } from '../api.js';
import type { AgentConfig, AgentList, CreateProjectRequest, DeleteResponse, InferenceParam, InferenceParamList, ProjectConfig, ProjectList, ProviderInfo, ProviderList, ProviderModel, ProviderModelList, RepositoryConfig, RepositoryList, UpdateProjectRequest } from '../api.js';
import { Row, Stack, Grid, Page, Heading, Card, Hint, Notice, Button, LinkButton, Input, TextArea, Select, Option, Spacer, List, ListItem, Form, Summary } from '../components.js';
import { createRepositoryAccessEditor, type RepositoryAccessEntry } from './repositoryAccess.js';

export function mount(route: Route, _options: MountOptions = {}) {
  const root = Page({});
  const listContainer = el('div', { class: 'stack' });
  const editorContainer = el('div', { class: 'inset stack', id: 'project-editor' });
  const listHeader = el('header', {},
    Row(Heading(2, 'Projects'), LinkButton('+ New project', '#/projects/new', { variant: 'primary', small: true })),
  );
  const listCard = Card(listHeader, listContainer);
  root.appendChild(Heading(1, 'Projects'));
  root.appendChild(listCard);

  interface ProjectsData {
    projects: ProjectConfig[];
    agents: AgentConfig[];
    repositories: RepositoryConfig[];
    providers: ProviderInfo[];
    agentsLoaded: boolean;
    repositoriesLoaded: boolean;
    providersLoaded: boolean;
  }

  interface ProjectEditor {
    key: string;
    onProvidersLoaded(providers: ProviderInfo[]): void;
    onProvidersError(error: Error, retry: () => void): void;
    onRepositoriesLoaded(repositories: RepositoryConfig[]): void;
    onRepositoriesError(error: Error, retry: () => void): void;
    onAgentsLoaded(agents: AgentConfig[]): void;
    onAgentsError(error: Error, retry: () => void): void;
  }

  const data: ProjectsData = {
    projects: [],
    agents: [],
    repositories: [],
    providers: [],
    agentsLoaded: false,
    repositoriesLoaded: false,
    providersLoaded: false,
  };

  let editor: ProjectEditor | null = null;
  let destroyed = false;
  let currentMode: string | undefined = route.mode;
  let currentKey: string | null = route.key || null;

  function renderList(mode: 'ready' | 'loading' | 'error' = 'ready', message = ''): void {
    clear(listContainer);
    if (mode === 'loading') {
      listContainer.appendChild(loadingState('Loading projects…'));
      return;
    }
    if (mode === 'error') {
      listContainer.appendChild(errorState(message || 'Could not load projects.', loadProjects));
      return;
    }
    if (currentMode === 'new') listContainer.appendChild(editorContainer);
    if (data.projects.length === 0) {
      listContainer.appendChild(emptyState(
        'No projects',
        'Create one to bundle an agent, model, instructions and repositories.',
      ));
      return;
    }
    const list = List();
    for (const project of data.projects) {
      const selected = currentMode === 'edit' && project.internal_name === currentKey;
      const open = () => navigate(selected
        ? '#/projects'
        : `#/projects/${encodeKey(project.internal_name)}`);
      const row = ListItem(el('button', { type: 'button', class: 'collection-row-main', onClick: open },
        el('span', { class: 'collection-row-title' }, project.display_name || project.internal_name),
        el('span', { class: 'collection-row-path truncate' }, project.internal_name),
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
    if (currentMode === 'edit' && currentKey && !data.projects.some((project) => project.internal_name === currentKey)) {
      listContainer.appendChild(emptyState(
        'Project not found',
        `There is no project with the key “${currentKey}”.`,
        LinkButton('Back to projects', '#/projects', { small: true }),
      ));
    }
  }

  async function loadProjects(): Promise<void> {
    renderList('loading');
    try {
      const response = await api.get<ProjectList>('/api/projects');
      if (destroyed) return;
      data.projects = Array.isArray(response.projects) ? response.projects : [];
      renderList();
      if (currentMode === 'edit') {
        const project = data.projects.find((item) => item.internal_name === currentKey);
        if (project && (!editor || editor.key !== currentKey)) renderEditor(project);
      }
    } catch (error) {
      if (destroyed) return;
      renderList('error', messageOf(error));
    }
  }

  async function loadRepositories(): Promise<void> {
    try {
      const response = await api.get<RepositoryList>('/api/repositories');
      if (destroyed) return;
      data.repositories = Array.isArray(response.repositories) ? response.repositories : [];
      data.repositoriesLoaded = true;
      editor?.onRepositoriesLoaded(data.repositories);
    } catch (error) {
      if (destroyed) return;
      data.repositoriesLoaded = false;
      editor?.onRepositoriesError(asError(error), loadRepositories);
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

  async function loadAgents(): Promise<void> {
    try {
      const response = await api.get<AgentList>('/api/agents');
      if (destroyed) return;
      data.agents = Array.isArray(response.agents) ? response.agents : [];
      data.agentsLoaded = true;
      editor?.onAgentsLoaded(data.agents);
    } catch (error) {
      if (destroyed) return;
      data.agentsLoaded = false;
      editor?.onAgentsError(asError(error), loadAgents);
    }
  }

  function renderEditor(project: ProjectConfig | null): void {
    clear(editorContainer);
    const isEdit = Boolean(project);
    const originalKey = project?.internal_name || '';
    let providerKey = project?.provider_key || '';
    let modelId = project?.model_id || '';
    let models: ProviderModel[] = [];
    let params: InferenceParam[] = [];
    let config: Record<string, ParamValue> = { ...project?.inference_config };
    const initialAccess: RepositoryAccessEntry[] =
      (project?.repository_access || []).map((entry) => ({
        repository_id: entry.repository_id,
        repository_internal_name: entry.repository_internal_name,
        security_policy_override: entry.security_policy_override || null,
      }));
    let dirty = false;
    let saving = false;
    let requestToken = 0;
    let providers = data.providers;
    let repositories = data.repositories;
    let agents = data.agents;

    function markDirty(): void {
      dirty = true;
    }

    const repoAccess = createRepositoryAccessEditor({
      initial: initialAccess,
      onChange: markDirty,
      subject: 'project',
    });

    const heading = Heading(2, isEdit ? `Edit ${project?.display_name || originalKey}` : 'New project');
    const formError = Notice({ level: 'error' });

    const nameInput = Input({ name: 'display_name', value: project?.display_name || '', autocomplete: 'off' });
    const keyInput = Input({
      name: 'internal_name',
      value: originalKey,
      autocomplete: 'off',
      readonly: isEdit,
      placeholder: 'my-project',
    });
    const descriptionInput = TextArea({ name: 'description', rows: 3, value: project?.description || '' });
    const instructionsInput = TextArea({ name: 'instructions', rows: 5, value: project?.instructions || '', placeholder: 'Instructions that apply to every conversation in this project…' });

    const nameField = labeledField('Name', nameInput);
    const keyField = labeledField('Key', keyInput, {
      hint: isEdit ? 'The key cannot be changed after saving.' : 'Letters, numbers, hyphens and underscores only.',
    });
    const descriptionField = labeledField('Description', descriptionInput);
    const instructionsField = labeledField('Instructions', instructionsInput, {
      hint: 'Applied to every conversation in this project, before the agent instructions.',
    });

    const defaultAgentSelect = Select({ name: 'default_agent' });
    const defaultAgentField = labeledField('Default agent', defaultAgentSelect, {
      hint: 'Preselected in the chat when this project is chosen.',
    });
    const agentStatus = Hint();
    defaultAgentField.wrapper.appendChild(agentStatus);

    const providerSelect = Select({ name: 'provider_key' });
    const modelSelect = Select({ name: 'model_id' });
    const providerField = labeledField('Provider', providerSelect);
    const modelField = labeledField('Model', modelSelect);
    const providerStatus = Hint();
    const modelStatus = Hint();
    providerField.wrapper.appendChild(providerStatus);
    modelField.wrapper.appendChild(modelStatus);

    const settingsDetails = el('details', { hidden: true });
    const settingsBody = Stack();
    const paramGrid = Grid({});
    const resetSettings = Button('Reset to defaults', { variant: 'quiet', small: true });
    settingsBody.appendChild(paramGrid);
    settingsBody.appendChild(Row(resetSettings));
    settingsDetails.appendChild(Summary('Model settings'));
    settingsDetails.appendChild(settingsBody);

    const saveButton = Button(isEdit ? 'Save' : 'Create project', { variant: 'primary', submit: true });
    const cancelButton = Button('Cancel');
    const deleteButton = isEdit ? Button('Delete', { variant: 'danger' }) : null;

    const form = Form(
      { novalidate: true },
      Grid({ columns: 2 }, nameField.wrapper, keyField.wrapper),
      descriptionField.wrapper,
      instructionsField.wrapper,
      defaultAgentField.wrapper,
      Grid({ columns: 2 }, providerField.wrapper, modelField.wrapper),
      providerStatus,
      modelStatus,
      settingsDetails,
      repoAccess.section,
      formError,
      Row(saveButton, cancelButton, Spacer(), deleteButton),
    );

    editorContainer.appendChild(el('header', {}, heading));
    editorContainer.appendChild(form);

    [nameInput, keyInput, descriptionInput, instructionsInput].forEach((input) => {
      input.addEventListener('input', markDirty);
    });
    defaultAgentSelect.addEventListener('change', markDirty);
    providerSelect.addEventListener('change', () => {
      markDirty();
      const previous = modelId;
      providerKey = providerSelect.value;
      modelId = previous;
      if (providerKey) {
        loadModelsFor(providerKey, { previousModel: previous, providerChanged: true });
      } else {
        models = [];
        modelId = '';
        params = [];
        config = {};
        clear(modelSelect);
        modelSelect.appendChild(Option('', 'Select a provider'));
        modelSelect.disabled = true;
        renderSettings();
      }
    });
    modelSelect.addEventListener('change', () => {
      markDirty();
      modelId = modelSelect.value;
      modelField.setError('');
      renderSettings();
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
      providerSelect.disabled = false;
      providerSelect.appendChild(Option('', 'No model preset'));
      for (const provider of providers) {
        providerSelect.appendChild(Option(provider.key, provider.display_name));
      }
      providerSelect.value = providerKey;
    }

    function fillAgentSelect(): void {
      clear(defaultAgentSelect);
      if (!data.agentsLoaded) {
        defaultAgentSelect.appendChild(Option('', 'Loading agents…'));
        defaultAgentSelect.disabled = true;
        return;
      }
      defaultAgentSelect.disabled = false;
      defaultAgentSelect.appendChild(Option('', 'No default agent'));
      for (const agent of agents) {
        const text = agent.display_name && agent.display_name !== agent.internal_name
          ? `${agent.display_name} (${agent.internal_name})`
          : agent.internal_name;
        defaultAgentSelect.appendChild(Option(String(agent.id), text));
      }
      defaultAgentSelect.value = project?.default_agent ? String(project.default_agent) : '';
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
      }, { idPrefix: 'project-setting' });
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
      models = [];
      modelId = options.previousModel ?? modelId;
      clear(modelSelect);
      modelSelect.appendChild(Option('', 'Loading models…'));
      modelSelect.disabled = true;
      clear(modelStatus);
      if (!key) {
        modelId = '';
        clear(modelSelect);
        modelSelect.appendChild(Option('', 'Select a provider'));
        params = [];
        renderSettings();
        return;
      }
      modelStatus.appendChild(spinnerLabel('Loading models…'));
      try {
        const response = await api.get<ProviderModelList>(`/api/providers/${encodeURIComponent(key)}/models`);
        if (destroyed || token !== requestToken) return;
        models = Array.isArray(response.models) ? response.models : [];
        clear(modelSelect);
        modelSelect.appendChild(Option('', models.length ? 'Select a model' : 'No models available'));
        for (const model of models) {
          modelSelect.appendChild(Option(model.id, model.name));
        }
        let missing = false;
        if (modelId && models.some((model) => model.id === modelId)) {
          modelSelect.value = modelId;
        } else if (modelId) {
          missing = true;
          modelId = '';
          modelSelect.value = '';
        } else if (models.length > 0) {
          const first = models[0];
          if (first) {
            modelId = first.id;
            modelSelect.value = modelId;
          }
        }
        modelSelect.disabled = models.length === 0;
        modelField.setError(missing ? `The previous model is not available from “${key}”. Choose a model.` : '');
        setStatus(modelStatus, missing ? 'Choose a model for the new provider.' : '', { error: missing });
        await loadParamsFor(key, token);
      } catch (error) {
        if (destroyed || token !== requestToken) return;
        clear(modelSelect);
        modelSelect.appendChild(Option('', 'Models unavailable'));
        modelSelect.disabled = true;
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
      modelField.setError('');
      setFormError('');
      const displayName = nameInput.value.trim();
      const internalName = keyInput.value.trim();
      const selectedProvider = providerSelect.value;
      const selectedModel = modelSelect.value;
      const errors: FieldControl[] = [];
      validateRequired(nameField, displayName, 'Name is required.', errors);
      if (!isEdit) validateKey(keyField, internalName, errors);
      if (selectedProvider) {
        validateRequired(modelField, selectedModel, 'Choose a model or clear the provider.', errors);
      }
      if (reportFieldErrors(errors)) return;

      const defaultAgent = defaultAgentSelect.value ? Number(defaultAgentSelect.value) : null;
      const payload: UpdateProjectRequest = {
        display_name: displayName,
        description: descriptionInput.value.trim(),
        instructions: instructionsInput.value.trim() || null,
        default_agent: defaultAgent,
        provider_key: selectedProvider || null,
        model_id: selectedProvider ? (selectedModel || null) : null,
        inference_config: selectedProvider ? config : {},
        repository_access: repoAccess.entries.map((entry) => ({
          repository_id: entry.repository_id,
          security_policy_override: entry.security_policy_override,
        })),
      };
      saving = true;
      saveButton.disabled = true;
      saveButton.textContent = isEdit ? 'Saving…' : 'Creating…';
      try {
        let saved;
        if (isEdit) {
          saved = await api.put<ProjectConfig>(`/api/projects/${encodeKey(originalKey)}`, payload);
        } else {
          const createPayload: CreateProjectRequest = {
            ...payload,
            display_name: displayName,
            internal_name: internalName,
          };
          saved = await api.post<ProjectConfig>('/api/projects', createPayload);
        }
        if (destroyed) return;
        dirty = false;
        notify(isEdit ? 'Project saved.' : 'Project created.', { level: 'success' });
        if (!isEdit) {
          navigate(`#/projects/${encodeKey(saved.internal_name)}`);
        } else {
          const index = data.projects.findIndex((item) => item.internal_name === saved.internal_name);
          if (index >= 0) data.projects[index] = saved;
          renderList();
          renderEditor(saved);
        }
      } catch (error) {
        if (destroyed) return;
        setFormError(messageOf(error) || 'Could not save the project.');
        notify(messageOf(error) || 'Could not save the project.', { level: 'error' });
      } finally {
        if (!destroyed) {
          saving = false;
          saveButton.disabled = false;
          saveButton.textContent = isEdit ? 'Save' : 'Create project';
        }
      }
    });

    cancelButton.addEventListener('click', async () => {
      if (dirty) {
        const discard = await confirmAction({
          title: 'Discard changes?',
          message: 'The project form has unsaved changes.',
          confirmLabel: 'Discard changes',
        });
        if (!discard) return;
      }
      navigate('#/projects');
    });

    deleteButton?.addEventListener('click', async () => {
      const confirmed = await confirmAction({
        title: 'Delete project?',
        message: `“${project?.display_name || originalKey}” will be removed. Conversations keep their history but lose the project.`,
        confirmLabel: 'Delete project',
      });
      if (!confirmed) return;
      deleteButton.disabled = true;
      try {
        await api.del<DeleteResponse>(`/api/projects/${encodeKey(originalKey)}`);
        notify('Project deleted.', { level: 'success' });
        navigate('#/projects');
      } catch (error) {
        deleteButton.disabled = false;
        setFormError(messageOf(error) || 'Could not delete the project.');
        notify(messageOf(error) || 'Could not delete the project.', { level: 'error' });
      }
    });

    fillProviderSelect();
    fillAgentSelect();
    repoAccess.setRepositories(repositories, data.repositoriesLoaded);
    renderSettings();

    if (providerKey) {
      loadModelsFor(providerKey, { previousModel: modelId });
    } else {
      clear(modelSelect);
      modelSelect.appendChild(Option('', 'Select a provider'));
      modelSelect.disabled = true;
    }

    editor = {
      key: originalKey || '__new__',
      onProvidersLoaded(nextProviders) {
        providers = nextProviders;
        fillProviderSelect();
      },
      onProvidersError(error, retry) {
        setStatus(providerStatus, `Could not load providers: ${messageOf(error)}`, { error: true, retry });
      },
      onRepositoriesLoaded(nextRepositories) {
        repositories = nextRepositories;
        repoAccess.setRepositories(repositories, true);
      },
      onRepositoriesError(error, retry) {
        repoAccess.setError(error, retry);
      },
      onAgentsLoaded(nextAgents) {
        agents = nextAgents;
        fillAgentSelect();
      },
      onAgentsError(error, retry) {
        setStatus(agentStatus, `Could not load agents: ${messageOf(error)}`, { error: true, retry });
      },
    };
  }

  if (currentMode === 'new') renderEditor(null);
  loadProjects();
  loadRepositories();
  loadProviders();
  loadAgents();

  return {
    element: root,
    destroy() {
      destroyed = true;
      editor = null;
    },
  };
}
