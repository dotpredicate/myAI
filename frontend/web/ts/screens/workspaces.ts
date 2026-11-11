import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, navigate, writeValue, plural, messageOf, createFolderBrowser, type MountOptions, type Route } from '../ui.js';
import { api } from '../core/api.js';
import type { AddWorkspaceSourceRequest, DeleteResponse, RepositoryConfig, RepositoryList, ResolveWorkspaceResponse, UpdateWorkspaceRequest, WorkspaceConfig, WorkspaceList, WorkspaceSource } from '../core/api.js';
import { Row, Stack, Page, Heading, Card, Hint, InlineHint, FieldGroup, Badge, Notice, Button, LinkButton, Input, Select, Option, List, ListItem, Form, IconButton } from '../components.js';

function sourceLabel(source: WorkspaceSource): string {
  return source.display_name || source.internal_name;
}

function openChat(workspace: WorkspaceConfig): void {
  writeValue('chat.workspace', String(workspace.id));
  navigate('#/chat');
}

export function mount(route: Route, options: MountOptions = {}) {
  const root = Page({});

  // New workspace form
  const addToggle = Button('+ New workspace', { variant: 'primary', small: true });
  const pathInput = Input({ name: 'path', autocomplete: 'off', placeholder: '/home/you/code/project' });
  const browseButton = Button('Browse');
  const browser = createFolderBrowser(selectFolder);
  const formError = Notice({ level: 'error' });
  const addSave = Button('Create workspace', { variant: 'primary', submit: true });
  const addCancel = Button('Cancel');
  const pathField = labeledField('Folder', pathInput, {
    hint: 'The folder becomes the first source of the workspace and gives it its name.',
  });
  const addForm = Form({ novalidate: true, hidden: true, class: 'inset' },
    el('header', {}, Heading(3, 'New workspace')),
    FieldGroup(pathField.wrapper, Row(browseButton)),
    browser.element,
    formError,
    Row(addSave, addCancel),
  );

  const listContainer = el('div', {});
  const listHeader = el('header', {}, Row(Heading(2, 'Workspaces'), addToggle));
  root.appendChild(Heading(1, 'Workspaces'));
  root.appendChild(Card(listHeader, addForm, listContainer));

  interface ScreenState {
    workspaces: WorkspaceConfig[];
    repositories: RepositoryConfig[];
    repositoriesFailed: boolean;
    selectedId: number | null;
    actionsBusy: boolean;
  }

  const state: ScreenState = {
    workspaces: [],
    repositories: [],
    repositoriesFailed: false,
    selectedId: Number(route.key) || null,
    actionsBusy: false,
  };
  let destroyed = false;

  function openAddForm(): void {
    addForm.hidden = false;
  }

  function closeAddForm(): void {
    addForm.hidden = true;
    browser.hide();
  }

  function resetAddForm(): void {
    pathInput.value = '';
    pathField.setError('');
    formError.textContent = '';
    browser.hide();
  }

  function selectFolder(path: string): void {
    pathInput.value = path;
    pathInput.focus();
  }

  addToggle.addEventListener('click', () => {
    openAddForm();
    pathInput.focus();
  });
  addCancel.addEventListener('click', () => {
    closeAddForm();
    resetAddForm();
  });
  browseButton.addEventListener('click', () => {
    browser.open(pathInput.value.trim() || '~');
  });

  addForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.actionsBusy) return;
    pathField.setError('');
    formError.textContent = '';
    const path = pathInput.value.trim();
    if (!path) {
      pathField.setError('A folder is required.');
      pathInput.focus();
      return;
    }
    state.actionsBusy = true;
    addSave.disabled = true;
    addSave.replaceChildren(spinnerLabel('Creating…'));
    try {
      const resolved = await api.post<ResolveWorkspaceResponse>('/api/workspaces/resolve', { path });
      if (destroyed) return;
      notify(
        resolved.created
          ? `Workspace “${resolved.workspace.name}” created.`
          : `That folder already belongs to “${resolved.workspace.name}”.`,
        { level: resolved.created ? 'success' : 'info' },
      );
      resetAddForm();
      closeAddForm();
      navigate(`#/workspaces/${resolved.workspace.id}`);
    } catch (error) {
      if (destroyed) return;
      formError.textContent = messageOf(error) || 'Could not create the workspace.';
      notify(messageOf(error) || 'Could not create the workspace.', { level: 'error' });
    } finally {
      if (!destroyed) {
        state.actionsBusy = false;
        addSave.disabled = false;
        addSave.textContent = 'Create workspace';
      }
    }
  });

  function workspaceRow(workspace: WorkspaceConfig): HTMLLIElement {
    const selected = workspace.id === state.selectedId;
    const open = () => navigate(selected ? '#/workspaces' : `#/workspaces/${workspace.id}`);
    const sources = workspace.sources.map(sourceLabel).join(', ');
    const main = el('button', { type: 'button', class: 'collection-row-main', onClick: open },
      el('span', { class: 'collection-row-title' }, workspace.name),
      workspace.sources.length > 0
        ? el('span', { class: 'collection-row-meta' }, Badge(plural(workspace.sources.length, 'source')))
        : null,
      el('span', { class: 'collection-row-path truncate' }, sources || 'No sources'),
    );
    const deleteButton = IconButton('×', { title: 'Delete workspace' });
    deleteButton.addEventListener('click', () => deleteWorkspace(workspace));
    const row = ListItem(main, deleteButton);
    row.dataset.selected = selected ? 'true' : 'false';
    return row;
  }

  function renderList(mode: 'ready' | 'loading' | 'error' = 'ready', message = ''): void {
    clear(listContainer);
    if (mode === 'loading') {
      listContainer.appendChild(loadingState('Loading workspaces…'));
      return;
    }
    if (mode === 'error') {
      listContainer.appendChild(errorState(message || 'Could not load workspaces.', loadWorkspaces));
      return;
    }
    if (state.workspaces.length === 0) {
      listContainer.appendChild(emptyState(
        'No workspaces',
        'Create a workspace to give conversations their sources and their own model space.',
        Button('+ New workspace', { variant: 'primary', small: true, onClick: openAddForm }),
      ));
      return;
    }
    const list = List();
    for (const workspace of state.workspaces) {
      list.appendChild(workspaceRow(workspace));
      if (workspace.id === state.selectedId) {
        const detail = el('li', { class: 'list-detail' });
        const panel = el('div', { class: 'inset stack' });
        detail.appendChild(panel);
        list.appendChild(detail);
        renderDetails(workspace, panel);
      }
    }
    listContainer.appendChild(list);
    if (state.selectedId !== null && !state.workspaces.some((workspace) => workspace.id === state.selectedId)) {
      listContainer.appendChild(emptyState(
        'Workspace not found',
        `There is no workspace with the id “${state.selectedId}”.`,
        LinkButton('Back to workspaces', '#/workspaces', { small: true }),
      ));
    }
  }

  function renderDetails(workspace: WorkspaceConfig, container: HTMLElement): void {
    clear(container);

    const chatButton = Button('New chat', { variant: 'primary', small: true });
    chatButton.addEventListener('click', () => openChat(workspace));
    const deleteButton = Button('Delete', { variant: 'danger', small: true });
    deleteButton.addEventListener('click', () => deleteWorkspace(workspace));

    container.appendChild(el('header', {},
      el('div', {},
        Heading(2, workspace.name),
        Badge(plural(workspace.sources.length, 'source')),
      ),
      Row(chatButton, IconButton('×', { title: 'Close workspace', onClick: () => navigate('#/workspaces') })),
    ));

    // Rename
    const renameInput = Input({ value: workspace.name, autocomplete: 'off' });
    const renameField = labeledField('Name', renameInput);
    const renameButton = Button('Rename', { small: true, submit: true });
    const renameForm = Form({}, Row(renameField.wrapper, renameButton));
    renameForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const name = renameInput.value.trim();
      if (!name) {
        renameField.setError('Name is required.');
        renameInput.focus();
        return;
      }
      if (name === workspace.name) return;
      renameField.setError('');
      renameButton.disabled = true;
      try {
        await api.patch<WorkspaceConfig>(`/api/workspaces/${workspace.id}`, { name } satisfies UpdateWorkspaceRequest);
        notify('Workspace renamed.', { level: 'success' });
        await loadWorkspaces();
      } catch (error) {
        renameField.setError(messageOf(error) || 'Could not rename the workspace.');
      } finally {
        if (!destroyed) renameButton.disabled = false;
      }
    });
    container.appendChild(renameForm);

    // Sources
    const sourcesSection = Stack(el('header', {}, Heading(3, 'Sources')));
    container.appendChild(sourcesSection);
    if (workspace.sources.length === 0) {
      sourcesSection.appendChild(Hint('This workspace has no sources, so the model sees no repositories and search finds nothing.'));
    } else {
      const sourcesList = List();
      for (const source of workspace.sources) {
        const removeButton = IconButton('×', { title: `Remove ${source.internal_name}` });
        removeButton.addEventListener('click', () => removeSource(workspace, source));
        sourcesList.appendChild(ListItem(
          el('div', { class: 'collection-row-main' },
            el('span', { class: 'collection-row-title' }, sourceLabel(source)),
            el('span', { class: 'collection-row-meta' }, Badge(source.repo_type)),
            el('span', { class: 'collection-row-path truncate' }, source.path),
          ),
          removeButton,
        ));
      }
      sourcesSection.appendChild(sourcesList);
    }

    const used = new Set(workspace.sources.map((source) => source.repository_id));
    const candidates = state.repositories.filter((repository) => !used.has(repository.id));
    if (state.repositoriesFailed) {
      sourcesSection.appendChild(Row(
        InlineHint('Could not load repositories.'),
        Button('Retry', { small: true, variant: 'quiet', onClick: () => {
          void retryRepositories();
        } }),
      ));
    } else if (state.repositories.length === 0) {
      sourcesSection.appendChild(Row(
        InlineHint('No repositories exist yet.'),
        LinkButton('Add a repository', '#/repositories', { small: true, variant: 'quiet' }),
      ));
    } else if (candidates.length === 0) {
      sourcesSection.appendChild(InlineHint('Every repository is already a source of this workspace.'));
    } else {
      const sourceSelect = Select();
      for (const repository of candidates) {
        sourceSelect.appendChild(Option(String(repository.id), repository.display_name || repository.internal_name));
      }
      const addSourceButton = Button('Add source', { small: true, submit: true });
      const addSourceForm = Form({}, Row(sourceSelect, addSourceButton));
      addSourceForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const repositoryId = Number(sourceSelect.value);
        if (!Number.isFinite(repositoryId) || repositoryId <= 0) return;
        addSourceButton.disabled = true;
        sourceSelect.disabled = true;
        try {
          await api.post<WorkspaceConfig>(`/api/workspaces/${workspace.id}/sources`, { repository_id: repositoryId } satisfies AddWorkspaceSourceRequest);
          notify('Source added.', { level: 'success' });
          await loadWorkspaces();
        } catch (error) {
          notify(messageOf(error) || 'Could not add the source.', { level: 'error' });
          if (!destroyed) {
            addSourceButton.disabled = false;
            sourceSelect.disabled = false;
          }
        }
      });
      sourcesSection.appendChild(addSourceForm);
    }

    container.appendChild(Row(deleteButton));
  }

  async function removeSource(workspace: WorkspaceConfig, source: WorkspaceSource): Promise<void> {
    const confirmed = await confirmAction({
      title: 'Remove source?',
      message: `“${sourceLabel(source)}” stops being available in “${workspace.name}”. The repository and its files stay.`,
      confirmLabel: 'Remove source',
    });
    if (!confirmed) return;
    try {
      await api.del<WorkspaceConfig>(`/api/workspaces/${workspace.id}/sources/${source.repository_id}`);
      notify('Source removed.', { level: 'success' });
      await loadWorkspaces();
    } catch (error) {
      notify(messageOf(error) || 'Could not remove the source.', { level: 'error' });
    }
  }

  async function deleteWorkspace(workspace: WorkspaceConfig): Promise<void> {
    const confirmed = await confirmAction({
      title: 'Delete workspace?',
      message: `“${workspace.name}” will be removed. Its conversations move to another workspace and its model space is kept.`,
      confirmLabel: 'Delete workspace',
    });
    if (!confirmed) return;
    try {
      await api.del<DeleteResponse>(`/api/workspaces/${workspace.id}`);
      notify('Workspace deleted.', { level: 'success' });
      options.onConversationsChanged?.();
      if (state.selectedId === workspace.id) {
        navigate('#/workspaces');
        return;
      }
      await loadWorkspaces();
    } catch (error) {
      notify(messageOf(error) || 'Could not delete the workspace.', { level: 'error' });
    }
  }

  async function retryRepositories(): Promise<void> {
    await loadRepositories();
    renderList();
  }

  async function loadRepositories(): Promise<void> {
    try {
      const data = await api.get<RepositoryList>('/api/repositories');
      if (destroyed) return;
      state.repositories = Array.isArray(data.repositories) ? data.repositories : [];
      state.repositoriesFailed = false;
    } catch {
      if (destroyed) return;
      state.repositories = [];
      state.repositoriesFailed = true;
    }
  }

  async function loadWorkspaces(): Promise<void> {
    renderList('loading');
    try {
      const data = await api.get<WorkspaceList>('/api/workspaces');
      if (destroyed) return;
      state.workspaces = Array.isArray(data.workspaces) ? data.workspaces : [];
      renderList();
    } catch (error) {
      if (destroyed) return;
      renderList('error', messageOf(error));
    }
  }

  async function bootstrap(): Promise<void> {
    await loadRepositories();
    await loadWorkspaces();
  }

  void bootstrap();

  return {
    element: root,
    destroy() {
      destroyed = true;
    },
  };
}
