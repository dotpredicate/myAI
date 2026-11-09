import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, navigate, encodeKey, slugifyKey, plural, policyLabel, POLICIES, validateKey, validateRequired, reportFieldErrors, compactError, messageOf, type FieldControl, type MountOptions, type Route } from '../ui.js';
import { api } from '../api.js';
import type { BrowseEntry, BrowseResult, DeleteResponse, RepositoryConfig, RepositoryFiles, RepositoryList, SearchResponse, SyncResponse } from '../api.js';
import { Row, RowEnd, Stack, Grid, Page, Heading, Card, Hint, InlineHint, LineHint, FieldGroup, Badge, Notice, Button, LinkButton, Input, Select, Option, List, ListItem, Form, IconButton } from '../components.js';

export function mount(route: Route, _options: MountOptions = {}) {
  const root = Page({});

  // Index actions
  const updateButton = Button('Update index');
  const clearButton = Button('Clear index');
  const indexStatus = Hint();

  root.appendChild(el('header', {},
    Heading(1, 'Repositories'),
    Row(updateButton, clearButton),
  ));
  root.appendChild(indexStatus);

  // Add repository form
  const addToggle = Button('+ New repository', { variant: 'primary', small: true });
  const nameInput = Input({ name: 'display_name', required: true, autocomplete: 'off' });
  const pathInput = Input({ name: 'path', required: true, autocomplete: 'off', placeholder: '/home/you/code/project' });
  const keyInput = Input({ name: 'internal_name', required: true, autocomplete: 'off', placeholder: 'project' });
  const typeSelect = Select({ name: 'type' },
    Option('', 'Auto detect'),
    Option('plain', 'Plain folder'),
    Option('git', 'Git repository'),
  );
  const securitySelect = Select({ name: 'security' },
    POLICIES.map((policy) => Option(policy.value, policy.label)));
  const browseButton = Button('Browse');
  const browsePanel = el('div', { class: 'inset stack', hidden: true });
  const formError = Notice({ level: 'error' });
  const addSave = Button('Add repository', { variant: 'primary', submit: true });
  const addCancel = Button('Cancel');
  const nameField = labeledField('Name', nameInput);
  const pathField = labeledField('Path', pathInput);
  const keyField = labeledField('Key', keyInput, { hint: 'Letters, numbers, hyphens and underscores only.' });
  const typeField = labeledField('Type', typeSelect);
  const securityField = labeledField('Security', securitySelect, { hint: 'Highest permissions the repository allows.' });
  const addForm = Form({ novalidate: true, hidden: true, class: 'inset' },
    el('header', {}, Heading(3, 'New repository')),
    Grid({ columns: 2 }, nameField.wrapper, keyField.wrapper),
    FieldGroup(pathField.wrapper, Row(browseButton)),
    Grid({ columns: 2 }, typeField.wrapper, securityField.wrapper),
    browsePanel,
    formError,
    Row(addSave, addCancel),
  );

  const repoList = el('div', {});
  const listHeader = el('header', {}, Row(Heading(2, 'Repositories'), addToggle));
  const listCard = Card(listHeader, addForm, repoList);
  root.appendChild(listCard);

  interface ScreenState {
    repositories: RepositoryConfig[];
    browsePath: string;
    browseEntries: BrowseEntry[];
    browseToken: number;
    selectedKey: string | null;
    detailToken: number;
    actionsBusy: boolean;
  }

  const state: ScreenState = {
    repositories: [],
    browsePath: '',
    browseEntries: [],
    browseToken: 0,
    selectedKey: route.key || null,
    detailToken: 0,
    actionsBusy: false,
  };
  let destroyed = false;

  function openAddForm() {
    addForm.hidden = false;
  }

  function closeAddForm() {
    addForm.hidden = true;
    browsePanel.hidden = true;
  }

  function resetAddForm() {
    nameInput.value = '';
    pathInput.value = '';
    keyInput.value = '';
    typeSelect.value = '';
    securitySelect.value = 'read-only';
    nameField.setError('');
    pathField.setError('');
    keyField.setError('');
    formError.textContent = '';
    browsePanel.hidden = true;
  }

  addToggle.addEventListener('click', () => {
    openAddForm();
    nameInput.focus();
  });
  addCancel.addEventListener('click', () => {
    closeAddForm();
    resetAddForm();
  });

  // Browse
  function renderBreadcrumbs(path: string): void {
    clear(browsePanel);
    const crumbs = el('nav', { class: 'row' });
    const rootButton = Button('/', { variant: 'quiet', small: true });
    rootButton.addEventListener('click', () => loadBrowse('/'));
    crumbs.appendChild(rootButton);
    const parts = path.split('/').filter(Boolean);
    let accumulated = '';
    for (const part of parts) {
      accumulated += `/${part}`;
      const target = accumulated;
      crumbs.appendChild(document.createTextNode(' / '));
      const button = Button(part, { variant: 'quiet', small: true });
      button.addEventListener('click', () => loadBrowse(target));
      crumbs.appendChild(button);
    }
    browsePanel.appendChild(crumbs);

    const selectFolderButton = Button('Select this folder', { small: true });
    selectFolderButton.addEventListener('click', () => selectFolder(path));
    const current = Row(selectFolderButton);
    browsePanel.appendChild(current);

    if (state.browseEntries.length === 0) {
      browsePanel.appendChild(LineHint('This folder has no subfolders.'));
      return;
    }
    const list = el('ul', { class: 'browse-list' });
    for (const entry of state.browseEntries) {
      const openButton = el('button', { type: 'button', class: 'browse-open' }, `${entry.name}/`);
      const selectButton = el('button', { type: 'button', class: 'browse-select' }, 'Select');
      openButton.addEventListener('click', () => loadBrowse(entry.path));
      selectButton.addEventListener('click', () => selectFolder(entry.path));
      list.appendChild(el('li', { class: 'browse-entry' }, openButton, selectButton));
    }
    browsePanel.appendChild(list);
  }

  function selectFolder(path: string): void {
    pathInput.value = path;
    const folderName = path.replace(/\/+$/, '').split('/').pop() || 'repository';
    if (!nameInput.value.trim()) nameInput.value = folderName;
    if (!keyInput.value.trim()) keyInput.value = slugifyKey(folderName);
    browsePanel.hidden = true;
    pathInput.focus();
  }

  async function loadBrowse(path: string): Promise<void> {
    const token = ++state.browseToken;
    browsePanel.hidden = false;
    clear(browsePanel);
    browsePanel.appendChild(loadingState('Reading folders…'));
    try {
      const data = await api.get<BrowseResult>(`/api/browse?path=${encodeURIComponent(path)}&include=folders`);
      if (destroyed || token !== state.browseToken) return;
      state.browsePath = data.path;
      state.browseEntries = Array.isArray(data.entries) ? data.entries : [];
      renderBreadcrumbs(state.browsePath);
    } catch (error) {
      if (destroyed || token !== state.browseToken) return;
      clear(browsePanel);
      browsePanel.appendChild(compactError(messageOf(error) || 'Could not read this folder.'));
      const retry = Button('Retry', { small: true });
      retry.addEventListener('click', () => loadBrowse(path));
      browsePanel.appendChild(retry);
      const close = Button('Close', { variant: 'quiet', small: true });
      close.addEventListener('click', () => {
        browsePanel.hidden = true;
      });
      browsePanel.appendChild(close);
    }
  }

  browseButton.addEventListener('click', () => {
    const path = pathInput.value.trim() || '~';
    loadBrowse(path);
  });

  addForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.actionsBusy) return;
    nameField.setError('');
    pathField.setError('');
    keyField.setError('');
    formError.textContent = '';
    const displayName = nameInput.value.trim();
    const path = pathInput.value.trim();
    const internalName = keyInput.value.trim();
    const errors: FieldControl[] = [];
    validateRequired(nameField, displayName, 'Name is required.', errors);
    validateRequired(pathField, path, 'Path is required.', errors);
    validateKey(keyField, internalName, errors);
    if (reportFieldErrors(errors)) return;
    const payload: { display_name: string; internal_name: string; path: string; security: string; type?: string } = {
      display_name: displayName,
      internal_name: internalName,
      path,
      security: securitySelect.value,
    };
    if (typeSelect.value) payload.type = typeSelect.value;
    state.actionsBusy = true;
    addSave.disabled = true;
    addSave.replaceChildren(spinnerLabel('Adding…'));
    try {
      const saved = await api.post<RepositoryConfig>('/api/repositories', payload);
      notify('Repository added.', { level: 'success' });
      resetAddForm();
      closeAddForm();
      navigate(`#/repositories/${encodeKey(saved.internal_name)}`);
    } catch (error) {
      formError.textContent = messageOf(error) || 'Could not add the repository.';
      notify(messageOf(error) || 'Could not add the repository.', { level: 'error' });
    } finally {
      if (!destroyed) {
        state.actionsBusy = false;
        addSave.disabled = false;
        addSave.textContent = 'Add repository';
      }
    }
  });

  // Index actions
  updateButton.addEventListener('click', async () => {
    if (state.actionsBusy) return;
    state.actionsBusy = true;
    updateButton.disabled = true;
    indexStatus.textContent = '';
    try {
      await api.post<SyncResponse>('/api/sync');
      indexStatus.textContent = 'Indexing started in the background. The server does not report progress.';
      notify('Index update started.', { level: 'success' });
    } catch (error) {
      indexStatus.textContent = `Could not start indexing: ${messageOf(error)}`;
      notify(messageOf(error) || 'Could not start indexing.', { level: 'error' });
    } finally {
      state.actionsBusy = false;
      updateButton.disabled = false;
    }
  });

  clearButton.addEventListener('click', async () => {
    if (state.actionsBusy) return;
    const confirmed = await confirmAction({
      title: 'Clear the search index?',
      message: 'Search entries will be removed from the index. Files on disk are not touched.',
      confirmLabel: 'Clear index',
    });
    if (!confirmed) return;
    state.actionsBusy = true;
    clearButton.disabled = true;
    indexStatus.textContent = '';
    try {
      await api.del<DeleteResponse>('/api/search/index');
      indexStatus.textContent = 'Search index cleared. Files on disk remain untouched.';
      notify('Search index cleared.', { level: 'success' });
    } catch (error) {
      indexStatus.textContent = `Could not clear the index: ${messageOf(error)}`;
      notify(messageOf(error) || 'Could not clear the index.', { level: 'error' });
    } finally {
      state.actionsBusy = false;
      clearButton.disabled = false;
    }
  });

  function repositoryRow(repo: RepositoryConfig): HTMLLIElement {
    const selected = repo.internal_name === state.selectedKey;
    const open = () => navigate(selected
      ? '#/repositories'
      : `#/repositories/${encodeKey(repo.internal_name)}`);
    const main = el('button', { type: 'button', class: 'collection-row-main', onClick: open },
      el('span', { class: 'collection-row-title' }, repo.display_name || repo.internal_name),
      el('span', { class: 'collection-row-meta' },
        Badge(repo.repo_type),
        Badge(policyLabel(repo.security_policy), { accent: true }),
      ),
      el('span', { class: 'collection-row-path truncate' }, repo.path),
    );
    const deleteButton = IconButton('×', { title: 'Delete repository' });
    deleteButton.addEventListener('click', () => deleteRepository(repo));
    const row = ListItem(main, deleteButton);
    row.dataset.selected = selected ? 'true' : 'false';
    return row;
  }

  function renderRepoList(mode: 'ready' | 'loading' | 'error' = 'ready', message = ''): void {
    clear(repoList);
    if (mode === 'loading') {
      repoList.appendChild(loadingState('Loading repositories…'));
      return;
    }
    if (mode === 'error') {
      repoList.appendChild(errorState(message || 'Could not load repositories.', loadRepositories));
      return;
    }
    if (state.repositories.length === 0) {
      repoList.appendChild(emptyState(
        'No repositories',
        'Add a folder to make its files available for search.',
      ));
      return;
    }
    const list = List();
    for (const repo of state.repositories) {
      list.appendChild(repositoryRow(repo));
      if (repo.internal_name === state.selectedKey) {
        const detail = el('li', { class: 'list-detail' });
        const panel = el('div', { class: 'inset stack' });
        detail.appendChild(panel);
        list.appendChild(detail);
        renderDetails(repo, panel);
      }
    }
    repoList.appendChild(list);
    if (state.selectedKey && !state.repositories.some((repo) => repo.internal_name === state.selectedKey)) {
      repoList.appendChild(emptyState(
        'Repository not found',
        `There is no repository with the key “${state.selectedKey}”.`,
        LinkButton('Back to repositories', '#/repositories', { small: true }),
      ));
    }
  }

  async function deleteRepository(repo: RepositoryConfig): Promise<void> {
    const confirmed = await confirmAction({
      title: 'Delete repository?',
      message: `“${repo.display_name || repo.internal_name}” will be removed from myAI. Files on disk are not touched.`,
      confirmLabel: 'Delete repository',
    });
    if (!confirmed) return;
    try {
      await api.del<DeleteResponse>(`/api/repositories/${encodeKey(repo.internal_name)}`);
      notify('Repository removed. Files on disk remain untouched.', { level: 'success' });
      if (state.selectedKey === repo.internal_name) {
        navigate('#/repositories');
      } else {
        await loadRepositories();
      }
    } catch (error) {
      notify(messageOf(error) || 'Could not delete the repository.', { level: 'error' });
    }
  }

  async function loadRepositories(): Promise<void> {
    renderRepoList('loading');
    try {
      const data = await api.get<RepositoryList>('/api/repositories');
      if (destroyed) return;
      state.repositories = Array.isArray(data.repositories) ? data.repositories : [];
      renderRepoList();
      if (state.repositories.length === 0) openAddForm();
    } catch (error) {
      if (destroyed) return;
      renderRepoList('error', messageOf(error));
    }
  }

  function renderDetails(repo: RepositoryConfig, container: HTMLElement): void {
    const token = ++state.detailToken;
    clear(container);

    container.appendChild(el('header', {},
      el('div', {},
        Heading(2, repo.display_name || repo.internal_name),
        Row(Badge(repo.internal_name, { accent: true }), Badge(repo.repo_type), Badge(policyLabel(repo.security_policy))),
      ),
      IconButton('×', { title: 'Close repository', onClick: () => navigate('#/repositories') }),
    ));
    container.appendChild(LineHint(el('code', {}, repo.path)));

    const searchInput = Input({ type: 'search', placeholder: 'Describe what you are looking for…' });
    const searchField = labeledField('Semantic search', searchInput);
    const searchButton = Button('Search', { variant: 'primary', submit: true });
    const searchResults = el('div', {});
    const searchForm = Form({}, RowEnd(searchField.wrapper, searchButton));
    const searchSection = Stack(el('header', {}, Heading(3, 'Search')), searchForm, searchResults);
    searchResults.appendChild(emptyState(
      'Search this repository',
      'Results show the file, fragment number, distance and a short excerpt.',
    ));

    searchForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = searchInput.value.trim();
      if (!query) {
        searchField.setError('Enter search words.');
        searchInput.focus();
        return;
      }
      searchField.setError('');
      clear(searchResults);
      searchResults.appendChild(loadingState('Searching…'));
      searchButton.disabled = true;
      try {
        const data = await api.post<SearchResponse>('/api/search', {
          query,
          top_k: 5,
          scopes: [repo.internal_name],
        });
        if (destroyed || token !== state.detailToken) return;
        clear(searchResults);
        const results = Array.isArray(data.results) ? data.results : [];
        if (results.length === 0) {
          searchResults.appendChild(emptyState(
            'No matches',
            'Try different words, or update the index if this repository changed recently.',
          ));
          return;
        }
        for (const result of results) {
          searchResults.appendChild(el('div', { class: 'search-result' },
            el('div', { class: 'row row--between' },
              el('strong', {}, result.title),
              InlineHint(`distance ${result.cosine_distance.toFixed(3)}`),
            ),
            LineHint(result.snippet),
          ));
        }
      } catch (error) {
        if (destroyed || token !== state.detailToken) return;
        clear(searchResults);
        searchResults.appendChild(errorState(messageOf(error) || 'Search failed.', () => searchForm.requestSubmit()));
      } finally {
        if (!destroyed) searchButton.disabled = false;
      }
    });

    const filesSection = Stack();
    const filesHeader = el('header', {}, Heading(3, 'Files'));
    const filesContainer = el('div', {});
    const filterWrap = el('div', { hidden: true });
    const filterInput = Input({ type: 'search', placeholder: 'Filter files…' });
    const filterField = labeledField('Filter files', filterInput);
    filterWrap.appendChild(filterField.wrapper);
    filesSection.appendChild(filesHeader);
    filesSection.appendChild(filterWrap);
    filesSection.appendChild(filesContainer);
    container.appendChild(searchSection);
    container.appendChild(filesSection);

    let allFiles: string[] = [];
    let fileCount: HTMLSpanElement | null = null;

    function renderFiles(): void {
      const query = filterInput.value.trim().toLowerCase();
      const visible = query
        ? allFiles.filter((file) => file.toLowerCase().includes(query))
        : allFiles;
      clear(filesContainer);
      if (allFiles.length === 0) {
        filesContainer.appendChild(emptyState(
          'No files found',
          'The folder may be empty or may not be readable by the server.',
        ));
        return;
      }
      if (visible.length === 0) {
        filesContainer.appendChild(emptyState('No matching files', 'Change the filter to see more files.'));
        return;
      }
      const list = el('ul', { class: 'file-list' });
      for (const file of visible) list.appendChild(el('li', {}, file));
      filesContainer.appendChild(list);
      if (visible.length < allFiles.length) {
        filesContainer.appendChild(LineHint(`Showing ${visible.length} of ${plural(allFiles.length, 'file')}.`));
      }
    }

    filterInput.addEventListener('input', renderFiles);

    async function loadFiles(): Promise<void> {
      clear(filesContainer);
      filesContainer.appendChild(loadingState('Listing files…'));
      try {
        const data = await api.get<RepositoryFiles>(`/api/repositories/${encodeURIComponent(repo.internal_name)}/files`);
        if (destroyed || token !== state.detailToken) return;
        allFiles = Array.isArray(data.files) ? data.files : [];
        filterWrap.hidden = allFiles.length <= 20;
        fileCount?.remove();
        fileCount = InlineHint(plural(allFiles.length, 'file'));
        filesHeader.appendChild(fileCount);
        renderFiles();
      } catch (error) {
        if (destroyed || token !== state.detailToken) return;
        clear(filesContainer);
        filesContainer.appendChild(errorState(messageOf(error) || 'Could not list files.', loadFiles));
      }
    }

    loadFiles();
  }

  loadRepositories();

  return {
    element: root,
    destroy() {
      destroyed = true;
      state.browseToken += 1;
      state.detailToken += 1;
    },
  };
}
