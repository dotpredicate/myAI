import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, formatBytes, formatIdle, formatNumber, formatPercent, compactError, messageOf, statusMessage as setStatus, type MountOptions, type Route } from '../ui.js';
import { api } from '../core/api.js';
import type { BenchmarkResult, CachedModelList, GpuBenchmark, ManagedServer, ModelInfo, ModelQuantInfo, ProviderCreateRequest, ProviderDetails, ProviderInfo, ProviderList, ProviderUpdateRequest, StopProcessResponse, SystemSnapshot, SystemStats } from '../core/api.js';
import { Row, Stack, Split, Grid, Page, Heading, Card, CardHeader, Hint, InlineHint, LineHint, FieldError, Badge, Button, LinkButton, Input, Form, Select, Option, Table, TableHead, TableRow, TableHeaderCell, TableCell } from '../components.js';

const TABS: { key: string; label: string }[] = [
  { key: 'system', label: 'System' },
  { key: 'providers', label: 'Providers' },
  { key: 'models', label: 'Models' },
];

function metric(label: string, value: string | number, percent: number | null = null): HTMLElement {
  const box = el('div', { class: 'inset' },
    Hint(label),
    el('div', { class: 'metric-value' }, value),
  );
  if (percent !== null) {
    box.appendChild(Row(el('progress', { max: '100', value: String(percent) }), InlineHint(formatPercent(percent))));
  }
  return box;
}

function sharePercent(used: number | null | undefined, total: number | null | undefined): number | null {
  if (typeof used !== 'number' || typeof total !== 'number' || !Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return null;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

function usageText(used: number | null | undefined, total: number | null | undefined, format: (value: number) => string): string {
  if (typeof used !== 'number' || typeof total !== 'number') return '—';
  return `${format(used)} / ${format(total)}`;
}

function errorWithRetry(container: HTMLElement, message: string, retry: (() => void) | null = null): void {
  clear(container);
  container.appendChild(compactError(message));
  if (!retry) return;
  container.appendChild(Button('Retry', { small: true, onClick: retry }));
}

function parseModels(value: string): string[] {
  return value.split(/[,\n]/).map((item) => item.trim()).filter(Boolean);
}

function kindLabel(kind: string | undefined): string {
  if (kind === 'openai_compatible') return 'OpenAI-compatible';
  return 'built-in';
}

export function mount(route: Route, _options: MountOptions = {}) {
  const root = Page({});
  const tab = route.tab ?? 'system';
  const tabBar = el('nav', { class: 'tabs' },
    TABS.map((item) => el('a', {
      href: `#/inference/${item.key}`,
      dataset: { active: item.key === tab ? 'true' : 'false' },
    }, item.label)),
  );
  const content = Stack();
  root.appendChild(Heading(1, 'Inference'));
  root.appendChild(tabBar);
  root.appendChild(content);

  let destroyed = false;
  let pollTimer: number | null = null;
  let snapshotInFlight = false;

  // System tab
  function mountSystem() {
    const resourcesCard = Card();
    const resourcesStatus = Hint();
    const resourcesBody = el('div', {});
    resourcesCard.appendChild(CardHeader('Resources'));
    resourcesCard.appendChild(LineHint('Live usage of the whole machine. CPU is shown in cores (1.00 is one fully busy core).'));
    resourcesCard.appendChild(resourcesStatus);
    resourcesCard.appendChild(resourcesBody);

    const managedCard = Card();
    const managedContainer = el('div', {});
    managedCard.appendChild(CardHeader('Managed processes'));
    managedCard.appendChild(LineHint('Model servers run on demand. Idle instances are stopped automatically after five minutes. CPU is per process, in cores; VRAM comes from ROCm.'));
    managedCard.appendChild(managedContainer);

    const benchmarkCard = Card();
    const tflopsInput = Input({ type: 'number', min: '1', step: '1', value: '4096', required: true });
    const bwInput = Input({ type: 'number', min: '1', step: '1', value: '8192', required: true });
    const tflopsField = labeledField('TFLOPS matrix size', tflopsInput);
    const bwField = labeledField('Bandwidth size', bwInput);
    const runButton = Button('Run benchmark', { variant: 'primary', submit: true });
    const benchmarkStatus = Hint();
    const benchmarkResults = el('div', { class: 'grid', hidden: true });
    const form = Form({ class: 'row row--end' }, tflopsField.wrapper, bwField.wrapper, runButton);
    benchmarkCard.appendChild(CardHeader('GPU benchmark'));
    benchmarkCard.appendChild(form);
    benchmarkCard.appendChild(benchmarkStatus);
    benchmarkCard.appendChild(benchmarkResults);

    content.appendChild(resourcesCard);
    content.appendChild(managedCard);
    content.appendChild(benchmarkCard);

    let lastSnapshot: SystemSnapshot | null = null;

    function renderResources(stats: SystemStats): void {
      clear(resourcesBody);
      const cpuPercent = sharePercent(stats.cpu.used_cores, stats.cpu.max_cores);
      const ramPercent = sharePercent(stats.ram.used_bytes, stats.ram.total_bytes);
      const vramPercent = sharePercent(stats.gpu.vram_used_bytes, stats.gpu.vram_total_bytes);
      const cpuText = typeof stats.cpu.used_cores === 'number'
        ? `${formatNumber(stats.cpu.used_cores, 2)} / ${formatNumber(stats.cpu.max_cores)} cores`
        : '—';
      resourcesBody.appendChild(Grid({},
        metric('CPU', cpuText, cpuPercent),
        metric('RAM', usageText(stats.ram.used_bytes, stats.ram.total_bytes, formatBytes), ramPercent),
        metric('VRAM', usageText(stats.gpu.vram_used_bytes, stats.gpu.vram_total_bytes, formatBytes), vramPercent),
        metric('GPU load', formatPercent(stats.gpu.utilization_percent)),
      ));
    }

    function renderServers(servers: ManagedServer[]): void {
      clear(managedContainer);
      if (!servers.length) {
        managedContainer.appendChild(emptyState(
          'No model servers are running',
          'One will start automatically the next time you send a prompt that needs it.',
        ));
        return;
      }
      const table = Table(TableHead(TableRow(TableHeaderCell('Model'), TableHeaderCell('Provider'), TableHeaderCell('Port'), TableHeaderCell('PID'), TableHeaderCell('Status'), TableHeaderCell('CPU'), TableHeaderCell('RAM'), TableHeaderCell('VRAM'), TableHeaderCell('Active'), TableHeaderCell('Idle'), TableHeaderCell('Actions'))));
      const body = el('tbody');
      for (const server of servers) {
        const kill = Button('Kill', { variant: 'danger', small: true });
        kill.addEventListener('click', async () => {
          const confirmed = await confirmAction({
            title: 'Stop model server?',
            message: `The ${server.provider} server for “${server.model}” on port ${server.port} will be stopped.`,
            confirmLabel: 'Kill process',
          });
          if (!confirmed) return;
          kill.disabled = true;
          try {
            await api.post<StopProcessResponse>(`/api/managed-processes/${encodeURIComponent(server.port)}/stop`);
            notify('Model server stopped.', { level: 'success' });
            await loadSnapshot();
          } catch (error) {
            kill.disabled = false;
            notify(messageOf(error) || 'Could not stop the process.', { level: 'error' });
          }
        });
        body.appendChild(TableRow(
          TableCell(server.model || '—'),
          TableCell(server.provider || '—'),
          TableCell(server.port),
          TableCell(server.pid === null ? '—' : String(server.pid)),
          TableCell(Badge(server.status || 'unknown')),
          TableCell(formatNumber(server.cpu_cores, 2)),
          TableCell(formatBytes(server.rss_bytes)),
          TableCell(formatBytes(server.vram_bytes)),
          TableCell(String(server.active_requests)),
          TableCell(formatIdle(server.idle_seconds)),
          TableCell(kill),
        ));
      }
      table.appendChild(body);
      managedContainer.appendChild(el('div', { class: 'table-wrap' }, table));
    }

    function renderSnapshot(snapshot: SystemSnapshot): void {
      renderResources(snapshot.resources);
      renderServers(snapshot.servers);
      lastSnapshot = snapshot;
    }

    async function loadSnapshot(initial = false): Promise<void> {
      if (snapshotInFlight) return;
      snapshotInFlight = true;
      if (initial && !lastSnapshot) {
        clear(resourcesBody);
        resourcesBody.appendChild(loadingState('Reading system resources…'));
        clear(managedContainer);
        managedContainer.appendChild(loadingState('Loading managed processes…'));
      }
      try {
        const data = await api.get<SystemSnapshot>('/api/system-stats');
        if (destroyed) return;
        renderSnapshot(data);
        clear(resourcesStatus);
      } catch (error) {
        if (destroyed) return;
        if (lastSnapshot) {
          renderSnapshot(lastSnapshot);
          setStatus(resourcesStatus, `Could not refresh system overview: ${messageOf(error)}`, {
            error: true,
            retry: () => loadSnapshot(),
          });
          resourcesStatus.appendChild(InlineHint('Showing the last known values.'));
        } else {
          clear(resourcesBody);
          resourcesBody.appendChild(errorState(messageOf(error) || 'Could not read system resources.', () => loadSnapshot(true)));
          errorWithRetry(managedContainer, messageOf(error) || 'Could not load managed processes.', () => loadSnapshot(true));
        }
      } finally {
        snapshotInFlight = false;
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (runButton.disabled) return;
      const tflopsSize = Number.parseInt(tflopsInput.value, 10);
      const bwSize = Number.parseInt(bwInput.value, 10);
      clear(benchmarkStatus);
      if (!Number.isFinite(tflopsSize) || tflopsSize <= 0 || !Number.isFinite(bwSize) || bwSize <= 0) {
        benchmarkStatus.appendChild(FieldError('Both benchmark sizes must be positive whole numbers.'));
        return;
      }
      runButton.disabled = true;
      runButton.replaceChildren(spinnerLabel('Running…'));
      benchmarkResults.hidden = true;
      clear(benchmarkResults);
      benchmarkStatus.replaceChildren(spinnerLabel('Running GPU benchmark…'));
      try {
        const data = await api.get<GpuBenchmark>(`/api/gpu-benchmark?tflops_size=${tflopsSize}&bw_size=${bwSize}`);
        if (destroyed) return;
        if (data.status !== 'success') throw new Error(data.message || 'The GPU benchmark failed.');
        benchmarkResults.replaceChildren(
          metric('FP32', `${formatNumber(data.tflops?.fp32?.tflops, 2)} TFLOPS`),
          metric('FP16', `${formatNumber(data.tflops?.fp16?.tflops, 2)} TFLOPS`),
          metric('INT8', `${formatNumber(data.tflops?.int8?.tflops, 2)} TOPS`),
          metric('Memory bandwidth', `${formatNumber(data.bandwidth?.gb_s, 1)} GB/s`),
        );
        benchmarkResults.hidden = false;
        clear(benchmarkStatus);
      } catch (error) {
        if (destroyed) return;
        errorWithRetry(benchmarkStatus, messageOf(error) || 'The GPU benchmark failed.', () => form.requestSubmit());      } finally {
        if (!destroyed) {
          runButton.disabled = false;
          runButton.textContent = 'Run benchmark';
        }
      }
    });

    loadSnapshot(true);
    return { refresh: () => loadSnapshot() };
  }

  // Providers tab
  function mountProviders() {
    const listCard = Card();
    const listStatus = Hint();
    const listContainer = el('div', {});
    listCard.appendChild(CardHeader('Providers'));
    listCard.appendChild(LineHint('Local llama.cpp and OpenAI-compatible endpoints. API keys are write-only.'));
    listCard.appendChild(listStatus);
    listCard.appendChild(listContainer);

    const formCard = Card();
    const formTitle = Heading(2, 'Add provider');
    const keyInput = Input({ type: 'text', required: true, placeholder: 'openrouter', autocomplete: 'off' });
    const displayNameInput = Input({ type: 'text', required: true, placeholder: 'OpenRouter' });
    const descriptionInput = Input({ type: 'text', placeholder: 'Optional' });
    const baseUrlInput = Input({ type: 'url', placeholder: 'https://openrouter.ai/api/v1' });
    const apiKeyInput = Input({ type: 'password', autocomplete: 'new-password', placeholder: 'Write-only' });
    const clearKeyInput = Input({ type: 'checkbox' });
    const modelsInput = Input({ type: 'text', placeholder: 'model-a, model-b' });
    const timeoutInput = Input({ type: 'number', min: '1', step: '1', value: '120', required: true });
    const reasoningSelect = Select({}, Option('false', 'No'), Option('true', 'Yes'));

    const keyField = labeledField('Key', keyInput, { hint: 'Letters, numbers, dots, hyphens and underscores.' });
    const displayNameField = labeledField('Display name', displayNameInput);
    const descriptionField = labeledField('Description', descriptionInput);
    const baseUrlField = labeledField('Base URL', baseUrlInput);
    const apiKeyField = labeledField('API key', apiKeyInput, { hint: 'Leave blank to keep the current key.' });
    const clearKeyField = labeledField('Clear API key', clearKeyInput);
    const modelsField = labeledField('Models', modelsInput, { hint: 'Comma-separated fallback list.' });
    const timeoutField = labeledField('Timeout (seconds)', timeoutInput);
    const reasoningField = labeledField('Send reasoning content', reasoningSelect);

    const submit = Button('Add provider', { variant: 'primary', submit: true });
    const cancel = Button('Cancel', { hidden: true });
    const formStatus = Hint();
    const form = Form({},
      Grid({ columns: 2 },
        keyField.wrapper,
        displayNameField.wrapper,
        descriptionField.wrapper,
        baseUrlField.wrapper,
        apiKeyField.wrapper,
        clearKeyField.wrapper,
        modelsField.wrapper,
        timeoutField.wrapper,
        reasoningField.wrapper,
      ),
      Row(submit, cancel),
      formStatus,
    );
    formCard.appendChild(formTitle);
    formCard.appendChild(form);
    content.appendChild(listCard);
    content.appendChild(formCard);

    let editingKey: string | null = null;

    const fields = [
      keyField,
      displayNameField,
      descriptionField,
      baseUrlField,
      apiKeyField,
      clearKeyField,
      modelsField,
      timeoutField,
      reasoningField,
    ];

    function resetForm(): void {
      editingKey = null;
      formTitle.textContent = 'Add provider';
      submit.textContent = 'Add provider';
      cancel.hidden = true;
      keyInput.disabled = false;
      keyInput.value = '';
      displayNameInput.value = '';
      descriptionInput.value = '';
      baseUrlInput.value = '';
      apiKeyInput.value = '';
      clearKeyInput.checked = false;
      modelsInput.value = '';
      timeoutInput.value = '120';
      reasoningSelect.value = 'false';
      for (const field of fields) field.setError('');
      setStatus(formStatus, '');
    }

    function setEditing(provider: ProviderDetails): void {
      editingKey = provider.key;
      formTitle.textContent = `Edit ${provider.display_name}`;
      submit.textContent = 'Save changes';
      cancel.hidden = false;
      keyInput.value = provider.key;
      keyInput.disabled = true;
      displayNameInput.value = provider.display_name;
      descriptionInput.value = provider.description;
      baseUrlInput.value = provider.base_url;
      apiKeyInput.value = '';
      clearKeyInput.checked = false;
      modelsInput.value = provider.models.join(', ');
      timeoutInput.value = String(provider.timeout_seconds);
      reasoningSelect.value = provider.include_reasoning_content ? 'true' : 'false';
      for (const field of fields) field.setError('');
      setStatus(formStatus, '');
      displayNameInput.focus();
    }

    function renderProviders(items: ProviderInfo[]): void {
      clear(listContainer);
      if (items.length === 0) {
        listContainer.appendChild(emptyState('No providers', 'Add an OpenAI-compatible endpoint below.'));
        return;
      }
      const table = Table(
        TableHead(TableRow(
          TableHeaderCell('Name'),
          TableHeaderCell('Key'),
          TableHeaderCell('Kind'),
          TableHeaderCell('API key'),
          TableHeaderCell('Actions'),
        )),
      );
      const body = el('tbody');
      for (const provider of items) {
        const builtin = provider.kind === 'builtin';
        const edit = Button('Edit', { small: true, disabled: builtin });
        edit.addEventListener('click', () => { void beginEdit(provider.key); });
        const remove = Button('Delete', { variant: 'danger', small: true, disabled: builtin });
        remove.addEventListener('click', () => { void removeProvider(provider); });
        body.appendChild(TableRow(
          TableCell(provider.display_name || provider.key),
          TableCell(el('code', {}, provider.key)),
          TableCell(Badge(kindLabel(provider.kind))),
          TableCell(provider.has_api_key ? 'set' : 'none'),
          TableCell(Row(edit, remove)),
        ));
      }
      table.appendChild(body);
      listContainer.appendChild(el('div', { class: 'table-wrap' }, table));
    }

    async function loadProviders(): Promise<void> {
      clear(listContainer);
      listContainer.appendChild(loadingState('Loading providers…'));
      try {
        const data = await api.get<ProviderList>('/api/providers');
        if (destroyed) return;
        renderProviders(Array.isArray(data.providers) ? data.providers : []);
        clear(listStatus);
      } catch (error) {
        if (destroyed) return;
        clear(listContainer);
        listContainer.appendChild(errorState(messageOf(error) || 'Could not load providers.', () => { void loadProviders(); }));
      }
    }

    async function beginEdit(key: string): Promise<void> {
      try {
        const provider = await api.get<ProviderDetails>(`/api/providers/${encodeURIComponent(key)}`);
        if (!destroyed) setEditing(provider);
      } catch (error) {
        if (!destroyed) notify(messageOf(error) || 'Could not load the provider.', { level: 'error' });
      }
    }

    async function removeProvider(provider: ProviderInfo): Promise<void> {
      const confirmed = await confirmAction({
        title: 'Delete provider?',
        message: `“${provider.display_name || provider.key}” will be removed. Agents using it will stop working.`,
        confirmLabel: 'Delete provider',
      });
      if (!confirmed) return;
      try {
        await api.del<{ status: string; key: string }>(`/api/providers/${encodeURIComponent(provider.key)}`);
        notify('Provider deleted.', { level: 'success' });
        if (editingKey === provider.key) resetForm();
        await loadProviders();
      } catch (error) {
        notify(messageOf(error) || 'Could not delete the provider.', { level: 'error' });
      }
    }

    cancel.addEventListener('click', resetForm);

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (submit.disabled) return;
      for (const field of fields) field.setError('');
      setStatus(formStatus, '');

      const key = keyInput.value.trim();
      const displayName = displayNameInput.value.trim();
      const description = descriptionInput.value.trim();
      const baseUrl = baseUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      const models = parseModels(modelsInput.value);
      const timeout = Number.parseInt(timeoutInput.value, 10);
      const includeReasoning = reasoningSelect.value === 'true';

      if (!displayName) {
        displayNameField.setError('Display name is required.');
        return;
      }
      if (!editingKey && !/^[a-zA-Z0-9._-]+$/.test(key)) {
        keyField.setError('Use letters, numbers, dots, hyphens and underscores.');
        return;
      }
      if (!/^https?:\/\/.+/.test(baseUrl)) {
        baseUrlField.setError('Enter an http:// or https:// URL.');
        return;
      }
      if (!Number.isFinite(timeout) || timeout <= 0) {
        timeoutField.setError('Enter a positive number of seconds.');
        return;
      }

      submit.disabled = true;
      try {
        if (editingKey) {
          const payload: ProviderUpdateRequest = {
            display_name: displayName,
            description,
            base_url: baseUrl,
            models,
            timeout_seconds: timeout,
            include_reasoning_content: includeReasoning,
          };
          if (apiKey) payload.api_key = apiKey;
          else if (clearKeyInput.checked) payload.api_key = null;
          await api.patch<ProviderDetails>(`/api/providers/${encodeURIComponent(editingKey)}`, payload);
          notify('Provider updated.', { level: 'success' });
        } else {
          const payload: ProviderCreateRequest = {
            key,
            display_name: displayName,
            description,
            base_url: baseUrl,
            api_key: apiKey || null,
            models,
            timeout_seconds: timeout,
            headers: {},
            extra_body: {},
            include_reasoning_content: includeReasoning,
          };
          await api.post<ProviderDetails>('/api/providers', payload);
          notify('Provider added.', { level: 'success' });
        }
        resetForm();
        await loadProviders();
      } catch (error) {
        if (!destroyed) setStatus(formStatus, messageOf(error) || 'Could not save the provider.', { error: true });
      } finally {
        if (!destroyed) submit.disabled = false;
      }
    });

    loadProviders();
    return { refresh: loadProviders };
  }

  // Models tab
  function mountModels() {
    const layout = Split();
    const left = Card();
    const right = Card();

    const modelInput = Input({
      type: 'text',
      placeholder: 'org/model or org/model:Q4_K_M',
      autocomplete: 'off',
      required: true,
    });
    const modelField = labeledField('HuggingFace model', modelInput, {
      hint: 'Enter a repository id, optionally with a quant tag.',
    });
    const loadButton = Button('Load model', { variant: 'primary', submit: true });
    const cachedList = el('div', {});

    const searchForm = Form({ novalidate: true }, modelField.wrapper, Row(loadButton));
    left.appendChild(CardHeader('Models'));
    left.appendChild(searchForm);
    left.appendChild(el('header', {},
      Heading(3, 'Cached models'),
      LinkButton('Find GGUF ↗', 'https://huggingface.co/models?library=gguf', { variant: 'quiet', small: true, target: '_blank', rel: 'noopener noreferrer' }),
    ));
    left.appendChild(cachedList);

    const details = Stack();
    right.appendChild(details);
    layout.appendChild(left);
    layout.appendChild(right);
    content.appendChild(layout);

    let selectedModel = '';
    let benchmarkRunning = false;
    interface QuantRow {
      result: HTMLTableCellElement;
      button: HTMLButtonElement;
      quant: ModelQuantInfo;
    }

    let quantRows = new Map<string, QuantRow>();
    let allButton: HTMLButtonElement | null = null;
    let tableWrap: HTMLElement | null = null;
    let busyLine: HTMLDivElement | null = null;

    function showDetailsPlaceholder(): void {
      clear(details);
      details.appendChild(emptyState(
        'Choose a model',
        'Load a HuggingFace model or pick one from the cache to see its quantization variants.',
      ));
    }

    async function loadCached(): Promise<void> {
      clear(cachedList);
      cachedList.appendChild(loadingState('Looking for cached models…'));
      try {
        const data = await api.get<CachedModelList>('/api/cached-models');
        if (destroyed) return;
        const models = Array.isArray(data.models) ? data.models : [];
        clear(cachedList);
        if (models.length === 0) {
          cachedList.appendChild(LineHint('No cached GGUF models yet. Start one with Load model.'));
          return;
        }
        const list = el('div', { class: 'list' });
        for (const model of models) {
          const item = el('button', { type: 'button', class: 'list-item' },
            el('span', {}, model),
          );
          item.addEventListener('click', () => {
            modelInput.value = model;
            loadModelInfo(model);
          });
          list.appendChild(item);
        }
        cachedList.appendChild(list);
      } catch (error) {
        if (destroyed) return;
        clear(cachedList);
        cachedList.appendChild(compactError(messageOf(error) || 'Could not load cached models.'));
        cachedList.appendChild(Button('Retry', { small: true, onClick: loadCached }));
      }
    }

    function setResult(row: QuantRow, result: BenchmarkResult | null): void {
      const cell = row.result;
      clear(cell);
      if (!result) {
        cell.appendChild(InlineHint('—'));
        return;
      }
      if (result.error) {
        cell.appendChild(FieldError(result.error));
        return;
      }
      cell.appendChild(Hint(`${formatNumber(result.tps, 1)} tok/s`));
      cell.appendChild(Hint(`${formatNumber(result.pp_tps, 1)} prompt tok/s`));
    }

    function renderInfo(data: ModelInfo): void {
      clear(details);
      quantRows = new Map();
      allButton = null;
      tableWrap = null;
      busyLine = null;
      if (data.error) {
        details.appendChild(errorState(data.error, () => loadModelInfo(selectedModel)));
        return;
      }
      const quants = Array.isArray(data.quants) ? data.quants : [];
      const header = el('header', {},
        el('div', {},
          Heading(2, data.model_id || selectedModel),
          LineHint(`${quants.length} quantization${quants.length === 1 ? '' : 's'}`),
        ),
      );
      allButton = Button('Benchmark all', { variant: 'primary', small: true });
      allButton.disabled = quants.length === 0;
      allButton.addEventListener('click', () => runBenchmarks([...quantRows.values()].map((row) => row.quant)));
      header.appendChild(allButton);
      details.appendChild(header);

      if (quants.length === 0) {
        details.appendChild(emptyState('No quantization variants', 'This model did not return any GGUF variants.'));
        return;
      }

      const table = Table(TableHead(TableRow(TableHeaderCell('Quant'), TableHeaderCell('Size'), TableHeaderCell('Parameters'), TableHeaderCell('Context'), TableHeaderCell('Architecture'), TableHeaderCell('Benchmark'), TableHeaderCell('Result'))));
      const body = el('tbody');
      for (const quant of quants) {
        const resultCell = TableCell();
        const benchButton = Button('Benchmark', { small: true });
        benchButton.disabled = Boolean(quant.error);
        benchButton.addEventListener('click', () => runBenchmarks([quant]));
        const row = { result: resultCell, button: benchButton, quant };
        quantRows.set(quant.quant ?? '', row);
        setResult(row, null);
        body.appendChild(TableRow(TableCell(quant.quant || '—'), TableCell(quant.error ? FieldError(quant.error) : formatBytes(quant.size_bytes)), TableCell(quant.parameters ? formatNumber(quant.parameters) : '—'), TableCell(quant.context_length ? formatNumber(quant.context_length) : '—'), TableCell(quant.architecture || '—'), TableCell(benchButton), resultCell));
      }
      table.appendChild(body);
      tableWrap = el('div', { class: 'table-wrap' }, table);
      details.appendChild(tableWrap);
      details.appendChild(LineHint('Results show generation speed and prompt prefill speed. Each variant is tested independently.'));
    }

    async function loadModelInfo(model: string): Promise<void> {
      selectedModel = model;
      clear(details);
      details.appendChild(loadingState('Reading model metadata…'));
      try {
        const data = await api.get<ModelInfo>(`/models/${encodeURIComponent(model)}/info`);
        if (destroyed || selectedModel !== model) return;
        renderInfo(data);
      } catch (error) {
        if (destroyed || selectedModel !== model) return;
        clear(details);
        details.appendChild(errorState(messageOf(error) || 'Could not load model information.', () => loadModelInfo(model)));
      }
    }

    function setBenchmarkBusy(busy: boolean, message = ''): void {
      benchmarkRunning = busy;
      for (const row of quantRows.values()) {
        row.button.disabled = busy || Boolean(row.quant.error);
      }
      if (allButton) allButton.disabled = busy || quantRows.size === 0;
      busyLine?.remove();
      busyLine = null;
      if (message) {
        busyLine = Hint(spinnerLabel(message));
        details.insertBefore(busyLine, tableWrap);
      }
    }

    async function runBenchmarks(quants: ModelQuantInfo[]): Promise<void> {
      if (benchmarkRunning || !selectedModel) return;
      const targets = quants.filter((quant) => !quant.error);
      if (targets.length === 0) return;
      const query = targets.map((quant) => `quant=${encodeURIComponent(quant.quant ?? '')}`).join('&');
      const message = targets.length === 1
        ? `Benchmarking ${targets[0]?.quant ?? 'variant'}…`
        : `Benchmarking ${targets.length} variants…`;
      setBenchmarkBusy(true, message);
      for (const quant of targets) {
        const row = quantRows.get(quant.quant ?? '');
        if (!row) continue;
        clear(row.result);
        row.result.appendChild(spinnerLabel('Running…'));
      }
      try {
        const data = await api.get<BenchmarkResult[]>(`/models/${encodeURIComponent(selectedModel)}/benchmark?${query}`);
        if (destroyed) return;
        const list = Array.isArray(data) ? data : [data];
        for (const quant of targets) {
          const row = quantRows.get(quant.quant ?? '');
          if (row) setResult(row, list.find((item) => item.quant === quant.quant) || { error: 'No benchmark result returned.' });
        }
      } catch (error) {
        if (destroyed) return;
        for (const quant of targets) {
          const row = quantRows.get(quant.quant ?? '');
          if (row) setResult(row, { error: messageOf(error) || 'Benchmark failed.' });
        }
      } finally {
        if (!destroyed) setBenchmarkBusy(false);
      }
    }

    searchForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const model = modelInput.value.trim();
      if (!model) {
        modelField.setError('Enter a model name.');
        modelInput.focus();
        return;
      }
      modelField.setError('');
      loadModelInfo(model);
    });

    showDetailsPlaceholder();
    loadCached();
    return { refresh: null };
  }

  let activeTab: { refresh?: (() => void) | null } | null = null;
  if (tab === 'system') activeTab = mountSystem();
  else if (tab === 'providers') activeTab = mountProviders();
  else activeTab = mountModels();

  function refreshActive(): void {
    if (document.visibilityState !== 'visible') return;
    activeTab?.refresh?.();
  }

  function onVisibility(): void {
    if (document.visibilityState === 'visible') refreshActive();
  }
  document.addEventListener('visibilitychange', onVisibility);

  if (activeTab.refresh) {
    pollTimer = window.setInterval(refreshActive, 5000);
  }

  return {
    element: root,
    destroy() {
      destroyed = true;
      if (pollTimer) window.clearInterval(pollTimer);
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}
