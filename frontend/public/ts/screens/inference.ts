import { el, clear, loadingState, errorState, emptyState, spinnerLabel, labeledField, notify, confirmAction, formatBytes, formatGigabytes, formatIdle, formatNumber, formatPercent, compactError, messageOf, statusMessage as setStatus, type MountOptions, type Route } from '../ui.js';
import { api } from '../api.js';
import type { BenchmarkResult, CachedModelList, GpuBenchmark, GpuStats, ManagedProcessList, ManagedServer, ModelInfo, ModelQuantInfo, StopProcessResponse } from '../api.js';
import { Row, Stack, Split, Grid, Page, Heading, Card, CardHeader, Hint, InlineHint, LineHint, FieldError, Badge, Button, LinkButton, Input, Form, Table, TableHead, TableRow, TableHeaderCell, TableCell } from '../components.js';

const TABS: { key: string; label: string }[] = [
  { key: 'system', label: 'System' },
  { key: 'managed', label: 'Managed processes' },
  { key: 'models', label: 'Models' },
];

function metric(label: string, value: string | number, options: { progress?: number; percent?: number } = {}): HTMLElement {
  const box = el('div', { class: 'inset' },
    Hint(label),
    el('div', { class: 'metric-value' }, value),
  );
  if (options.progress !== undefined) {
    box.appendChild(Row(el('progress', { max: '100', value: String(options.progress) }), InlineHint(formatPercent(options.percent))));
  }
  return box;
}

function benchmarkError(container: HTMLElement, message: string, retry?: (() => void) | null): void {
  clear(container);
  container.appendChild(compactError(message));
  if (!retry) return;
  container.appendChild(Button('Retry', { small: true, onClick: retry }));
}

export function mount(route: Route, _options: MountOptions = {}) {
  const root = Page({});
  const tab = route.tab;
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
  let gpuInFlight = false;
  let managedInFlight = false;

  // System tab
  function mountSystem() {
    const gpuCard = Card();
    const gpuHeader = CardHeader('GPU');
    const gpuStatus = Hint();
    const gpuBody = el('div', {});
    gpuCard.appendChild(gpuHeader);
    gpuCard.appendChild(gpuStatus);
    gpuCard.appendChild(gpuBody);

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

    content.appendChild(gpuCard);
    content.appendChild(benchmarkCard);

    let lastStats: GpuStats | null = null;

    function renderGpu(stats: GpuStats): void {
      lastStats = stats;
      clear(gpuBody);
      const used = Math.max(0, stats.total - stats.free);
      const percent = stats.total > 0 ? (used / stats.total) * 100 : 0;
      gpuBody.appendChild(Grid({}, metric('Free', `${formatGigabytes(stats.free)} GB`), metric('Total', `${formatGigabytes(stats.total)} GB`), metric('Used', `${formatGigabytes(used)} GB`), metric('Usage', formatPercent(percent), { progress: percent, percent })));
    }

    async function loadGpu(initial = false): Promise<void> {
      if (gpuInFlight) return;
      gpuInFlight = true;
      if (initial && !lastStats) {
        clear(gpuBody);
        gpuBody.appendChild(loadingState('Reading GPU memory…'));
      }
      try {
        const stats = await api.get<GpuStats>('/api/gpu-stats');
        if (destroyed) return;
        if (typeof stats.free !== 'number' || typeof stats.total !== 'number') {
          throw new Error('The GPU returned an unexpected response.');
        }
        renderGpu({ free: stats.free, total: stats.total });
        clear(gpuStatus);
      } catch (error) {
        if (destroyed) return;
        if (lastStats) {
          renderGpu(lastStats);
          setStatus(gpuStatus, `Could not refresh GPU memory: ${messageOf(error)}`, {
            error: true,
            retry: () => loadGpu(),
          });
          gpuStatus.appendChild(InlineHint('Showing the last known values.'));
        } else {
          clear(gpuBody);
          gpuBody.appendChild(errorState(messageOf(error) || 'Could not read GPU memory.', () => loadGpu(true)));
        }
      } finally {
        gpuInFlight = false;
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
        benchmarkError(benchmarkStatus, messageOf(error) || 'The GPU benchmark failed.', () => form.requestSubmit());
      } finally {
        if (!destroyed) {
          runButton.disabled = false;
          runButton.textContent = 'Run benchmark';
        }
      }
    });

    loadGpu(true);
    return { refresh: () => loadGpu() };
  }

  // Managed processes tab
  function mountManaged() {
    const card = Card();
    const status = Hint();
    const tableContainer = el('div', {});
    card.appendChild(CardHeader('Managed processes'));
    card.appendChild(LineHint('Model servers run on demand. Idle instances are stopped automatically after five minutes.'));
    card.appendChild(status);
    card.appendChild(tableContainer);
    content.appendChild(card);

    let lastServers: ManagedServer[] | null = null;

    function renderTable(servers: ManagedServer[]): void {
      clear(tableContainer);
      if (!servers.length) {
        tableContainer.appendChild(emptyState(
          'No model servers are running',
          'One will start automatically the next time you send a prompt that needs it.',
        ));
        return;
      }
      const table = Table(TableHead(TableRow(TableHeaderCell('Model'), TableHeaderCell('Provider'), TableHeaderCell('Port'), TableHeaderCell('PID'), TableHeaderCell('Status'), TableHeaderCell('Active'), TableHeaderCell('Idle'), TableHeaderCell('Actions'))));
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
            await loadManaged();
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
          TableCell(String(server.active_requests)),
          TableCell(formatIdle(server.idle_seconds)),
          TableCell(kill),
        ));
      }
      table.appendChild(body);
      tableContainer.appendChild(el('div', { class: 'table-wrap' }, table));
    }

    async function loadManaged(): Promise<void> {
      if (managedInFlight) return;
      managedInFlight = true;
      if (lastServers === null) {
        clear(tableContainer);
        tableContainer.appendChild(loadingState('Loading managed processes…'));
      }
      try {
        const data = await api.get<ManagedProcessList>('/api/managed-processes/list');
        if (destroyed) return;
        lastServers = Array.isArray(data.servers) ? data.servers : [];
        renderTable(lastServers);
        clear(status);
      } catch (error) {
        if (destroyed) return;
        if (lastServers) {
          setStatus(status, `Could not refresh processes: ${messageOf(error)}`, {
            error: true,
            retry: () => loadManaged(),
          });
        } else {
          clear(tableContainer);
          tableContainer.appendChild(errorState(messageOf(error) || 'Could not load managed processes.', () => loadManaged()));
        }
      } finally {
        managedInFlight = false;
      }
    }

    loadManaged();
    return { refresh: () => loadManaged() };
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
  else if (tab === 'managed') activeTab = mountManaged();
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
