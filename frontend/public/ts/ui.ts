import { clear, el, uid } from './dom.js';
import { RowRight, Stack, Heading, Paragraph, InlineHint, FieldError, FieldGroup, FieldLabel, Notice, Button, Input, Select, Option, InlineLoading } from './components.js';

export { el, clear, uid } from './dom.js';

export type FieldControl = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

export interface MountOptions {
  onConversationsChanged?: () => void;
}

export function loadingState(message = 'Loading…'): HTMLDivElement {
  return Notice({ tight: true }, InlineLoading(message));
}

export function errorState(message: string, retry?: (() => void) | null): HTMLDivElement {
  const box = Notice({ level: 'error', stack: true },
    Heading(3, 'Something went wrong'),
    Paragraph(message),
  );
  if (retry) box.appendChild(Button('Retry', { small: true, onClick: retry }));
  return box;
}

export function compactError(message: string): HTMLDivElement {
  return Notice({ level: 'error', tight: true }, Paragraph(message));
}

export function emptyState(title: string, message?: string | null, action?: Node | null): HTMLDivElement {
  const box = Notice({ level: 'state', stack: true }, Heading(3, title));
  if (message) box.appendChild(Paragraph(message));
  if (action) box.appendChild(action);
  return box;
}

export function spinnerLabel(label: string): HTMLSpanElement {
  return InlineLoading(label);
}

export interface StatusOptions {
  loading?: boolean;
  error?: boolean;
  retry?: (() => void) | null;
}

export function statusMessage(
  container: HTMLElement,
  message?: string | null,
  options: StatusOptions = {},
): void {
  clear(container);
  if (!message) return;
  if (options.loading) {
    container.appendChild(spinnerLabel(message));
    return;
  }
  container.appendChild(options.error ? FieldError(message) : InlineHint(message));
  if (options.retry) {
    container.appendChild(document.createTextNode(' '));
    container.appendChild(Button('Retry', { small: true, variant: 'quiet', onClick: options.retry }));
  }
}

export interface Field<T extends FieldControl = FieldControl> {
  wrapper: HTMLDivElement;
  control: T;
  errorEl: HTMLSpanElement;
  id: string;
  setError(message?: string | null): void;
}

export interface FieldOptions {
  id?: string;
  hint?: string;
  className?: string;
}

export function labeledField<T extends FieldControl>(
  labelText: string,
  control: T,
  options: FieldOptions = {},
): Field<T> {
  const id = control.id || options.id || uid('field');
  control.id = id;
  const errorEl = FieldError(undefined, { hidden: true });
  const hintEl = options.hint ? InlineHint(options.hint) : null;

  const wrapper = el('div', { class: `field ${options.className || ''}`.trim() },
    el('label', { class: 'field-label', for: id }, labelText),
    hintEl,
    control,
    errorEl,
  );

  function setError(message?: string | null): void {
    if (message) {
      errorEl.textContent = message;
      errorEl.hidden = false;
      control.dataset.invalid = 'true';
    } else {
      errorEl.textContent = '';
      errorEl.hidden = true;
      delete control.dataset.invalid;
    }
  }

  return { wrapper, control, errorEl, setError, id };
}

const KEY_PATTERN = /^[a-zA-Z0-9_-]+$/;

export function validateRequired(
  field: Field,
  value: string,
  message: string,
  errors: FieldControl[],
): void {
  if (value) return;
  field.setError(message);
  errors.push(field.control);
}

export function validateKey(field: Field, value: string, errors: FieldControl[]): void {
  if (!value) {
    field.setError('Key is required.');
    errors.push(field.control);
    return;
  }
  if (!KEY_PATTERN.test(value)) {
    field.setError('Use letters, numbers, hyphens and underscores only.');
    errors.push(field.control);
  }
}

export function reportFieldErrors(errors: FieldControl[]): boolean {
  if (errors.length === 0) return false;
  notify('Please fix the highlighted fields.', { level: 'error' });
  errors[0]?.focus();
  return true;
}

export interface ConfirmOptions {
  title?: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

export function confirmAction(options: ConfirmOptions = {}): Promise<boolean> {
  const title = options.title || 'Are you sure?';
  const message = options.message || '';
  const confirmLabel = options.confirmLabel || 'Confirm';
  const cancelLabel = options.cancelLabel || 'Cancel';
  const danger = options.danger !== false;

  const dialog = el('dialog', {});
  const cancel = Button(cancelLabel);
  const confirm = Button(confirmLabel, { variant: danger ? 'danger' : 'primary' });

  dialog.appendChild(Stack(
    Heading(2, title),
    message ? Paragraph(message) : null,
    RowRight(cancel, confirm),
  ));

  document.body.appendChild(dialog);

  return new Promise((resolve) => {
    let settled = false;
    function finish(value: boolean): void {
      if (settled) return;
      settled = true;
      resolve(value);
      dialog.close();
      dialog.remove();
    }
    cancel.addEventListener('click', () => finish(false));
    confirm.addEventListener('click', () => finish(true));
    dialog.addEventListener('cancel', (event) => {
      event.preventDefault();
      finish(false);
    });
    dialog.addEventListener('close', () => {
      if (!settled) finish(false);
    });
    dialog.showModal();
    cancel.focus();
  });
}

const MAX_TOASTS = 3;
const TIMEOUT_MS = 5000;
const NOTIFY_LEVELS = ['info', 'success', 'error'] as const;
export type NotifyLevel = (typeof NOTIFY_LEVELS)[number];
let toastContainer: HTMLElement | null = null;

export function initNotifications(element: HTMLElement | null): void {
  toastContainer = element;
}

export interface NotifyOptions {
  level?: NotifyLevel;
  timeout?: number;
}

export function notify(message: string, options: NotifyOptions = {}): void {
  if (!toastContainer) return;
  const level: NotifyLevel = options.level && NOTIFY_LEVELS.includes(options.level) ? options.level : 'info';
  const toast = document.createElement('div');
  toast.className = `toast toast--${level}`;

  const text = document.createElement('p');
  text.textContent = message;
  toast.appendChild(text);

  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'icon-btn';
  close.textContent = '×';
  toast.appendChild(close);

  let timer = window.setTimeout(remove, options.timeout ?? TIMEOUT_MS);
  function remove(): void {
    window.clearTimeout(timer);
    toast.remove();
  }
  close.addEventListener('click', remove);
  toast.addEventListener('mouseenter', () => window.clearTimeout(timer));
  toast.addEventListener('mouseleave', () => {
    timer = window.setTimeout(remove, 2500);
  });

  toastContainer.appendChild(toast);
  while (toastContainer.children.length > MAX_TOASTS) {
    toastContainer.firstElementChild?.remove();
  }
}

export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = value >= 100 || index === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[index]}`;
}

export function formatGigabytes(bytes: number | null | undefined): string {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes)) return '—';
  return (bytes / (1024 ** 3)).toFixed(2);
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPercent(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `${value.toFixed(1)}%`;
}

export function formatIdle(seconds: number | null | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—';
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, '0')}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, '0')}m`;
}

export function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null) return {};
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(value)) result[key] = Reflect.get(value, key);
  return result;
}

export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : '';
}

export function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export type ParamValue = string | number | boolean;

/** Copies only scalar values; used for untrusted config maps read from storage. */
export function asParamValues(value: unknown): Record<string, ParamValue> {
  const result: Record<string, ParamValue> = {};
  for (const [key, item] of Object.entries(asRecord(value))) {
    if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') result[key] = item;
  }
  return result;
}

export function asParamValueMap(value: Record<string, unknown>): Record<string, Record<string, ParamValue>> {
  const result: Record<string, Record<string, ParamValue>> = {};
  for (const [key, item] of Object.entries(asRecord(value))) result[key] = asParamValues(item);
  return result;
}

export interface ConversationRef {
  id?: number | string | null;
  title?: string | null;
}

export function conversationLabel(conversation?: ConversationRef | null): string {
  if (!conversation) return 'Conversation';
  return conversation.title || `Conversation #${conversation.id}`;
}

export function slugifyKey(name: string | number): string {
  const slug = String(name || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'repository';
}

export function prettyJson(text: string | null | undefined): string {
  if (!text) return '';
  const trimmed = text.trim();
  if (!trimmed) return '';
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return text;
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function parseJsonObject(text: string | null | undefined): Record<string, unknown> | null {
  if (!text?.trim()) return null;
  try {
    const parsed: unknown = JSON.parse(text);
    return isPlainObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function plural(count: number, singular: string, pluralForm?: string): string {
  return `${count} ${count === 1 ? singular : (pluralForm || `${singular}s`)}`;
}

const PREFIX = 'myai.';

export function readValue<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    const parsed: T = JSON.parse(raw);
    return parsed;
  } catch {
    return fallback;
  }
}

export function writeValue(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* Storage can be unavailable; preferences are best-effort. */
  }
}

export function readString(key: string, fallback = ''): string {
  const value = readValue<unknown>(key, fallback);
  return typeof value === 'string' ? value : fallback;
}

function decode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

export function encodeKey(value: string | number): string {
  return encodeURIComponent(String(value));
}

export type ScreenName = 'chat' | 'images' | 'inference' | 'agents' | 'repositories';

export interface Route {
  name: ScreenName;
  id?: number | null;
  tab?: string;
  mode?: string;
  key?: string | null;
  invalid?: boolean;
}

export function parseRoute(hash = window.location.hash): Route {
  let path = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!path.startsWith('/')) path = `/${path}`;
  const clean = path.split('?')[0] ?? '';
  const segments = clean.split('/').filter(Boolean).map(decode);
  const screen = segments[0] || 'chat';

  if (screen === 'chat') {
    const id = segments[1] && /^\d+$/.test(segments[1]) ? Number(segments[1]) : null;
    return { name: 'chat', id };
  }
  if (screen === 'images') return { name: 'images' };
  if (screen === 'inference') {
    const tab = segments[1];
    if (tab === 'system' || tab === 'managed' || tab === 'models') return { name: 'inference', tab };
    return { name: 'inference', tab: 'system' };
  }
  if (screen === 'agents') {
    if (!segments[1]) return { name: 'agents', mode: 'list' };
    if (segments[1] === 'new') return { name: 'agents', mode: 'new' };
    return { name: 'agents', mode: 'edit', key: segments[1] };
  }
  if (screen === 'repositories') {
    if (!segments[1]) return { name: 'repositories', key: null };
    return { name: 'repositories', key: segments[1] };
  }
  return { name: 'chat', id: null, invalid: true };
}

export function navigate(hash: string, { replace = false }: { replace?: boolean } = {}): void {
  if (window.location.hash === hash) return;
  if (replace) window.history.replaceState(null, '', hash);
  else window.location.hash = hash;
}

export function onRoute(handler: (route: Route) => void): void {
  window.addEventListener('hashchange', () => handler(parseRoute()));
}

export type PolicyValue = 'read-only' | 'privileged-write' | 'write';

export interface Policy {
  value: PolicyValue;
  label: string;
}

export const POLICIES: Policy[] = [
  { value: 'read-only', label: 'Read only' },
  { value: 'privileged-write', label: 'Privileged write' },
  { value: 'write', label: 'Write' },
];

const ORDER: string[] = POLICIES.map((policy) => policy.value);

export function asPolicy(value: string): PolicyValue | null {
  return value === 'read-only' || value === 'privileged-write' || value === 'write' ? value : null;
}

export function policyLabel(value?: string | null): string {
  return POLICIES.find((policy) => policy.value === value)?.label || value || '—';
}

export function allowedPolicies(basePolicy?: string | null): Policy[] {
  const baseIndex = ORDER.indexOf(basePolicy || '');
  if (baseIndex < 0) return POLICIES.slice();
  return POLICIES.filter((policy) => ORDER.indexOf(policy.value) <= baseIndex);
}

export interface ParamSpec {
  name: string;
  type: string;
  default?: ParamValue | null;
  description?: string | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
}

export type ParamValues = Record<string, ParamValue>;

function defaultValueLabel(param: ParamSpec): string {
  if (param.default === null || param.default === undefined || param.default === '') return 'Default';
  return `Default: ${String(param.default)}`;
}

function currentValue(values: ParamValues | null | undefined, param: ParamSpec): ParamValue | undefined {
  return values && Object.prototype.hasOwnProperty.call(values, param.name) ? values[param.name] : undefined;
}

function parseParamValue(param: ParamSpec, raw: string): ParamValue | undefined {
  if (raw === '') return undefined;
  if (param.type === 'int') {
    const number = Number.parseInt(raw, 10);
    return Number.isNaN(number) ? undefined : number;
  }
  if (param.type === 'float') {
    const number = Number.parseFloat(raw);
    return Number.isNaN(number) ? undefined : number;
  }
  if (param.type === 'bool') {
    if (raw === 'true') return true;
    if (raw === 'false') return false;
    return undefined;
  }
  return raw;
}

export interface ParamFieldOptions {
  idPrefix?: string;
  prefillDefaults?: boolean;
}

export function renderParamFields(
  container: HTMLElement,
  params: ParamSpec[],
  values: ParamValues | null | undefined,
  onChange: (name: string, value: ParamValue | undefined) => void,
  options: ParamFieldOptions = {},
): void {
  container.replaceChildren();
  const prefix = options.idPrefix || uid('param');
  const prefillDefaults = Boolean(options.prefillDefaults);

  for (const param of params) {
    const id = `${prefix}-${param.name}`;
    const labelText = param.name;
    const stored = currentValue(values, param);
    let control: FieldControl;

    if (param.type === 'bool') {
      const select = Select({ id });
      select.appendChild(Option('', defaultValueLabel(param)));
      select.appendChild(Option('true', 'true'));
      select.appendChild(Option('false', 'false'));
      select.value = stored === true ? 'true' : stored === false ? 'false' : '';
      select.addEventListener('change', () => {
        onChange(param.name, select.value === '' ? undefined : select.value === 'true');
      });
      control = select;
    } else {
      const type = param.type === 'int' || param.type === 'float' ? 'number' : 'text';
      const initial = stored !== undefined
        ? stored
        : (prefillDefaults && param.default !== null && param.default !== undefined ? param.default : '');
      const input = Input({
        id,
        type,
        value: String(initial),
        placeholder: defaultValueLabel(param),
      });
      if (param.min !== null && param.min !== undefined) input.min = String(param.min);
      if (param.max !== null && param.max !== undefined) input.max = String(param.max);
      if (param.step !== null && param.step !== undefined) input.step = String(param.step);
      input.addEventListener('input', () => {
        onChange(param.name, parseParamValue(param, input.value));
      });
      control = input;
    }

    container.appendChild(FieldGroup(
      FieldLabel(labelText, id),
      control,
      InlineHint(param.description || defaultValueLabel(param)),
    ));
  }
}

export function pickDefined(values?: ParamValues | null): ParamValues {
  const result: ParamValues = {};
  for (const [key, value] of Object.entries(values || {})) {
    if (value !== '') result[key] = value;
  }
  return result;
}

export function createCopyButton(getText: () => string, label = 'Copy'): HTMLButtonElement {
  const button = Button(label, { small: true, variant: 'quiet' });
  let timer: number | null = null;
  button.addEventListener('click', async (event) => {
    event.stopPropagation();
    event.preventDefault();
    try {
      await navigator.clipboard.writeText(getText());
      button.textContent = 'Copied!';
    } catch {
      button.textContent = 'Copy failed';
    }
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      if (button.isConnected) {
        button.textContent = label;
      }
    }, 2000);
  });
  return button;
}
