// API transport plus the wire types produced by the Python back-end.
// Shapes mirror backend/domain.py, backend/agents.py, backend/workspaces.py,
// backend/conversation.py, backend/repositories.py, backend/images.py,
// backend/inference/ and backend/system.py; keep them in sync.

export type ParamValue = string | number | boolean;

/* ----------------------------------------------------------------- domain */

export interface RepositoryConfig {
  id: number;
  display_name: string;
  internal_name: string;
  repo_type: 'plain' | 'git';
  path: string;
}

export interface AgentConfig {
  id: number;
  display_name: string;
  internal_name: string;
  description: string;
  instructions: string | null;
  provider_key: string | null;
  model_id: string | null;
  inference_config: Record<string, ParamValue>;
  spawnable_agents: string[];
}

export interface CreateAgentRequest {
  display_name: string;
  internal_name: string;
  description?: string;
  instructions?: string | null;
  provider_key?: string | null;
  model_id?: string | null;
  inference_config?: Record<string, ParamValue>;
  spawnable_agents?: string[];
}

export interface UpdateAgentRequest {
  display_name?: string;
  description?: string;
  instructions?: string | null;
  provider_key?: string | null;
  model_id?: string | null;
  inference_config?: Record<string, ParamValue>;
  spawnable_agents?: string[];
}

export interface WorkspaceSource {
  repository_id: number;
  internal_name: string;
  display_name: string;
  repo_type: 'plain' | 'git';
  path: string;
}

export interface WorkspaceConfig {
  id: number;
  name: string;
  created_at: string;
  sources: WorkspaceSource[];
}

export interface WorkspaceList {
  workspaces: WorkspaceConfig[];
}

export interface UpdateWorkspaceRequest {
  name: string;
}

export interface AddWorkspaceSourceRequest {
  repository_id: number;
}

export interface ResolveWorkspaceResponse {
  workspace: WorkspaceConfig;
  created: boolean;
  repository_created: boolean;
  sync_started: boolean;
}

export type ConversationElement =
  | { type: 'message'; author: 'user' | 'assistant'; content: string }
  | { type: 'thinking'; content: string }
  | { type: 'tool_call'; name: string; parameters: string; result: string }
  | { type: 'approval_request'; original_message_id: number }
  | { type: 'tool_call_result'; original_message_id: number; result: string }
  | {
      type: 'tool_call_decision';
      original_message_id: number;
      decision: 'approve' | 'reject';
      comment: string | null;
    };

export interface StoredMessageRecord {
  id: number;
  conversation_id: number;
  sequence_id: number;
  role: string;
  element: ConversationElement;
  created_at: string;
}

/* ----------------------------------------------------------- conversations */

export interface ConversationSummary {
  id: number;
  title: string | null;
  created_at: string;
}

export interface ConversationDetails extends ConversationSummary {
  blocking_message_id: number | null;
  workspace_id: number | null;
  workspace_name: string | null;
  provider_key: string | null;
  model_id: string | null;
  agent_id: string | null;
  inference_config: Record<string, ParamValue>;
  messages: StoredMessageRecord[];
}

export interface CreateConversationRequest {
  workspace_id: number;
  agent_id?: string | null;
}

export type StreamEvent =
  | { type: 'message'; content: string; role: 'user' | 'assistant' }
  | { type: 'thinking'; content: string }
  | { type: 'finalized'; id: number; sequence_id: number }
  | { type: 'tool_call'; name: string; parameters: string; result: string }
  | { type: 'approval_request'; original_message_id: number }
  | {
      type: 'tool_call_decision';
      original_message_id: number;
      decision: 'approve' | 'reject';
      comment: string | null;
    }
  | { type: 'tool_call_result'; original_message_id: number; result: string }
  | { type: 'generation_error'; message: string };

/* ---------------------------------------------------------- providers */

export interface ProviderInfo {
  key: string;
  display_name: string;
  description: string;
  kind?: string;
  has_api_key?: boolean;
}

export interface ProviderList {
  providers: ProviderInfo[];
}

export interface ProviderDetails {
  key: string;
  display_name: string;
  description: string;
  base_url: string;
  has_api_key: boolean;
  models: string[];
  timeout_seconds: number;
  extra_body: Record<string, unknown>;
  include_reasoning_content: boolean;
}

export interface ProviderCreateRequest {
  key: string;
  display_name: string;
  description: string;
  base_url: string;
  api_key: string | null;
  models: string[];
  timeout_seconds: number;
  headers: Record<string, string>;
  extra_body: Record<string, unknown>;
  include_reasoning_content: boolean;
}

export interface ProviderUpdateRequest {
  display_name?: string;
  description?: string | null;
  base_url?: string;
  api_key?: string | null;
  models?: string[] | null;
  timeout_seconds?: number;
  include_reasoning_content?: boolean;
}

export interface ProviderModel {
  id: string;
  name: string;
}

export interface ProviderModelList {
  models: ProviderModel[];
}

export interface InferenceParam {
  name: string;
  type: 'int' | 'float' | 'bool' | 'str' | 'file';
  default: ParamValue | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  description?: string | null;
}

export type InferenceParamList = InferenceParam[];

export interface CachedModelList {
  models: string[];
}

export interface ModelQuantInfo {
  quant?: string;
  filename?: string;
  size_bytes?: number;
  parameters?: number;
  architecture?: string;
  context_length?: number;
  error?: string;
}

export interface ModelInfo {
  model_id: string;
  quants: ModelQuantInfo[];
  error?: string;
}

export interface BenchmarkResult {
  quant?: string;
  tps?: number;
  pp_tps?: number;
  size_bytes?: number;
  parameters?: number;
  architecture?: string;
  context_length?: number;
  error?: string;
}

/* ------------------------------------------------------- repositories */

export interface RepositoryList {
  repositories: RepositoryConfig[];
}

export interface RepositoryFiles {
  files: string[];
}

export interface AgentList {
  agents: AgentConfig[];
}

export interface DeleteResponse {
  status: string;
  internal_name?: string;
  id?: number;
}

export interface ToolCallDecisionResponse {
  status: string;
  executed: boolean;
  result: string | null;
}

export interface BrowseEntry {
  name: string;
  type: 'folder' | 'file';
  path: string;
}

export interface BrowseResult {
  path: string;
  entries: BrowseEntry[];
}

export interface SearchResult {
  id: number;
  title: string;
  snippet: string;
  cosine_distance: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

/* ------------------------------------------------------------- images */

export interface ImageModel {
  id: string;
  name: string;
  recipe: string;
  available: boolean;
  missing: string[];
}

export interface ImageModelList {
  models: ImageModel[];
  default: string;
}

export interface ImageRecipeSlot {
  id: string;
  label: string;
  required: boolean;
  categories: string[];
}

export interface ImageRecipe {
  id: string;
  name: string;
  description: string | null;
  model_url: string | null;
  slots: ImageRecipeSlot[];
  required_any: string[][];
}

export interface ImageRecipeList {
  recipes: ImageRecipe[];
}

export interface ImageFile {
  id: string;
  name: string;
  category: string;
}

export interface ImageFileList {
  files: ImageFile[];
}

export interface AddImageModelRequest {
  name: string;
  recipe: string;
  files: Record<string, string>;
}

export interface GenerateImageRequest {
  model?: string | null;
  prompt: string;
  params?: Record<string, ParamValue>;
}

export interface GenerateImageResponse {
  image: string;
}

/* ------------------------------------------------------------ system */

export interface SystemCpuStats {
  used_cores: number | null;
  max_cores: number;
}

export interface SystemMemoryStats {
  used_bytes: number | null;
  total_bytes: number | null;
}

export interface SystemGpuStats {
  vram_used_bytes: number | null;
  vram_total_bytes: number | null;
  utilization_percent: number | null;
}

export interface SystemStats {
  cpu: SystemCpuStats;
  ram: SystemMemoryStats;
  gpu: SystemGpuStats;
}

export interface SystemSnapshot {
  resources: SystemStats;
  servers: ManagedServer[];
}

export interface ManagedServer {
  port: string;
  provider: string;
  model: string;
  pid: number | null;
  status: string;
  active_requests: number;
  idle_seconds: number;
  cpu_cores: number | null;
  rss_bytes: number | null;
  vram_bytes: number | null;
}

export interface StopProcessResponse {
  status: string;
  port: string;
}

export interface SyncResponse {
  status: string;
}

export interface DeleteImageModelResponse {
  deleted: boolean;
}

export interface GpuBenchmarkMetric {
  tflops?: number;
}

export interface GpuBenchmark {
  status: string;
  message?: string;
  tflops?: {
    fp32?: GpuBenchmarkMetric;
    fp16?: GpuBenchmarkMetric;
    int8?: GpuBenchmarkMetric;
  };
  bandwidth?: { gb_s?: number };
}

/* ----------------------------------------------------------- transport */

export interface RequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  signal?: AbortSignal;
}

function property(source: unknown, key: string): unknown {
  if (typeof source !== 'object' || source === null) return undefined;
  return Reflect.get(source, key);
}

export class ApiError extends Error {
  status: number;
  blockingMessageId: number | null;

  constructor(message: string, status = 0, body: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    const blockingId = property(body, 'blocking_message_id');
    this.blockingMessageId = typeof blockingId === 'number' ? blockingId : null;
  }
}

async function parseBody(response: Response): Promise<unknown> {
  const type = response.headers.get('content-type') || '';
  if (type.includes('application/json')) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  const text = await response.text();
  return text ? { error: text } : null;
}

function errorMessage(data: unknown, status: number): string {
  const error = property(data, 'error');
  return typeof error === 'string' ? error : `Request failed (${status})`;
}

export function isAbort(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  return 'name' in error && error.name === 'AbortError';
}

async function send(path: string, options: RequestOptions = {}): Promise<Response> {
  const headers: Record<string, string> = { ...options.headers };
  const init: RequestInit = { method: options.method || 'GET', headers };
  if (options.signal) init.signal = options.signal;
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(options.body);
  }
  try {
    return await fetch(path, init);
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError('The server could not be reached.');
  }
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await send(path, options);
  const data = await parseBody(response);
  if (!response.ok) throw new ApiError(errorMessage(data, response.status), response.status, data);
  // Trust boundary: the server contract is described by the interfaces above.
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- generic response type is defined by the server contract
  return data as T;
}

export const api = {
  get: <T>(path: string, options: RequestOptions = {}) => request<T>(path, { ...options, method: 'GET' }),
  post: <T>(path: string, body?: unknown, options: RequestOptions = {}) =>
    request<T>(path, { ...options, method: 'POST', body }),
  put: <T>(path: string, body?: unknown, options: RequestOptions = {}) =>
    request<T>(path, { ...options, method: 'PUT', body }),
  patch: <T>(path: string, body?: unknown, options: RequestOptions = {}) =>
    request<T>(path, { ...options, method: 'PATCH', body }),
  del: <T>(path: string, options: RequestOptions = {}) => request<T>(path, { ...options, method: 'DELETE' }),
};

async function* readNdjson<T>(response: Response): AsyncGenerator<T, void, unknown> {
  function parseLine(line: string): T {
    const event: T = JSON.parse(line);
    return event;
  }
  if (!response.body) {
    const text = await response.text();
    for (const line of text.split('\n')) if (line.trim()) yield parseLine(line);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf('\n');
      while (newline !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) yield parseLine(line);
        newline = buffer.indexOf('\n');
      }
    }
    buffer += decoder.decode();
    for (const line of buffer.split('\n')) if (line.trim()) yield parseLine(line);
  } finally {
    reader.releaseLock();
  }
}

export interface NdjsonStream<T> {
  conversationId: number | null;
  events: AsyncGenerator<T, void, unknown>;
}

export async function openNdjson<T = unknown>(
  path: string,
  options: RequestOptions = {},
): Promise<NdjsonStream<T>> {
  const response = await send(path, options);
  if (!response.ok) {
    const data = await parseBody(response);
    throw new ApiError(errorMessage(data, response.status), response.status, data);
  }
  const header = response.headers.get('X-Conversation-ID');
  return {
    conversationId: header ? Number(header) : null,
    events: readNdjson<T>(response),
  };
}
