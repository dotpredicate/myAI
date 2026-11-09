import { el, clear, prettyJson, parseJsonObject, createCopyButton, messageOf } from '../ui.js';
import { default_renderer, parser, parser_end, parser_write } from '../../vendor/smd.js';
import type { ConversationElement, StreamEvent, StoredMessageRecord } from '../core/api.js';
import { Row, Stack, LineHint, FieldLabel, Notice, Button, TextArea, Summary } from '../components.js';

type Role = 'user' | 'assistant';

interface MarkdownWriter {
  write(chunk: string): void;
  end(): void;
}

interface BlockBase {
  node: HTMLElement;
  setId(id: number): void;
  end(): void;
}

interface MessageBlock extends BlockBase {
  kind: 'message' | 'thinking';
  role?: Role;
  append(chunk: string): void;
  setContent(value: string): void;
  rawText(): string;
}

interface ToolHandlers {
  onApprove?: () => void | Promise<void>;
  onReject?: (comment: string) => void | Promise<void>;
}

export interface ToolBlock extends BlockBase {
  kind: 'tool';
  node: HTMLDetailsElement;
  name: string;
  pending: boolean;
  setPending(pending: boolean, handlers?: ToolHandlers): void;
  setDecision(decision: 'approve' | 'reject', comment: string | null): void;
  setResult(result: string | null | undefined): void;
  hideActions(): void;
  focusAction(): HTMLElement | null;
}

interface NoteBlock extends BlockBase {
  kind: 'note';
}

type Block = MessageBlock | ToolBlock | NoteBlock;

type ActiveState =
  | { kind: 'message'; role: Role; block: MessageBlock }
  | { kind: 'thinking'; block: MessageBlock }
  | { kind: 'tool'; block: ToolBlock };

export interface TimelineCallbacks {
  onBlockingChange?: (id: number | null) => void;
  onGenerationError?: (message: string) => void;
  onApprove?: (id: number, block: ToolBlock) => void | Promise<void>;
  onReject?: (id: number, comment: string, block: ToolBlock) => void | Promise<void>;
}


export function createMarkdown(container: Element): MarkdownWriter {
  const instance = parser(default_renderer(container));
  let ended = false;
  return {
    write(chunk: string) {
      if (ended || !chunk) return;
      parser_write(instance, chunk);
    },
    end() {
      if (ended) return;
      ended = true;
      parser_end(instance);
    },
  };
}

export function renderDiff(container: Element, diffText: string | null | undefined): void {
  const diff = el('div', { class: 'diff' });
  const lines = (diffText ?? '').split('\n');
  for (const line of lines) {
    let modifier = '';
    if (line.startsWith('+++') || line.startsWith('---')) modifier = 'diff-line--meta';
    else if (line.startsWith('@@')) modifier = 'diff-line--meta';
    else if (line.startsWith('+')) modifier = 'diff-line--add';
    else if (line.startsWith('-')) modifier = 'diff-line--remove';
    diff.appendChild(el('span', { class: `diff-line ${modifier}`.trim() }, line || ' '));
  }
  container.appendChild(diff);
}

export function renderOutput(container: Element, text: string | null | undefined): void {
  if (text === null || text === undefined || text === '') return;
  const parsed = parseJsonObject(text);
  if (parsed && (parsed.type === 'replace_proposal' || parsed.type === 'diff_proposal') && typeof parsed.diff === 'string') {
    if (typeof parsed.target === 'string' && parsed.target) {
      container.appendChild(LineHint('Target: ', el('code', {}, parsed.target)));
    }
    renderDiff(container, parsed.diff);
    return;
  }
  container.appendChild(el('pre', {}, prettyJson(text)));
}

export function outputCopyText(text: string | null | undefined): string {
  const parsed = parseJsonObject(text);
  if (parsed && typeof parsed.diff === 'string') return parsed.diff;
  return prettyJson(text);
}

function createMessageBlock(role: Role): MessageBlock {
  const raw = { text: '' };
  const author = el('strong', { class: 'message-author' }, `${role}:`);
  const head = Row(author);
  const body = el('div', { class: `message-body message-body--${role}` });
  let markdown: MarkdownWriter | null = null;
  let textNode: Text | null = null;

  if (role === 'assistant') {
    const content = el('div', { class: 'markdown' });
    body.appendChild(content);
    markdown = createMarkdown(content);
    head.appendChild(createCopyButton(() => raw.text, 'Copy'));
  } else {
    textNode = document.createTextNode('');
    body.appendChild(textNode);
  }

  const node = el('article', { class: 'message', dataset: { role } }, head, body);

  return {
    kind: 'message',
    role,
    node,
    setId(id: number) {
      node.dataset.messageId = String(id);
    },
    append(content: string) {
      if (!content) return;
      raw.text += content;
      if (markdown) markdown.write(content);
      else if (textNode) textNode.textContent = raw.text;
    },
    setContent(content: string) {
      raw.text = content || '';
      if (markdown) markdown.write(raw.text);
      else if (textNode) textNode.textContent = raw.text;
    },
    end() {
      if (markdown) markdown.end();
    },
    rawText() {
      return raw.text;
    },
  };
}

function createThinkingBlock(): MessageBlock {
  const raw = { text: '' };
  const content = el('div', { class: 'reasoning-content stack' });
  const text = el('div', {});
  content.appendChild(text);
  const actions = Row(createCopyButton(() => raw.text, 'Copy'));
  content.insertBefore(actions, text);
  const details = el('details', {},
    Summary('Reasoning'),
    content,
  );
  const node = details;

  return {
    kind: 'thinking',
    node,
    setId(id: number) {
      node.dataset.messageId = String(id);
    },
    append(chunk: string) {
      if (!chunk) return;
      raw.text += chunk;
      text.textContent = raw.text;
    },
    setContent(value: string) {
      raw.text = value || '';
      text.textContent = raw.text;
    },
    end() {},
    rawText() {
      return raw.text;
    },
  };
}

function createToolBlock(name: string, parameters: string, initialResult: string | null | undefined): ToolBlock {
  let finalResult = initialResult || '';
  const seenOutputs = new Set<string>();
  const actions = Row();
  const notes = Stack();
  const outputs = Stack();
  const body = Stack();

  const copy = createCopyButton(() => outputCopyText(finalResult || parameters), 'Copy');
  body.appendChild(Row(copy));
  body.appendChild(Stack(el('h4', { class: 'hint' }, 'Input'), el('pre', {}, prettyJson(parameters))));
  body.appendChild(actions);
  body.appendChild(outputs);
  body.appendChild(notes);

  const details = el('details', { class: 'tool-call' },
    Summary(`Function call: ${name || 'unknown'}`),
    body,
  );

  function addOutput(result: string | null | undefined, label: string): void {
    if (result === null || result === undefined || result === '') return;
    if (seenOutputs.has(result)) return;
    seenOutputs.add(result);
    const section = Stack(el('h4', { class: 'hint' }, label));
    renderOutput(section, result);
    outputs.appendChild(section);
  }

  addOutput(initialResult, 'Output');

  const block: ToolBlock = {
    kind: 'tool',
    name,
    node: details,
    pending: false,
    setId(id: number) {
      details.dataset.messageId = String(id);
    },
    setPending(pending: boolean, handlers: ToolHandlers = {}) {
      block.pending = pending;
      clear(actions);
      if (!pending) return;
      details.open = true;
      const approve = Button('Approve', { variant: 'primary', small: true });
      const reject = Button('Reject', { small: true });
      const error = el('div', { class: 'field-error', hidden: true });
      const rejectForm = el('div', { class: 'inset stack', hidden: true });
      const reasonId = `reject-${Math.random().toString(36).slice(2)}`;
      const reason = TextArea({ id: reasonId, rows: 2, placeholder: 'Optional reason' });
      const confirmReject = Button('Confirm reject', { variant: 'danger', small: true });
      const cancelReject = Button('Cancel', { small: true });
      rejectForm.appendChild(FieldLabel('Reason (optional)', reasonId));
      rejectForm.appendChild(reason);
      rejectForm.appendChild(Row(confirmReject, cancelReject));

      function setBusy(busy: boolean): void {
        approve.disabled = busy;
        reject.disabled = busy;
        confirmReject.disabled = busy;
        cancelReject.disabled = busy;
        reason.disabled = busy;
      }

      function showError(message: string): void {
        error.textContent = message || '';
        error.hidden = !message;
      }

      approve.addEventListener('click', async () => {
        showError('');
        setBusy(true);
        try {
          await handlers.onApprove?.();
        } catch (err) {
          showError(messageOf(err) || 'Approval failed.');
          setBusy(false);
        }
      });

      reject.addEventListener('click', () => {
        rejectForm.hidden = !rejectForm.hidden;
        if (!rejectForm.hidden) reason.focus();
      });
      cancelReject.addEventListener('click', () => {
        rejectForm.hidden = true;
      });
      confirmReject.addEventListener('click', async () => {
        showError('');
        setBusy(true);
        try {
          await handlers.onReject?.(reason.value.trim());
        } catch (err) {
          showError(messageOf(err) || 'Rejection failed.');
          setBusy(false);
        }
      });

      actions.appendChild(approve);
      actions.appendChild(reject);
      actions.appendChild(error);
      actions.appendChild(rejectForm);
    },
    setDecision(decision, comment) {
      block.pending = false;
      clear(actions);
      const approved = decision === 'approve';
      const text = approved ? '✓ Approved' : `✗ Rejected${comment ? `: ${comment}` : ''}`;
      const note = Notice({ tight: true, level: approved ? 'success' : 'error' }, text);
      notes.appendChild(note);
    },
    setResult(result) {
      if (result !== undefined && result !== null) finalResult = result;
      addOutput(result, 'Result');
    },
    hideActions() {
      clear(actions);
    },
    end() {},
    focusAction() {
      return actions.querySelector<HTMLElement>('button') || details.querySelector<HTMLElement>('summary');
    },
  };

  return block;
}

function createSystemNote(text: string, variant = ''): NoteBlock {
  return {
    kind: 'note',
    node: Notice({ tight: true, class: variant }, text),
    setId() {},
    end() {},
  };
}

export class Timeline {
  container: HTMLElement;
  callbacks: TimelineCallbacks;
  blocksById: Map<number, ToolBlock>;
  lastTool: ToolBlock | null;
  active: ActiveState | null;
  lastSequence: number;
  blockingId: number | null;

  constructor(container: HTMLElement, callbacks: TimelineCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this.blocksById = new Map();
    this.lastTool = null;
    this.active = null;
    this.lastSequence = 0;
    this.blockingId = null;
  }

  reset(): void {
    clear(this.container);
    this.blocksById.clear();
    this.lastTool = null;
    this.active = null;
    this.lastSequence = 0;
    this.blockingId = null;
  }

  finalizeActive(): void {
    if (!this.active) return;
    this.active.block.end();
    this.active = null;
  }

  register<T extends Block>(block: T): T {
    this.container.appendChild(block.node);
    return block;
  }

  addUserMessage(text: string): MessageBlock {
    this.finalizeActive();
    const block = createMessageBlock('user');
    block.append(text);
    block.end();
    this.register(block);
    return block;
  }

  removeBlock(block: Block | null | undefined): void {
    if (!block) return;
    if (this.active?.block === block) this.active = null;
    block.node.remove();
  }

  addStored(record: StoredMessageRecord): Block | null {
    const element: ConversationElement = record.element;
    const seq = record.sequence_id || 0;
    this.lastSequence = Math.max(this.lastSequence, seq);
    let block: Block | null = null;

    if (element.type === 'message') {
      const message = createMessageBlock(element.author === 'user' ? 'user' : 'assistant');
      message.append(element.content || '');
      message.end();
      message.setId(record.id);
      this.register(message);
      block = message;
    } else if (element.type === 'thinking') {
      const thinking = createThinkingBlock();
      thinking.setContent(element.content || '');
      thinking.setId(record.id);
      this.register(thinking);
      block = thinking;
    } else if (element.type === 'tool_call') {
      const tool = createToolBlock(element.name, element.parameters, element.result);
      tool.setId(record.id);
      this.blocksById.set(record.id, tool);
      this.lastTool = tool;
      this.register(tool);
      block = tool;
    } else if (element.type === 'tool_call_result') {
      const target = this.blocksById.get(element.original_message_id);
      if (target) target.setResult(element.result);
      else this.register(createSystemNote('Tool result', ''));
    } else if (element.type === 'tool_call_decision') {
      const target = this.blocksById.get(element.original_message_id);
      if (target) target.setDecision(element.decision, element.comment);
      else {
        const approved = element.decision === 'approve';
        this.register(createSystemNote(
          approved ? '✓ Approved' : `✗ Rejected${element.comment ? `: ${element.comment}` : ''}`,
          approved ? 'notice--success' : 'notice--error',
        ));
      }
    }
    return block;
  }

  handleEvent(event: StreamEvent): void {
    if (typeof event.type !== 'string') return;
    switch (event.type) {
      case 'generation_error': {
        this.finalizeActive();
        this.callbacks.onGenerationError?.(event.message || 'Generation failed.');
        return;
      }
      case 'message': {
        const role: Role = event.role === 'user' ? 'user' : 'assistant';
        if (!this.active || this.active.kind !== 'message' || this.active.role !== role) {
          this.finalizeActive();
          const block = this.register(createMessageBlock(role));
          this.active = { kind: 'message', role, block };
        }
        this.active.block.append(event.content || '');
        return;
      }
      case 'thinking': {
        if (!this.active || this.active.kind !== 'thinking') {
          this.finalizeActive();
          const block = this.register(createThinkingBlock());
          this.active = { kind: 'thinking', block };
        }
        this.active.block.append(event.content || '');
        return;
      }
      case 'tool_call': {
        this.finalizeActive();
        const block = createToolBlock(event.name, event.parameters, event.result);
        this.lastTool = block;
        this.register(block);
        this.active = { kind: 'tool', block };
        return;
      }
      case 'finalized': {
        const id = event.id;
        const seq = event.sequence_id || 0;
        this.lastSequence = Math.max(this.lastSequence, seq);
        if (this.active) {
          this.active.block.setId(id);
          if (this.active.kind === 'tool') {
            this.blocksById.set(id, this.active.block);
            if (this.blockingId === id) this.markBlocking(id);
          }
        }
        this.finalizeActive();
        return;
      }
      case 'approval_request': {
        const id = event.original_message_id;
        this.blockingId = id;
        this.markBlocking(id);
        this.callbacks.onBlockingChange?.(id);
        return;
      }
      case 'tool_call_decision': {
        const id = event.original_message_id;
        const block = this.blocksById.get(id) || this.lastTool;
        block?.setDecision(event.decision, event.comment);
        if (this.blockingId === id) {
          this.blockingId = null;
          this.callbacks.onBlockingChange?.(null);
        }
        return;
      }
      case 'tool_call_result': {
        const id = event.original_message_id;
        const block = this.blocksById.get(id) || this.lastTool;
        block?.setResult(event.result);
        return;
      }
    }
  }

  markBlocking(id: number): void {
    for (const [blockId, block] of this.blocksById) {
      if (blockId === id) {
        block.setPending(true, {
          onApprove: () => this.callbacks.onApprove?.(blockId, block),
          onReject: (comment: string) => this.callbacks.onReject?.(blockId, comment, block),
        });
      } else if (block.pending) {
        block.setPending(false);
      }
    }
  }

  setBlocking(id: number | string | null | undefined): void {
    this.blockingId = id ? Number(id) : null;
    if (this.blockingId) this.markBlocking(this.blockingId);
    else {
      for (const block of this.blocksById.values()) {
        if (block.pending) block.setPending(false);
      }
    }
  }

  applyDecision(
    id: number | string,
    decision: 'approve' | 'reject',
    comment: string | null,
    result?: string | null,
  ): void {
    const block = this.blocksById.get(Number(id));
    if (block) {
      block.setDecision(decision, comment);
      if (result) block.setResult(result);
    }
    if (this.blockingId === Number(id)) this.blockingId = null;
  }

  getBlock(id: number | string | null | undefined): ToolBlock | undefined {
    return this.blocksById.get(Number(id));
  }

  endStream(): void {
    this.finalizeActive();
  }
}
