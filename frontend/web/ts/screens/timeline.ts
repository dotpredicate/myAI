import { el, clear, createCopyButton, messageOf } from '../ui.js';
import { type DiffLineKind, parseDiff, proposalDiff } from '../core/diff.js';
import { createMarkdown } from '../markdown.js';
import { argumentsText, outcomeText } from '../core/api.js';
import type {
  ConversationElement,
  StreamEvent,
  StoredMessageRecord,
  ToolArguments,
  ToolOutcome,
} from '../core/api.js';
import { Row, Stack, LineHint, FieldLabel, Notice, Button, TextArea, Summary } from '../components.js';

type Role = 'user' | 'assistant';

interface BlockBase {
  node: HTMLElement;
  setId(id: number): void;
  end(): void;
}

interface MessageBlock extends BlockBase {
  kind: 'message';
  role: Role;
  append(chunk: string): void;
  setContent(value: string): void;
  rawText(): string;
}

interface MessagePart extends BlockBase {
  kind: 'message-part';
  append(chunk: string): void;
  setContent(value: string): void;
  rawText(): string;
}

interface ThinkingPart extends BlockBase {
  kind: 'thinking';
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
  setResult(outcome: ToolOutcome | null | undefined): void;
  hideActions(): void;
  focusAction(): HTMLElement | null;
}

interface NoteBlock extends BlockBase {
  kind: 'note';
}

type Block = MessageBlock | MessagePart | ThinkingPart | ToolBlock | NoteBlock;

type ActivePart = MessagePart | ThinkingPart | ToolBlock;

/** One assistant turn: a single role header with all of its parts inside. */
interface AssistantTurn {
  node: HTMLElement;
  addMessagePart(): MessagePart;
  addThinkingPart(): ThinkingPart;
  addBlock(block: ToolBlock | NoteBlock): void;
  appendText(text: string): void;
  markContent(): void;
  hasContent(): boolean;
}

export interface TimelineCallbacks {
  onBlockingChange?: (id: number | null) => void;
  onGenerationError?: (message: string) => void;
  onApprove?: (id: number, block: ToolBlock) => void | Promise<void>;
  onReject?: (id: number, comment: string, block: ToolBlock) => void | Promise<void>;
}


/** Exhaustive on purpose: a new DiffLineKind in the core must break this. */
const DIFF_CLASS: Record<DiffLineKind, string> = {
  add: 'diff-line diff-line--add',
  remove: 'diff-line diff-line--remove',
  meta: 'diff-line diff-line--meta',
  context: 'diff-line',
};

export function renderDiff(container: Element, diffText: string | null | undefined): void {
  const diff = el('div', { class: 'diff' });
  for (const line of parseDiff(diffText)) {
    diff.appendChild(el('span', { class: DIFF_CLASS[line.kind] }, line.text || ' '));
  }
  container.appendChild(diff);
}

export function renderOutput(container: Element, outcome: ToolOutcome | null | undefined): void {
  if (!outcome) return;
  const proposal = proposalDiff(outcome);
  if (proposal !== null) {
    if (proposal.target !== '') {
      container.appendChild(LineHint('Target: ', el('code', {}, proposal.target)));
    }
    renderDiff(container, proposal.diff);
    return;
  }
  container.appendChild(el('pre', {}, outcomeText(outcome)));
}

function createUserMessageBlock(): MessageBlock {
  const raw = { text: '' };
  const label = el('strong', { class: 'message-author' }, 'user:');
  const head = Row(label);
  const body = el('div', { class: 'message-body message-body--user' });
  const textNode = document.createTextNode('');
  body.appendChild(textNode);
  const node = el('article', { class: 'message', dataset: { role: 'user' } }, head, body);

  return {
    kind: 'message',
    role: 'user',
    node,
    setId(id: number) {
      node.dataset.messageId = String(id);
    },
    append(content: string) {
      if (!content) return;
      raw.text += content;
      textNode.textContent = raw.text;
    },
    setContent(content: string) {
      raw.text = content || '';
      textNode.textContent = raw.text;
    },
    end() {},
    rawText() {
      return raw.text;
    },
  };
}

function createMessagePart(): MessagePart {
  const raw = { text: '' };
  const node = el('div', { class: 'markdown' });
  const markdown = createMarkdown(node);

  return {
    kind: 'message-part',
    node,
    setId(id: number) {
      node.dataset.messageId = String(id);
    },
    append(content: string) {
      if (!content) return;
      raw.text += content;
      markdown.write(content);
    },
    setContent(value: string) {
      raw.text = value || '';
      markdown.set(raw.text);
    },
    end() {
      markdown.end();
    },
    rawText() {
      return raw.text;
    },
  };
}

function createThinkingPart(): ThinkingPart {
  const raw = { text: '' };
  const content = el('div', { class: 'reasoning-content stack' });
  const body = el('div', { class: 'markdown' });
  const actions = Row(createCopyButton(() => raw.text, 'Copy'));
  content.appendChild(actions);
  content.appendChild(body);
  const markdown = createMarkdown(body);
  const details = el('details', {},
    Summary('Reasoning'),
    content,
  );

  return {
    kind: 'thinking',
    node: details,
    setId(id: number) {
      details.dataset.messageId = String(id);
    },
    append(chunk: string) {
      if (!chunk) return;
      raw.text += chunk;
      markdown.write(chunk);
    },
    setContent(value: string) {
      raw.text = value || '';
      markdown.set(raw.text);
    },
    end() {
      markdown.end();
    },
    rawText() {
      return raw.text;
    },
  };
}

function createAssistantTurn(): AssistantTurn {
  const raw = { text: '' };
  const label = el('strong', { class: 'message-author' }, 'assistant:');
  const head = Row(label);
  head.appendChild(createCopyButton(() => raw.text, 'Copy'));
  const body = el('div', { class: 'message-body message-body--assistant stack' });
  const node = el('article', { class: 'message', dataset: { role: 'assistant' } }, head, body);
  let content = false;
  let freshPart = false;

  return {
    node,
    addMessagePart() {
      const part = createMessagePart();
      body.appendChild(part.node);
      freshPart = true;
      return part;
    },
    addThinkingPart() {
      const part = createThinkingPart();
      body.appendChild(part.node);
      return part;
    },
    addBlock(block) {
      body.appendChild(block.node);
      content = true;
    },
    appendText(text) {
      if (!text) return;
      if (freshPart) {
        if (raw.text) raw.text += '\n\n';
        freshPart = false;
      }
      raw.text += text;
      if (text.trim()) content = true;
    },
    markContent() {
      content = true;
    },
    hasContent() {
      return content;
    },
  };
}

function createToolBlock(
  name: string,
  args: ToolArguments | null,
  initialOutcome: ToolOutcome | null | undefined,
): ToolBlock {
  let finalOutcome = initialOutcome;
  const seenOutputs = new Set<string>();
  const actions = Row();
  const pendingForm = Stack();
  const notes = Stack();
  const outputs = Stack();
  const body = Stack();

  const copy = createCopyButton(() => {
    const outcome = finalOutcome;
    return outcome ? outcomeText(outcome) : argumentsText(args);
  }, 'Copy');
  body.appendChild(Row(copy));
  body.appendChild(Stack(el('h4', { class: 'hint' }, 'Input'), el('pre', {}, argumentsText(args))));
  body.appendChild(outputs);
  body.appendChild(notes);
  body.appendChild(actions);
  body.appendChild(pendingForm);

  const details = el('details', { class: 'tool-call' },
    Summary(`Function call: ${name || 'unknown'}`),
    body,
  );

  function addOutput(outcome: ToolOutcome | null | undefined, label: string): void {
    if (!outcome) return;
    const text = outcomeText(outcome);
    if (seenOutputs.has(text)) return;
    seenOutputs.add(text);
    const section = Stack(el('h4', { class: 'hint' }, label));
    renderOutput(section, outcome);
    outputs.appendChild(section);
  }

  addOutput(initialOutcome, 'Output');

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
      clear(pendingForm);
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
      pendingForm.appendChild(error);
      pendingForm.appendChild(rejectForm);
    },
    setDecision(decision, comment) {
      block.pending = false;
      clear(actions);
      clear(pendingForm);
      const approved = decision === 'approve';
      const text = approved ? '✓ Approved' : `✗ Rejected${comment ? `: ${comment}` : ''}`;
      const note = Notice({ tight: true, level: approved ? 'success' : 'error' }, text);
      notes.appendChild(note);
    },
    setResult(outcome) {
      if (outcome) finalOutcome = outcome;
      addOutput(outcome, 'Result');
    },
    hideActions() {
      clear(actions);
      clear(pendingForm);
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
  assistantTurn: AssistantTurn | null;
  activePart: ActivePart | null;
  lastSequence: number;
  blockingId: number | null;

  constructor(container: HTMLElement, callbacks: TimelineCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this.blocksById = new Map();
    this.lastTool = null;
    this.assistantTurn = null;
    this.activePart = null;
    this.lastSequence = 0;
    this.blockingId = null;
  }

  reset(): void {
    clear(this.container);
    this.blocksById.clear();
    this.lastTool = null;
    this.assistantTurn = null;
    this.activePart = null;
    this.lastSequence = 0;
    this.blockingId = null;
  }

  closePart(): void {
    const part = this.activePart;
    if (!part) return;
    if (part.kind !== 'tool' && part.rawText().trim() === '') {
      part.node.remove();
    } else {
      part.end();
    }
    this.activePart = null;
  }

  endAssistantTurn(): void {
    this.closePart();
    const turn = this.assistantTurn;
    if (!turn) return;
    if (!turn.hasContent()) turn.node.remove();
    this.assistantTurn = null;
  }

  finalizeActive(): void {
    this.endAssistantTurn();
  }

  register<T extends Block>(block: T): T {
    this.container.appendChild(block.node);
    return block;
  }

  ensureAssistantTurn(): AssistantTurn {
    if (!this.assistantTurn) {
      const turn = createAssistantTurn();
      this.container.appendChild(turn.node);
      this.assistantTurn = turn;
    }
    return this.assistantTurn;
  }

  addUserMessage(text: string): MessageBlock {
    this.finalizeActive();
    const block = createUserMessageBlock();
    block.append(text);
    block.end();
    this.register(block);
    return block;
  }

  removeBlock(block: Block | null | undefined): void {
    if (!block) return;
    if (this.activePart?.node === block.node) this.activePart = null;
    if (this.assistantTurn?.node === block.node) this.assistantTurn = null;
    block.node.remove();
  }

  addStored(record: StoredMessageRecord): Block | null {
    const element: ConversationElement = record.element;
    const seq = record.sequence_id || 0;
    this.lastSequence = Math.max(this.lastSequence, seq);
    let block: Block | null = null;

    if (element.type === 'prompt') {
      this.endAssistantTurn();
      const message = createUserMessageBlock();
      message.append(element.content || '');
      message.end();
      message.setId(record.id);
      this.register(message);
      block = message;
    } else if (element.type === 'message') {
      const turn = this.ensureAssistantTurn();
      const part = turn.addMessagePart();
      part.append(element.content || '');
      part.end();
      part.setId(record.id);
      turn.appendText(element.content || '');
      block = part;
    } else if (element.type === 'thinking') {
      const turn = this.ensureAssistantTurn();
      const thinking = turn.addThinkingPart();
      thinking.setContent(element.content || '');
      thinking.end();
      thinking.setId(record.id);
      if ((element.content || '').trim()) turn.markContent();
      block = thinking;
    } else if (element.type === 'tool_call') {
      const turn = this.ensureAssistantTurn();
      const tool = createToolBlock(element.name, element.arguments, element.outcome);
      tool.setId(record.id);
      this.blocksById.set(record.id, tool);
      this.lastTool = tool;
      turn.addBlock(tool);
      block = tool;
    } else if (element.type === 'tool_call_result') {
      const target = this.blocksById.get(element.original_message_id);
      if (target) target.setResult(element.outcome);
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
      case 'prompt': {
        this.endAssistantTurn();
        const block = createUserMessageBlock();
        block.append(event.content || '');
        block.end();
        this.register(block);
        return;
      }
      case 'message': {
        const turn = this.ensureAssistantTurn();
        if (!this.activePart || this.activePart.kind !== 'message-part') {
          this.closePart();
          this.activePart = turn.addMessagePart();
        }
        this.activePart.append(event.content || '');
        turn.appendText(event.content || '');
        return;
      }
      case 'thinking': {
        const turn = this.ensureAssistantTurn();
        if (!this.activePart || this.activePart.kind !== 'thinking') {
          this.closePart();
          this.activePart = turn.addThinkingPart();
        }
        this.activePart.append(event.content || '');
        if ((event.content || '').trim()) turn.markContent();
        return;
      }
      case 'tool_call': {
        const turn = this.ensureAssistantTurn();
        this.closePart();
        const block = createToolBlock(event.name, event.arguments, event.outcome);
        this.lastTool = block;
        turn.addBlock(block);
        this.activePart = block;
        return;
      }
      case 'finalized': {
        const id = event.id;
        const seq = event.sequence_id || 0;
        this.lastSequence = Math.max(this.lastSequence, seq);
        if (this.activePart) {
          this.activePart.setId(id);
          if (this.activePart.kind === 'tool') {
            this.blocksById.set(id, this.activePart);
            if (this.blockingId === id) this.markBlocking(id);
          }
          this.closePart();
        }
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
        block?.setResult(event.outcome);
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
    outcome?: ToolOutcome | null,
  ): void {
    const block = this.blocksById.get(Number(id));
    if (block) {
      block.setDecision(decision, comment);
      if (outcome) block.setResult(outcome);
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
