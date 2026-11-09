// Conversation transcript for the terminal client: server elements and stream
// events folded into the positions the TUI renders.

import type { ConversationElement, StreamEvent, StoredMessageRecord } from '#core/api';

export interface ToolItem {
  kind: 'tool';
  /** Server message id; null only while the tool call is still streaming. */
  id: number | null;
  name: string;
  parameters: string;
  /** What the tool proposes to do (the pending call body). */
  proposal: string | null;
  /** What the server actually executed after a decision. */
  result: string | null;
}

export type TranscriptItem =
  | { kind: 'user' | 'assistant' | 'thinking' | 'notice' | 'error'; text: string }
  | ToolItem;

export function truncate(text: string, max: number): string {
  // Counting code points is intentional: it is a display shortcut, not a
  // grapheme-safe truncation.
  // oxlint-disable-next-line typescript/no-misused-spread -- code points, not graphemes
  const chars = [...text];
  if (chars.length <= max) return text;
  return `${chars.slice(0, max).join('')}… (${chars.length - max} more chars)`;
}

export function findTool(items: TranscriptItem[], id: number | null): ToolItem | null {
  if (id === null) return null;
  for (const item of items) {
    if (item.kind === 'tool' && item.id === id) return item;
  }
  return null;
}

/** Attach an execution result to the tool call it belongs to. */
export function applyToolResult(items: TranscriptItem[], messageId: number, result: string): boolean {
  const tool = findTool(items, messageId);
  if (tool === null) return false;
  tool.result = result;
  return true;
}

function itemFromElement(element: ConversationElement, messageId: number): TranscriptItem | null {
  switch (element.type) {
    case 'message':
      return element.author === 'user'
        ? { kind: 'user', text: element.content }
        : { kind: 'assistant', text: element.content };
    case 'thinking':
      return { kind: 'thinking', text: element.content };
    case 'tool_call':
      return {
        kind: 'tool',
        id: messageId,
        name: element.name,
        parameters: element.parameters,
        proposal: element.result,
        result: null,
      };
    case 'approval_request':
      return { kind: 'notice', text: `approval requested for message ${element.original_message_id}` };
    case 'tool_call_decision':
      return { kind: 'notice', text: `decision: ${element.decision}` };
    default:
      return null;
  }
}

export function transcriptFromMessages(messages: StoredMessageRecord[]): TranscriptItem[] {
  const items: TranscriptItem[] = [];
  for (const message of messages) {
    const element = message.element;
    if (element.type === 'tool_call_result') {
      applyToolResult(items, element.original_message_id, element.result);
      continue;
    }
    const item = itemFromElement(element, message.id);
    if (item !== null) items.push(item);
  }
  return items;
}

function lastItem(items: TranscriptItem[]): TranscriptItem | undefined {
  return items.length === 0 ? undefined : items[items.length - 1];
}

/**
 * Fold one streamed event into the transcript. Mutates `items` in place so a
 * token stream does not clone the whole history per chunk.
 */
export function applyStreamEvent(items: TranscriptItem[], event: StreamEvent): void {
  const last = lastItem(items);
  switch (event.type) {
    case 'message': {
      if (event.role !== 'assistant') return;
      if (last?.kind === 'assistant') last.text += event.content;
      else items.push({ kind: 'assistant', text: event.content });
      return;
    }
    case 'thinking': {
      if (last?.kind === 'thinking') last.text += event.content;
      else items.push({ kind: 'thinking', text: event.content });
      return;
    }
    case 'tool_call':
      items.push({
        kind: 'tool',
        id: null,
        name: event.name,
        parameters: event.parameters,
        proposal: event.result,
        result: null,
      });
      return;
    case 'finalized':
      // A tool_call event is followed by the finalized message that carries its id.
      if (last?.kind === 'tool' && last.id === null) last.id = event.id;
      return;
    case 'tool_call_decision':
      items.push({ kind: 'notice', text: `decision: ${event.decision}` });
      return;
    case 'tool_call_result':
      if (!applyToolResult(items, event.original_message_id, event.result)) {
        items.push({ kind: 'notice', text: `tool result: ${truncate(event.result, 300)}` });
      }
      return;
    case 'generation_error':
      items.push({ kind: 'error', text: event.message });
      return;
    default:
      return;
  }
}
