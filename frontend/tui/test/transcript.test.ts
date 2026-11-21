import assert from 'node:assert/strict';
import test from 'node:test';
import type { StoredMessageRecord } from '#core/api';
import { applyStreamEvent, transcriptFromMessages, type TranscriptItem } from '../src/transcript.ts';

function record(id: number, sequence_id: number, element: StoredMessageRecord['element']): StoredMessageRecord {
  return { id, conversation_id: 1, sequence_id, element, created_at: '' };
}

test('the prompt is the user item and the message the assistant one', () => {
  const items = transcriptFromMessages([
    record(1, 1, { type: 'prompt', content: 'why?' }),
    record(2, 2, { type: 'message', content: 'because' }),
  ]);

  assert.deepEqual(items, [
    { kind: 'user', text: 'why?' },
    { kind: 'assistant', text: 'because' },
  ]);
});

test('a replayed prompt is not repeated as assistant text', () => {
  const items: TranscriptItem[] = [];

  applyStreamEvent(items, { type: 'prompt', content: 'why?' });
  applyStreamEvent(items, { type: 'message', content: 'because' });

  assert.deepEqual(items, [{ kind: 'assistant', text: 'because' }]);
});
