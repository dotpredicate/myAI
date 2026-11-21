import assert from 'node:assert/strict';
import test from 'node:test';
import { outcomeText } from '#core/api';
import type { ToolOutcome } from '#core/api';

test('a proposal reads as its diff', () => {
  const outcome: ToolOutcome = {
    type: 'proposal',
    format: 'patch',
    target: '/repositories/foo/a.ts',
    source: '/workspace/p.diff',
    diff: '@@ -1 +1 @@\n-a\n+b\n',
  };
  assert.equal(outcomeText(outcome), '@@ -1 +1 @@\n-a\n+b\n');
});

test('an error reads as its message, without the envelope', () => {
  assert.equal(outcomeText({ type: 'error', message: 'Source file not found' }), 'Source file not found');
});

test('a success reads as its payload', () => {
  const outcome: ToolOutcome = { type: 'success', tool: 'bash', returncode: 0, stdout: 'hi\n', stderr: '' };
  assert.equal(outcomeText(outcome), JSON.stringify(outcome, null, 2));
});
