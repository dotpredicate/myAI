import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyDiffLine, parseDiff, proposalDiff } from '#core/diff';
import type { ToolOutcome } from '#core/api';
import { Stack } from '../src/components.ts';
import { diffNodes } from '../src/diff.ts';
import type { VNode } from '../src/h.ts';
import { layout, paintLine } from '../src/render.ts';

function rowsOf(nodes: VNode[], width = 40): string[] {
  return layout(Stack(...nodes), width).map((line) => paintLine(line, width));
}

function textOf(nodes: VNode[], width = 40): string[] {
  // oxlint-disable-next-line eslint/no-control-regex -- stripping ANSI escapes is the point
  return rowsOf(nodes, width).map((row) => row.replace(/\x1b\[[0-9;]*m/g, '').trimEnd());
}

test('file markers and hunk headers win over the +/- rules', () => {
  assert.equal(classifyDiffLine('+++ b/tools.py'), 'meta');
  assert.equal(classifyDiffLine('--- a/tools.py'), 'meta');
  assert.equal(classifyDiffLine('@@ -113,6 +113,9 @@'), 'meta');
  assert.equal(classifyDiffLine('+added'), 'add');
  assert.equal(classifyDiffLine('-removed'), 'remove');
});

test('git headers and the no-newline marker are meta too', () => {
  assert.equal(classifyDiffLine('diff --git a/x.py b/x.py'), 'meta');
  assert.equal(classifyDiffLine('index 1a2b3c4..5d6e7f8 100644'), 'meta');
  assert.equal(classifyDiffLine('\\ No newline at end of file'), 'meta');
  assert.equal(classifyDiffLine('Binary files a/x and b/x differ'), 'meta');
});

test('context lines are the ones with a leading space or nothing at all', () => {
  assert.equal(classifyDiffLine(' unchanged'), 'context');
  assert.equal(classifyDiffLine(''), 'context');
});

test('parseDiff strips CR and keeps the empty line a trailing newline leaves', () => {
  const lines = parseDiff('+a\r\n-b\n');
  assert.deepEqual(lines.map((line) => line.text), ['+a', '-b', '']);
  assert.deepEqual(lines.map((line) => line.kind), ['add', 'remove', 'context']);
});

const proposal: ToolOutcome = {
  type: 'proposal',
  format: 'replace',
  target: '/repositories/x',
  source: '/workspace/x',
  diff: '--- a/x\n+++ b/x\n',
};

test('proposalDiff answers for proposals only', () => {
  assert.deepEqual(proposalDiff(proposal), { target: '/repositories/x', diff: proposal.diff });
  assert.equal(proposalDiff({ type: 'error', message: 'nope' }), null);
  assert.equal(
    proposalDiff({ type: 'success', tool: 'bash', returncode: 0, stdout: '', stderr: '' }),
    null,
  );
  assert.equal(proposalDiff(null), null);
});

test('a proposal without a target still renders its diff', () => {
  assert.deepEqual(proposalDiff({ ...proposal, target: '' }), { target: '', diff: proposal.diff });
});

test('every line kind keeps its own colour', () => {
  const [add, remove, meta, context] = rowsOf(diffNodes('+a\n-b\n@@ -1 +1 @@\n c'));
  assert.ok(add?.includes('\x1b[0;32m'), 'additions are green');
  assert.ok(remove?.includes('\x1b[0;31m'), 'removals are red');
  assert.ok(meta?.includes('\x1b[0;2m'), 'headers are dim');
  assert.ok(!context?.includes('\x1b[0;3'), 'context keeps the default colour');
});

test('blank diff lines still take a row, and an empty diff takes one', () => {
  assert.deepEqual(textOf(diffNodes('+a\n\n+b')), ['+a', '', '+b']);
  assert.deepEqual(textOf(diffNodes('')), ['']);
});

test('a line wider than the terminal wraps instead of being cut', () => {
  assert.deepEqual(textOf(diffNodes('+0123456789ab'), 10), ['+012345678', '9ab']);
});

test('wrapped context keeps its indentation', () => {
  assert.deepEqual(textOf(diffNodes('     value = 1 + 2'), 10), ['     value', ' = 1 + 2']);
});

test('no diff line is dropped, however long the patch', () => {
  const diff = Array.from({ length: 600 }, (_, index) => `+line ${index}`).join('\n');
  assert.equal(diffNodes(diff).length, 600);
});
