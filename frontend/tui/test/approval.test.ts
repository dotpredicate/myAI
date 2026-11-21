import assert from 'node:assert/strict';
import test from 'node:test';
import { approvalForShortcut } from '../src/approval.ts';

test('y approves while the composer is empty', () => {
  assert.deepEqual(approvalForShortcut('y', ''), { decision: 'approve', comment: '' });
});

test('n rejects while the composer is empty', () => {
  assert.deepEqual(approvalForShortcut('n', ''), { decision: 'reject', comment: '' });
});

test('no other letter decides', () => {
  assert.equal(approvalForShortcut('x', ''), null);
  assert.equal(approvalForShortcut('', ''), null);
});

test('once a message is started, y and n type instead of deciding', () => {
  assert.equal(approvalForShortcut('y', 'nie rób tego'), null);
  assert.equal(approvalForShortcut('n', 'nie rób tego'), null);
});
