import assert from 'node:assert/strict';
import test from 'node:test';
import { plain } from '../src/h.ts';
import { paintLine } from '../src/render.ts';

test('padding never inherits the style of the last segment', () => {
  // A reversed caret at the end of a row used to paint the rest of it white.
  const row = paintLine([{ text: 'x', style: { ...plain, reverse: true } }], 4);
  assert.equal(row, '\x1b[0;7mx\x1b[0m   \x1b[0m');
});

test('a full row is painted without padding', () => {
  assert.equal(paintLine([{ text: 'abcd', style: plain }], 4), '\x1b[0mabcd\x1b[0m');
});
