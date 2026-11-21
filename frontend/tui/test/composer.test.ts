import assert from 'node:assert/strict';
import test from 'node:test';
import {
  backspaceComposer,
  caretRowUpDown,
  caretToLineEnd,
  caretToLineStart,
  clearComposer,
  composerRows,
  deleteComposer,
  emptyComposer,
  insertText,
  MAX_ROWS,
  moveCaret,
  setComposerText,
} from '../src/composer.ts';

/** The visible text of each rendered row, with the rail stripped off. */
function rowTexts(composer: ReturnType<typeof emptyComposer>, width: number, maxRows: number): string[] {
  return composerRows(composer, width, { maxRows, blocked: false })
    .map((line) => line.map((segment) => segment.text).join(''))
    .map((row) => row.slice(2).replace(/\s+$/, ''));
}

test('editing works on grapheme clusters, not code units', () => {
  const composer = emptyComposer();
  insertText(composer, 'a🇵🇱b');

  moveCaret(composer, -1);
  backspaceComposer(composer);
  assert.equal(composer.text, 'ab');
  assert.equal(composer.caret, 1);

  deleteComposer(composer);
  assert.equal(composer.text, 'a');

  clearComposer(composer);
  assert.equal(composer.text, '');
  assert.equal(composer.caret, 0);
});

test('newlines become lines and Home/End stay inside one', () => {
  const composer = emptyComposer();
  insertText(composer, 'pierwsza\ndruga');

  caretToLineStart(composer);
  assert.equal(composer.caret, 9); // just past the newline
  insertText(composer, '> ');
  assert.equal(composer.text, 'pierwsza\n> druga');

  caretToLineEnd(composer);
  insertText(composer, '!');
  assert.equal(composer.text, 'pierwsza\n> druga!');
});

test('pasted text keeps its line structure and normalises CRLF', () => {
  const composer = emptyComposer();
  setComposerText(composer, 'one\r\ntwo\rthree');
  assert.equal(composer.text, 'one\ntwo\nthree');
  assert.equal(composer.caret, 13);
});

test('blank lines stay blank instead of leaking a newline into a row', () => {
  const composer = emptyComposer();
  setComposerText(composer, 'a\n\nb');

  const rows = rowTexts(composer, 40, MAX_ROWS);
  assert.deepEqual(rows, ['a', '', 'b']);
  for (const row of rows) {
    // oxlint-disable-next-line eslint/no-control-regex -- control characters are what we look for
    assert(!/[\u0000-\u001f]/.test(row), `row holds a control character: ${JSON.stringify(row)}`);
  }
});

test('a trailing newline leaves one empty row for the caret', () => {
  const composer = emptyComposer();
  setComposerText(composer, 'abc\n');
  assert.deepEqual(rowTexts(composer, 40, MAX_ROWS), ['abc', '']);
});

test('long text wraps into rows instead of being cut off', () => {
  const composer = emptyComposer();
  insertText(composer, 'Ala ma kota a kot ma Ale i to bardzo dlugie zdanie');

  const rows = rowTexts(composer, 24, 6);
  assert(rows.length >= 3);
  assert.equal(rows.join(' ').replace(/\s+/g, ' ').trim(), composer.text);
});

test('up and down move the caret between visual rows and keep the column', () => {
  const composer = emptyComposer();
  insertText(composer, 'Ala ma kota a kot ma Ale i to bardzo dlugie zdanie');
  const width = 24;

  let steps = 0;
  while (caretRowUpDown(composer, -1, width)) steps += 1;
  assert(steps > 0);
  const top = composer.caret;
  assert.equal(caretRowUpDown(composer, -1, width), false);

  // Down again lands back on the end of the text, column preserved.
  for (let step = 0; step < steps; step += 1) {
    assert.equal(caretRowUpDown(composer, 1, width), true);
  }
  assert.equal(caretRowUpDown(composer, 1, width), false);
  assert.equal(composer.caret, composer.text.length);
  assert(top < composer.caret);
});

test('the prompt stops growing at MAX_ROWS and scrolls under the caret', () => {
  const composer = emptyComposer();
  for (let line = 1; line <= 9; line += 1) insertText(composer, `linia ${line}\n`);

  const bottom = rowTexts(composer, 30, MAX_ROWS);
  assert.equal(bottom.length, MAX_ROWS);
  assert(bottom.includes('linia 9'));
  assert(!bottom.includes('linia 1'));

  while (caretRowUpDown(composer, -1, 30)) {
    // walk up to the first row
  }
  const top = rowTexts(composer, 30, MAX_ROWS);
  assert.equal(top.length, MAX_ROWS);
  assert.equal(top[0], 'linia 1');
});
