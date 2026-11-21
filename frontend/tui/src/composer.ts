// The prompt composer: a text buffer with a caret, and the rows it wraps into.
//
// Rows are derived from the text and the width on every frame, so painting and
// caret movement can never disagree about where a line ends. Newlines are hard
// breaks; anything longer than the width wraps at a space when it can.

import { styleOf } from './h.ts';
import type { Line, Segment } from './render.ts';
import { graphemeWidth, graphemes, textWidth } from './text.ts';

/** The prompt grows to this many rows and then scrolls under the caret. */
export const MAX_ROWS = 3;

export interface Composer {
  text: string;
  /** Grapheme offset into the text: 0 .. graphemes(text).length. */
  caret: number;
}

interface Row {
  text: string;
  /** Grapheme offset where the row starts. */
  start: number;
  /** Grapheme offset one past the row. A hard newline sits at `end`. */
  end: number;
  /** Whether the next row starts a new hard line rather than a soft wrap. */
  hard: boolean;
}

export function emptyComposer(): Composer {
  return { text: '', caret: 0 };
}

export function setComposerText(target: Composer, text: string): void {
  target.text = text.replace(/\r\n?/g, '\n');
  target.caret = graphemes(target.text).length;
}

export function clearComposer(target: Composer): void {
  target.text = '';
  target.caret = 0;
}

export function insertText(target: Composer, value: string): void {
  const added = graphemes(value.replace(/\r\n?/g, '\n'));
  if (added.length === 0) return;
  const parts = graphemes(target.text);
  parts.splice(target.caret, 0, ...added);
  target.text = parts.join('');
  target.caret += added.length;
}

export function backspaceComposer(target: Composer): void {
  if (target.caret === 0) return;
  const parts = graphemes(target.text);
  parts.splice(target.caret - 1, 1);
  target.text = parts.join('');
  target.caret -= 1;
}

export function deleteComposer(target: Composer): void {
  const parts = graphemes(target.text);
  if (target.caret >= parts.length) return;
  parts.splice(target.caret, 1);
  target.text = parts.join('');
}

export function moveCaret(target: Composer, by: -1 | 1): void {
  const last = graphemes(target.text).length;
  target.caret = Math.max(0, Math.min(last, target.caret + by));
}

/** Start of the hard line the caret sits on. */
export function caretToLineStart(target: Composer): void {
  const parts = graphemes(target.text);
  while (target.caret > 0 && parts[target.caret - 1] !== '\n') target.caret -= 1;
}

/** End of the hard line the caret sits on. */
export function caretToLineEnd(target: Composer): void {
  const parts = graphemes(target.text);
  while (target.caret < parts.length && parts[target.caret] !== '\n') target.caret += 1;
}

// Every row shows a two-cell rail (`› `) and keeps one more cell free so the
// caret at the end of a full row still has somewhere to be painted.
const MARGIN = 3;

function rowsOf(text: string, width: number): Row[] {
  const parts = graphemes(text);
  const limit = Math.max(1, width);
  const rows: Row[] = [];
  let start = 0;

  for (;;) {
    let used = 0;
    let space = -1;
    let end = start;
    while (end < parts.length && parts[end] !== '\n') {
      const part = parts[end] ?? '';
      const size = graphemeWidth(part);
      if (used + size > limit) break;
      if (part === ' ') space = end;
      used += size;
      end += 1;
    }
    // Break at the last space instead of inside a word; a word wider than the
    // row still has to be cut, and a single wide grapheme still has to advance.
    if (end < parts.length && parts[end] !== '\n' && space > start) end = space + 1;
    // An empty line is empty: its newline is the terminator, not a character of
    // the row. Painting a row that holds a newline would break the frame.
    if (end === start && end < parts.length && parts[end] !== '\n') end = start + 1;

    const hard = end < parts.length && parts[end] === '\n';
    rows.push({ text: parts.slice(start, end).join(''), start, end, hard });

    if (end >= parts.length) return rows;
    // A trailing newline still opens an empty row for the caret to sit on.
    if (hard && end + 1 >= parts.length) {
      rows.push({ text: '', start: end + 1, end: end + 1, hard: false });
      return rows;
    }
    start = hard ? end + 1 : end;
  }
}

function caretRow(rows: Row[], caret: number): number {
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index];
    if (row && caret >= row.start) return index;
  }
  return 0;
}

function caretColumn(row: Row, caret: number, parts: string[]): number {
  const within = Math.max(row.start, Math.min(caret, row.end));
  return textWidth(parts.slice(row.start, within).join(''));
}

function offsetAt(row: Row, column: number, parts: string[]): number {
  // A soft-wrapped row ends where the next one begins, so stop one grapheme
  // short: otherwise the caret would look like it fell to the next row.
  const last = row.hard || row.end >= parts.length ? row.end : Math.max(row.start, row.end - 1);
  let used = 0;
  let at = row.start;
  while (at < last) {
    const size = graphemeWidth(parts[at] ?? '');
    if (used + size > column) break;
    used += size;
    at += 1;
  }
  return at;
}

/** Moves the caret one visual row up or down; false at the top or bottom row. */
export function caretRowUpDown(target: Composer, by: -1 | 1, width: number): boolean {
  const parts = graphemes(target.text);
  const rows = rowsOf(target.text, Math.max(1, width - MARGIN));
  const index = caretRow(rows, target.caret);
  const current = rows[index];
  const row = rows[index + by];
  if (!current || !row) return false;
  target.caret = offsetAt(row, caretColumn(current, target.caret, parts), parts);
  return true;
}

/** Renders the composer as rows, scrolled so the caret stays in view. */
export function composerRows(
  target: Composer,
  width: number,
  options: { maxRows: number; blocked: boolean },
): Line[] {
  const parts = graphemes(target.text);
  const rows = rowsOf(target.text, Math.max(1, width - MARGIN));
  const index = caretRow(rows, target.caret);
  const visible = Math.max(1, Math.min(options.maxRows, rows.length));
  const first = Math.max(0, Math.min(index - visible + 1, rows.length - visible));

  const railStyle = styleOf({ bold: true, color: options.blocked ? 'yellow' : 'cyan' });
  const textStyle = styleOf({});
  const caretStyle = styleOf({ reverse: true });

  return rows.slice(first, first + visible).map((row, offset): Line => {
    const segments: Segment[] = [
      { text: options.blocked ? '! ' : '\u203a ', style: railStyle },
    ];
    if (first + offset !== index) {
      if (row.text) segments.push({ text: row.text, style: textStyle });
      return segments;
    }

    const local = Math.max(0, Math.min(target.caret - row.start, row.end - row.start));
    const before = parts.slice(row.start, row.start + local).join('');
    const under = row.start + local < row.end ? (parts[row.start + local] ?? '') : '';
    if (before) segments.push({ text: before, style: textStyle });
    segments.push({ text: under || ' ', style: caretStyle });
    const after = parts.slice(row.start + local + (under ? 1 : 0), row.end).join('');
    if (after) segments.push({ text: after, style: textStyle });
    return segments;
  });
}
