// Layout and paint for the tiny component tree.
//
// Everything is a list of styled segments. Layout flattens the tree into lines
// that never exceed the available width, paint turns lines into ANSI rows, and
// Screen writes only the rows that changed since the previous frame.

import { type Style, type VNode, plain, sanitizeText, styleOf } from './h.ts';
import { PRINTABLE_ASCII, textWidth, truncateToWidth } from './text.ts';

export interface Segment {
  text: string;
  style: Style;
}

export type Line = Segment[];

const COLORS = new Map<string, number>([
  ['red', 31],
  ['green', 32],
  ['yellow', 33],
  ['blue', 34],
  ['magenta', 35],
  ['cyan', 36],
  ['gray', 90],
]);

function sgr(style: Style): string {
  const codes: string[] = ['0'];
  if (style.bold) codes.push('1');
  if (style.dim) codes.push('2');
  if (style.italic) codes.push('3');
  if (style.reverse) codes.push('7');
  if (style.strike) codes.push('9');
  if (style.color) codes.push(String(COLORS.get(style.color)));
  return `\x1b[${codes.join(';')}m`;
}

function lineWidth(line: Line): number {
  let total = 0;
  for (const seg of line) total += textWidth(seg.text);
  return total;
}

export function truncateLine(line: Line, max: number): Line {
  const out: Line = [];
  let used = 0;
  for (const seg of line) {
    if (used >= max) break;
    const room = max - used;
    const width = textWidth(seg.text);
    if (width <= room) {
      out.push(seg);
      used += width;
      continue;
    }
    const cut = truncateToWidth(seg.text, room);
    if (cut !== '') out.push({ text: cut, style: seg.style });
    break;
  }
  return out;
}

function truncateText(value: string, max: number): string {
  return truncateToWidth(sanitizeText(value), max);
}

/* ------------------------------------------------------------------ inline */

function inlineOf(node: VNode, parent: Style = plain): Segment[] {
  const style = styleOf(node.props, parent);
  if (node.tag === 'text') return [{ text: node.text ?? '', style }];
  if (node.tag === 'break') return [{ text: '\n', style }];
  const out: Segment[] = [];
  for (const child of node.children) out.push(...inlineOf(child, style));
  return out;
}

/* ----------------------------------------------------------------- wrapping */

function trimTrailing(line: Line): Line {
  const out = line.slice();
  while (out.length > 0) {
    const last = out[out.length - 1];
    if (!last) break;
    const trimmed = last.text.replace(/\s+$/, '');
    if (trimmed === last.text) break;
    if (trimmed === '') out.pop();
    else {
      out[out.length - 1] = { text: trimmed, style: last.style };
      break;
    }
  }
  return out;
}

function pushWord(line: Line, style: Style, word: string): void {
  const last = line[line.length - 1];
  if (last && last.style === style) last.text += word;
  else line.push({ text: word, style });
}

function wrapSegments(segments: Segment[], max: number): Line[] {
  const widthLimit = Math.max(1, max);
  const lines: Line[] = [];
  let line: Line = [];
  let used = 0;

  const flush = (): void => {
    lines.push(trimTrailing(line));
    line = [];
    used = 0;
  };

  for (const seg of segments) {
    const chunks = seg.text.split('\n');
    chunks.forEach((chunk, chunkIndex) => {
      if (chunkIndex > 0) flush();
      let pos = 0;
      while (pos < chunk.length) {
        // Scan for the next run instead of slicing the rest of the chunk and
        // matching a regex on it: a paragraph is walked once, not once per word.
        const nextSpace = findSpace(chunk, pos);
        if (nextSpace === pos) {
          // Keep real spaces so style boundaries never invent one (the old
          // word-splitting version turned `**bold**,` into `bold ,`).
          const end = findWord(chunk, pos);
          if (used > 0) {
            const spaces = chunk.slice(pos, end);
            pushWord(line, seg.style, spaces);
            used += textWidth(spaces);
          }
          pos = end;
          continue;
        }

        const word = chunk.slice(pos, nextSpace);
        pos = nextSpace;
        const wordWidth = textWidth(word);
        if (wordWidth <= widthLimit - used) {
          pushWord(line, seg.style, word);
          used += wordWidth;
          continue;
        }
        // The word does not fit the rest of the line. Flush, then keep it whole
        // on the fresh line when it fits there.
        if (used > 0) {
          flush();
          if (wordWidth <= widthLimit) {
            pushWord(line, seg.style, word);
            used = wordWidth;
            continue;
          }
        }
        if (PRINTABLE_ASCII.test(word)) {
          // One cell per character, so filling the line is arithmetic.
          for (let index = 0; index < word.length; ) {
            if (used === widthLimit) flush();
            const take = Math.min(widthLimit - used, word.length - index);
            pushWord(line, seg.style, word.slice(index, index + take));
            used += take;
            index += take;
          }
          continue;
        }
        // Break the word across lines, measuring only the head of what is left.
        // Re-measuring the whole tail for every line it spans is what made a
        // long unbroken run quadratic: a 30 KB base64 or a minified JSON blob
        // took seconds.
        let rest = word;
        while (rest !== '') {
          const head = truncateToWidth(rest, widthLimit);
          if (head === rest) {
            pushWord(line, seg.style, rest);
            used = textWidth(rest);
            break;
          }
          // A single cluster wider than the line still has to make progress.
          const first = head === '' ? (rest[Symbol.iterator]().next().value ?? '') : head;
          pushWord(line, seg.style, first);
          flush();
          rest = rest.slice(first.length);
        }
      }
    });
  }
  if (line.length > 0 || lines.length === 0) lines.push(trimTrailing(line));
  return lines;
}

function isSpace(code: number): boolean {
  return code === 0x20 || code === 0x09;
}

/** Index of the first space at or after `from`. */
function findSpace(text: string, from: number): number {
  for (let index = from; index < text.length; index += 1) {
    if (isSpace(text.charCodeAt(index))) return index;
  }
  return text.length;
}

/** Index of the first non-space at or after `from`. */
function findWord(text: string, from: number): number {
  for (let index = from; index < text.length; index += 1) {
    if (!isSpace(text.charCodeAt(index))) return index;
  }
  return text.length;
}

/* ------------------------------------------------------------------ layout */

function layoutRow(node: VNode, maxWidth: number): Line[] {
  interface Part {
    lines: Line[];
    width: number;
    flex: boolean;
  }
  const parts: Part[] = node.children.map((child) => {
    if (child.tag === 'spacer') return { lines: [[]], width: 0, flex: true };
    const lines = layoutNode(child, maxWidth);
    const width = lines.reduce((max, line) => Math.max(max, lineWidth(line)), 0);
    return { lines, width, flex: false };
  });

  const fixed = parts.filter((part) => !part.flex).reduce((sum, part) => sum + part.width, 0);
  const flexParts = parts.filter((part) => part.flex);
  const share = flexParts.length > 0 ? Math.floor(Math.max(0, maxWidth - fixed) / flexParts.length) : 0;
  const widths = parts.map((part) => (part.flex ? share : part.width));
  const height = parts.reduce((max, part) => Math.max(max, part.lines.length), 1);

  const lines: Line[] = [];
  for (let row = 0; row < height; row += 1) {
    const line: Line = [];
    parts.forEach((part, index) => {
      const width = widths[index] ?? 0;
      const source = part.lines[row] ?? [];
      const sourceWidth = lineWidth(source);
      line.push(...source);
      if (sourceWidth < width) line.push({ text: ' '.repeat(width - sourceWidth), style: plain });
    });
    lines.push(truncateLine(line, maxWidth));
  }
  return lines;
}

function layoutBox(node: VNode, maxWidth: number): Line[] {
  const width = Math.max(4, Math.min(node.props.width ?? maxWidth, maxWidth));
  const inner = width - 2;
  const border = styleOf({ dim: true });
  const content: Line[] = [];
  for (const child of node.children) content.push(...layoutNode(child, inner));

  const title = truncateText(node.props.title ?? '', Math.max(0, width - 5));
  const head = title ? ` ${title} ` : '';
  const fill = Math.max(0, width - 2 - textWidth(head) - (title ? 1 : 0));
  const lines: Line[] = [
    [
      { text: '┌─', style: border },
      { text: head, style: styleOf({ bold: true }) },
      { text: '─'.repeat(fill), style: border },
      { text: '┐', style: border },
    ],
  ];
  for (const line of content) {
    const used = lineWidth(line);
    lines.push([
      { text: '│', style: border },
      ...line,
      { text: ' '.repeat(Math.max(0, inner - used)), style: plain },
      { text: '│', style: border },
    ]);
  }
  if (content.length === 0) {
    lines.push([{ text: '│', style: border }, { text: ' '.repeat(inner), style: plain }, { text: '│', style: border }]);
  }
  lines.push([{ text: `└${'─'.repeat(width - 2)}┘`, style: border }]);
  return lines;
}

function layoutIndent(node: VNode, maxWidth: number): Line[] {
  const prefix = sanitizeText(node.props.prefix ?? '  ');
  const prefixWidth = textWidth(prefix);
  const inner = Math.max(1, maxWidth - prefixWidth);
  const content: Line[] = [];
  for (const child of node.children) content.push(...layoutNode(child, inner));
  if (content.length === 0) content.push([]);
  const style = styleOf({ dim: true });
  return content.map((line) => [{ text: prefix, style }, ...line]);
}

function layoutNode(node: VNode, maxWidth: number): Line[] {
  switch (node.tag) {
    case 'row':
      return layoutRow(node, maxWidth);
    case 'stack': {
      const lines: Line[] = [];
      for (const child of node.children) lines.push(...layoutNode(child, maxWidth));
      return lines;
    }
    case 'indent':
      return layoutIndent(node, maxWidth);
    case 'break':
      return [[]];
    case 'p':
      return wrapSegments(inlineOf(node), maxWidth);
    case 'span':
      return [truncateLine(inlineOf(node), maxWidth)];
    case 'text':
      return [truncateLine(inlineOf(node), maxWidth)];
    case 'line':
      return [[{ text: '─'.repeat(Math.max(1, maxWidth)), style: styleOf({ dim: true }) }]];
    case 'box':
      return layoutBox(node, maxWidth);
    case 'spacer':
      return [[]];
    default:
      return [];
  }
}

export function layout(node: VNode, maxWidth: number): Line[] {
  return layoutNode(node, maxWidth);
}

/* ------------------------------------------------------------------- paint */

export function paintLine(line: Line, width: number): string {
  let out = '';
  let used = 0;
  for (const seg of truncateLine(line, width)) {
    if (!seg.text) continue;
    out += sgr(seg.style) + seg.text;
    used += textWidth(seg.text);
  }
  if (used < width) out += ' '.repeat(width - used);
  return `${out}\x1b[0m`;
}

/** Renders exactly `height` rows when it is given: extra lines are dropped, short frames padded. */
export function paint(lines: Line[], width: number, height?: number): string[] {
  const selected = height === undefined ? lines : lines.slice(0, Math.max(0, height));
  const rows = selected.map((line) => paintLine(line, width));
  if (height !== undefined) {
    for (let index = rows.length; index < height; index += 1) rows.push(' '.repeat(width));
  }
  return rows;
}

/* ------------------------------------------------------------------ screen */

// Clear the line before writing it: with the cursor on the last column an
// erase-after-write would eat the final character of a full-width row.
function writeRow(index: number, row: string): string {
  return `\x1b[${index + 1};1H\x1b[2K${row}`;
}

export class Screen {
  private previous: string[] = [];
  private active = false;
  private readonly out: NodeJS.WriteStream;

  constructor(out: NodeJS.WriteStream) {
    this.out = out;
  }

  enter(): void {
    if (this.active) return;
    this.active = true;
    // Full-width rows would otherwise trip autowrap on the last column.
    this.out.write('\x1b[?1049h\x1b[?25l\x1b[?7l');
  }

  leave(): void {
    if (!this.active) return;
    this.active = false;
    this.out.write('\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l');
  }

  draw(rows: string[]): void {
    if (!this.active) {
      this.out.write(`${rows.join('\n')}\n`);
      this.previous = rows;
      return;
    }
    const lengthChanged = rows.length !== this.previous.length;
    let out = '';
    if (lengthChanged) {
      out += '\x1b[2J';
      rows.forEach((row, index) => {
        out += writeRow(index, row);
      });
      for (let index = rows.length; index < this.previous.length; index += 1) {
        out += `\x1b[${index + 1};1H\x1b[2K`;
      }
    } else {
      rows.forEach((row, index) => {
        if (row === this.previous[index]) return;
        out += writeRow(index, row);
      });
    }
    this.out.write(out);
    this.previous = rows;
  }
}
