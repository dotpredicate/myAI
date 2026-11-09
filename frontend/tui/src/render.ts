// Layout and paint for the tiny component tree.
//
// Everything is a list of styled segments. Layout flattens the tree into lines
// that never exceed the available width, paint turns lines into ANSI rows, and
// Screen writes only the rows that changed since the previous frame.

import { type Style, type VNode, plain, sanitizeText, styleOf } from './h.ts';

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

export function textWidth(value: string): number {
  // Code points, not graphemes: good enough for a terminal layout shortcut.
  return Array.from(value).length;
}

function sgr(style: Style): string {
  const codes: string[] = ['0'];
  if (style.bold) codes.push('1');
  if (style.dim) codes.push('2');
  if (style.reverse) codes.push('7');
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
    const chars = Array.from(seg.text);
    if (chars.length <= room) {
      out.push(seg);
      used += chars.length;
    } else {
      out.push({ text: chars.slice(0, room).join(''), style: seg.style });
      break;
    }
  }
  return out;
}

function truncateText(value: string, max: number): string {
  const chars = Array.from(sanitizeText(value));
  return chars.length <= max ? value : chars.slice(0, max).join('');
}

/* ------------------------------------------------------------------ inline */

function inlineOf(node: VNode, parent: Style = plain): Segment[] {
  const style = styleOf(node.props, parent);
  if (node.tag === 'text') return [{ text: node.text ?? '', style }];
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
      for (const word of chunk.split(' ')) {
        if (word === '') continue;
        let rest = word;
        for (;;) {
          const gap = used > 0 ? 1 : 0;
          if (used + gap + textWidth(rest) <= widthLimit) {
            if (gap === 1) {
              pushWord(line, seg.style, ' ');
              used += 1;
            }
            pushWord(line, seg.style, rest);
            used += textWidth(rest);
            break;
          }
          if (used === 0) {
            const chars = Array.from(rest);
            pushWord(line, seg.style, chars.slice(0, widthLimit).join(''));
            flush();
            rest = chars.slice(widthLimit).join('');
            if (rest === '') break;
            continue;
          }
          flush();
        }
      }
    });
  }
  if (line.length > 0 || lines.length === 0) lines.push(trimTrailing(line));
  return lines;
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

function layoutNode(node: VNode, maxWidth: number): Line[] {
  switch (node.tag) {
    case 'row':
      return layoutRow(node, maxWidth);
    case 'stack': {
      const lines: Line[] = [];
      for (const child of node.children) lines.push(...layoutNode(child, maxWidth));
      return lines;
    }
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
