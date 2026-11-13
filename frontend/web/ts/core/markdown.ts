// Markdown -> a renderer-neutral tree, with no third-party dependency.
//
// The web client and the TUI must show the same Markdown, so the grammar lives
// here once and each client folds the resulting tree into its own primitives
// (DOM elements in the browser, styled segments in the terminal). Nothing here
// touches the DOM, so the module runs unchanged in Node, Deno and the browser.
//
// Scope is the CommonMark/GFM subset that model output actually uses: ATX and
// setext headings, paragraphs, fenced code, blockquotes, ordered/unordered and
// task lists, GFM tables, thematic breaks, and inline code, emphasis, strike,
// links, images, autolinks and backslash escapes. Raw HTML is not interpreted;
// it stays literal text and the clients sanitize it before display.
//
// Streaming: MarkdownStream accumulates text and re-parses lazily in blocks().
// A message is small and parsing is linear, so memoizing the last result keeps
// this cheap while leaving the parser stateless and easy to reason about.

/* ---------------------------------------------------------------- the tree */

export type BlockKind =
  | 'paragraph'
  | 'heading'
  | 'code'
  | 'list'
  | 'list_item'
  | 'blockquote'
  | 'rule'
  | 'table'
  | 'table_row'
  | 'table_cell'
  | 'equation';

/** A run of text with one inline style. Adjacent runs may share a style. */
export interface InlineRun {
  text: string;
  bold: boolean;
  italic: boolean;
  strike: boolean;
  code: boolean;
  equation: boolean;
  /** Link/image target when this run belongs to a link or an image. */
  href: string | null;
  image: boolean;
  /** A hard line break inside a block; `text` is then ''. */
  lineBreak: boolean;
}

/** A block node. Inline content lives in `children` together with nested blocks. */
export interface Block {
  kind: BlockKind;
  /** Heading level 1..6; 0 for every other kind. */
  level: number;
  /** Code fence language; '' when the fence carried none. */
  lang: string;
  /** Ordered-list start number; null for unordered lists and non-lists. */
  start: number | null;
  /** Task-list checkbox state; null when the item has no checkbox. */
  checked: boolean | null;
  children: Array<Block | InlineRun>;
}

export function isInline(node: Block | InlineRun): node is InlineRun {
  return 'text' in node;
}

function newBlock(kind: BlockKind): Block {
  return { kind, level: 0, lang: '', start: null, checked: null, children: [] };
}

function newRun(text: string): InlineRun {
  return {
    text,
    bold: false,
    italic: false,
    strike: false,
    code: false,
    equation: false,
    href: null,
    image: false,
    lineBreak: false,
  };
}

function lineBreakRun(): InlineRun {
  const run = newRun('');
  run.lineBreak = true;
  return run;
}

function sameStyle(a: InlineRun, b: InlineRun): boolean {
  return (
    a.bold === b.bold &&
    a.italic === b.italic &&
    a.strike === b.strike &&
    a.code === b.code &&
    a.equation === b.equation &&
    a.href === b.href &&
    a.image === b.image &&
    a.lineBreak === b.lineBreak
  );
}

function appendRun(out: InlineRun[], run: InlineRun): void {
  if (run.text === '' && !run.lineBreak) return;
  const last = out[out.length - 1];
  if (last && !last.lineBreak && !run.lineBreak && sameStyle(last, run)) {
    last.text += run.text;
    return;
  }
  out.push(run);
}

function mergeRuns(runs: InlineRun[]): InlineRun[] {
  const out: InlineRun[] = [];
  for (const run of runs) appendRun(out, run);
  return out;
}

function styleRuns(runs: InlineRun[], patch: Partial<InlineRun>): InlineRun[] {
  return mergeRuns(runs.map((run) => ({ ...run, ...patch })));
}

/* -------------------------------------------------------------- line helpers */

function isBlank(line: string): boolean {
  return /^[ \t]*$/.test(line);
}

function leadingSpaces(line: string): number {
  let count = 0;
  while (line[count] === ' ') count += 1;
  return count;
}

function stripIndent(line: string, count: number): string {
  let cut = 0;
  while (cut < count && line[cut] === ' ') cut += 1;
  return line.slice(cut);
}

/**
 * Removes up to three leading spaces (block content indentation). Trailing
 * spaces are preserved on purpose: two of them before a newline mean a hard
 * line break, and inline parsing needs to see them.
 */
function stripBlockIndent(line: string): string {
  return line.replace(/^ {0,3}/, '');
}

/* -------------------------------------------------------------- block markers */

interface Fence {
  char: string;
  length: number;
  lang: string;
}

function matchFence(line: string): Fence | null {
  const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
  if (!match) return null;
  const marker = match[1] ?? '';
  const info = (match[2] ?? '').trim();
  if (marker.startsWith('`') && info.includes('`')) return null;
  return { char: marker[0] ?? '`', length: marker.length, lang: info.split(/\s+/)[0] ?? '' };
}

function matchAtx(line: string): { level: number; text: string } | null {
  const match = /^ {0,3}(#{1,6})(?:[ \t]+(.*))?$/.exec(line);
  if (!match) return null;
  const marker = match[1] ?? '#';
  const text = (match[2] ?? '').replace(/[ \t]+#+[ \t]*$/, '').trim();
  return { level: marker.length, text };
}

function isThematicBreak(line: string): boolean {
  const trimmed = line.trim();
  if (trimmed.length < 3) return false;
  return (
    /^(?:\*[ \t]*){3,}$/.test(trimmed) ||
    /^(?:-[ \t]*){3,}$/.test(trimmed) ||
    /^(?:_[ \t]*){3,}$/.test(trimmed)
  );
}

function isSetextUnderline(line: string): boolean {
  return /^ {0,3}(=+|-+)[ \t]*$/.test(line);
}

interface ListMarker {
  ordered: boolean;
  number: number;
  indent: number;
  contentIndent: number;
  content: string;
}

function matchListMarker(line: string): ListMarker | null {
  const match = /^( *)([-*+]|\d{1,9}[.)])( +|$)(.*)$/.exec(line);
  if (!match) return null;
  const indent = (match[1] ?? '').length;
  const marker = match[2] ?? '-';
  const spaces = (match[3] ?? '').length;
  const ordered = /^\d/.test(marker);
  return {
    ordered,
    number: ordered ? Number.parseInt(marker, 10) : 0,
    indent,
    contentIndent: indent + marker.length + (spaces > 0 ? spaces : 1),
    content: match[4] ?? '',
  };
}

/* ------------------------------------------------------------------- tables */

function splitTableRow(line: string): string[] {
  let body = line.trim();
  if (body.startsWith('|')) body = body.slice(1);
  if (body.endsWith('|')) body = body.slice(0, -1);
  const cells: string[] = [];
  let current = '';
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    if (char === '\\' && body[index + 1] === '|') {
      current += '|';
      index += 1;
      continue;
    }
    if (char === '|') {
      cells.push(current.trim());
      current = '';
      continue;
    }
    current += char ?? '';
  }
  cells.push(current.trim());
  return cells;
}

function isTableDelimiter(line: string): boolean {
  if (!line.includes('|')) return false;
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-+:?$/.test(cell));
}

function isTableStart(lines: string[], index: number): boolean {
  const line = lines[index];
  const next = lines[index + 1];
  if (line === undefined || next === undefined) return false;
  return line.includes('|') && isTableDelimiter(next);
}

/* -------------------------------------------------------------- block parse */

function makeParagraph(text: string): Block {
  const node = newBlock('paragraph');
  node.children = parseInline(text);
  return node;
}

function makeHeading(level: number, text: string): Block {
  const node = newBlock('heading');
  node.level = level;
  node.children = parseInline(text);
  return node;
}

function tableRow(cells: string[]): Block {
  const row = newBlock('table_row');
  for (const cell of cells) {
    const node = newBlock('table_cell');
    node.children = parseInline(cell);
    row.children.push(node);
  }
  return row;
}

function startsBlock(lines: string[], index: number): boolean {
  const line = lines[index] ?? '';
  if (matchFence(line)) return true;
  if (/^ {0,3}\$\$/.test(line)) return true;
  if (isThematicBreak(line)) return true;
  if (matchAtx(line)) return true;
  if (/^ {0,3}>/.test(line)) return true;
  if (matchListMarker(line)) return true;
  if (isTableStart(lines, index)) return true;
  return false;
}

function parseParagraph(lines: string[], start: number, blocks: Block[]): number {
  const parts: string[] = [];
  let index = start;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (isBlank(line)) break;
    if (index > start && startsBlock(lines, index)) break;
    parts.push(stripBlockIndent(line));
    index += 1;
  }
  blocks.push(makeParagraph(parts.join('\n')));
  return index;
}

function parseFence(lines: string[], start: number, fence: Fence, blocks: Block[]): number {
  const code: string[] = [];
  let index = start + 1;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    const close = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
    if (close && (close[1] ?? '').startsWith(fence.char) && (close[1] ?? '').length >= fence.length) {
      index += 1;
      break;
    }
    code.push(line);
    index += 1;
  }
  const node = newBlock('code');
  node.lang = fence.lang;
  node.children = [newRun(code.join('\n'))];
  blocks.push(node);
  return index;
}

function parseEquation(lines: string[], start: number, blocks: Block[]): number {
  const body: string[] = [];
  let index = start + 1;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (/^ {0,3}\$\$[ \t]*$/.test(line)) {
      index += 1;
      break;
    }
    body.push(line);
    index += 1;
  }
  const node = newBlock('equation');
  node.children = [newRun(body.join('\n'))];
  blocks.push(node);
  return index;
}

function parseBlockquote(lines: string[], start: number, blocks: Block[]): number {
  const inner: string[] = [];
  let index = start;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    const match = /^ {0,3}> ?(.*)$/.exec(line);
    if (match) {
      inner.push(match[1] ?? '');
      index += 1;
      continue;
    }
    break;
  }
  const node = newBlock('blockquote');
  node.children = parseBlocks(inner);
  blocks.push(node);
  return index;
}

function parseTable(lines: string[], start: number, blocks: Block[]): number {
  const header = splitTableRow(lines[start] ?? '');
  const columns = header.length;
  const node = newBlock('table');
  node.children.push(tableRow(header));
  let index = start + 2;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (isBlank(line) || !line.includes('|')) break;
    const cells = splitTableRow(line);
    while (cells.length < columns) cells.push('');
    node.children.push(tableRow(cells.slice(0, columns)));
    index += 1;
  }
  blocks.push(node);
  return index;
}

function parseList(lines: string[], start: number, blocks: Block[]): number {
  const first = matchListMarker(lines[start] ?? '');
  if (!first) return start + 1;
  const baseIndent = first.indent;
  const ordered = first.ordered;
  const list = newBlock('list');
  list.start = ordered ? first.number : null;

  let index = start;
  while (index < lines.length) {
    const marker = matchListMarker(lines[index] ?? '');
    if (!marker || marker.indent !== baseIndent || marker.ordered !== ordered) break;

    const itemLines: string[] = [marker.content];
    const contentIndent = marker.contentIndent;
    index += 1;

    while (index < lines.length) {
      const line = lines[index] ?? '';
      if (isBlank(line)) {
        let next = index + 1;
        while (next < lines.length && isBlank(lines[next] ?? '')) next += 1;
        if (next >= lines.length) {
          index = next;
          break;
        }
        const peek = matchListMarker(lines[next] ?? '');
        const indented = leadingSpaces(lines[next] ?? '') >= contentIndent;
        if ((peek && peek.indent === baseIndent) || indented) {
          itemLines.push('');
          index += 1;
          continue;
        }
        index = next;
        break;
      }
      const sibling = matchListMarker(line);
      if (sibling && sibling.indent === baseIndent) break;
      if (leadingSpaces(line) >= contentIndent) {
        itemLines.push(stripIndent(line, contentIndent));
        index += 1;
        continue;
      }
      if (sibling) break;
      itemLines.push(stripBlockIndent(line));
      index += 1;
    }

    const item = newBlock('list_item');
    const task = /^\[([ xX])\][ \t]+(.*)$/.exec(itemLines[0] ?? '');
    if (task) {
      item.checked = (task[1] ?? '').toLowerCase() === 'x';
      itemLines[0] = task[2] ?? '';
    }
    const parsed = parseBlocks(itemLines);
    // A tight item is a single paragraph; keep its inline runs directly so the
    // tree stays flat for the common case.
    item.children = parsed.length === 1 && parsed[0]?.kind === 'paragraph'
      ? parsed[0].children
      : parsed;
    list.children.push(item);
  }

  blocks.push(list);
  return index;
}

function parseBlocks(lines: string[]): Block[] {
  const blocks: Block[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (isBlank(line)) {
      index += 1;
      continue;
    }

    const fence = matchFence(line);
    if (fence) {
      index = parseFence(lines, index, fence, blocks);
      continue;
    }
    const singleEquation = /^ {0,3}\$\$(.*?)\$\$[ \t]*$/.exec(line);
    if (singleEquation) {
      const node = newBlock('equation');
      node.children = [newRun((singleEquation[1] ?? '').trim())];
      blocks.push(node);
      index += 1;
      continue;
    }
    if (/^ {0,3}\$\$/.test(line)) {
      index = parseEquation(lines, index, blocks);
      continue;
    }
    if (isThematicBreak(line)) {
      blocks.push(newBlock('rule'));
      index += 1;
      continue;
    }
    const atx = matchAtx(line);
    if (atx) {
      blocks.push(makeHeading(atx.level, atx.text));
      index += 1;
      continue;
    }
    if (/^ {0,3}>/.test(line)) {
      index = parseBlockquote(lines, index, blocks);
      continue;
    }
    if (isTableStart(lines, index)) {
      index = parseTable(lines, index, blocks);
      continue;
    }
    if (matchListMarker(line)) {
      index = parseList(lines, index, blocks);
      continue;
    }
    const underline = lines[index + 1];
    if (underline !== undefined && isSetextUnderline(underline)) {
      const level = underline.trim().startsWith('=') ? 1 : 2;
      blocks.push(makeHeading(level, line.trim()));
      index += 2;
      continue;
    }
    index = parseParagraph(lines, index, blocks);
  }
  return blocks;
}

/* ------------------------------------------------------------- inline parse */

const ESCAPABLE = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/;

function isLeftFlanking(text: string, start: number, length: number, char: string): boolean {
  const before = start > 0 ? text[start - 1] ?? '' : '';
  const after = text[start + length] ?? '';
  if (after === '' || /\s/.test(after)) return false;
  if (char === '_' && before !== '' && /[A-Za-z0-9]/.test(before)) return false;
  return true;
}

function isRightFlanking(text: string, start: number, length: number, char: string): boolean {
  const before = start > 0 ? text[start - 1] ?? '' : '';
  const after = text[start + length] ?? '';
  if (before === '' || /\s/.test(before)) return false;
  if (char === '_' && after !== '' && /[A-Za-z0-9]/.test(after)) return false;
  return true;
}

function findCloser(text: string, from: number, char: string, length: number): number {
  let index = from;
  while (index < text.length) {
    const current = text[index] ?? '';
    if (current === '\\') {
      index += 2;
      continue;
    }
    if (current === char) {
      let end = index;
      while (text[end] === char) end += 1;
      if (end - index === length && isRightFlanking(text, index, end - index, char)) return index;
      index = end;
      continue;
    }
    index += 1;
  }
  return -1;
}

interface InlineResult {
  runs: InlineRun[];
  next: number;
}

function parseEmphasis(text: string, start: number): InlineResult | null {
  const char = text[start] ?? '';
  let length = 1;
  while (text[start + length] === char) length += 1;

  if (char === '~') {
    if (length < 2) return null;
    const close = findCloser(text, start + 2, '~', 2);
    if (close < 0) return null;
    const inner = parseInline(text.slice(start + 2, close));
    return { runs: styleRuns(inner, { strike: true }), next: close + 2 };
  }

  if (!isLeftFlanking(text, start, length, char)) return null;
  const close = findCloser(text, start + length, char, length);
  if (close < 0) return null;
  const inner = parseInline(text.slice(start + length, close));
  const style: Partial<InlineRun> = {};
  if (length >= 2) style.bold = true;
  if (length % 2 === 1) style.italic = true;
  return { runs: styleRuns(inner, style), next: close + length };
}

function parseCodeSpan(text: string, start: number): { run: InlineRun; next: number } | null {
  let openEnd = start;
  while (text[openEnd] === '`') openEnd += 1;
  const runLength = openEnd - start;
  let index = openEnd;
  while (index < text.length) {
    if (text[index] === '`') {
      let end = index;
      while (text[end] === '`') end += 1;
      if (end - index === runLength) {
        let content = text.slice(openEnd, index).replace(/\n/g, ' ');
        if (content.length >= 2 && content.startsWith(' ') && content.endsWith(' ') && content.trim() !== '') {
          content = content.slice(1, -1);
        }
        const run = newRun(content);
        run.code = true;
        return { run, next: end };
      }
      index = end;
      continue;
    }
    index += 1;
  }
  return null;
}

/**
 * Inline maths: `$x + y$`. A `$` only opens a span when it touches the body on
 * the right, and only closes it when it touches the body on the left, so
 * prices such as `$5 and $6` stay literal text.
 */
function parseInlineEquation(text: string, start: number): { run: InlineRun; next: number } | null {
  if (text[start] !== '$' || text[start + 1] === '$') return null;
  const first = text[start + 1];
  if (first === undefined || /\s/.test(first)) return null;

  let index = start + 1;
  while (index < text.length) {
    const char = text[index] ?? '';
    if (char === '\\') {
      index += 2;
      continue;
    }
    if (char === '\n') return null;
    if (char === '$') {
      if (index === start + 1) return null;
      const before = text[index - 1] ?? '';
      if (before === '' || /\s/.test(before)) return null;
      const run = newRun(text.slice(start + 1, index));
      run.equation = true;
      return { run, next: index + 1 };
    }
    index += 1;
  }
  return null;
}

/**
 * Extracts the URL from a link destination, ignoring an optional title:
 * `u`, `<u>`, `u "t"`, `u 't'` and `u (t)` all yield `u`.
 */
function splitDestination(raw: string): string {
  const angle = /^<([^>]*)>/.exec(raw);
  if (angle) return unescapeBackslashes(angle[1] ?? '');
  const bare = /^(?:\\.|[^\s\\])+/.exec(raw);
  return bare ? unescapeBackslashes(bare[0]) : '';
}

function unescapeBackslashes(value: string): string {
  return value.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, '$1');
}

function parseLink(text: string, start: number, image: boolean): InlineResult | null {
  const open = image ? start + 1 : start;
  if (text[open] !== '[') return null;

  let depth = 0;
  let close = open;
  for (; close < text.length; close += 1) {
    const char = text[close] ?? '';
    if (char === '\\') {
      close += 1;
      continue;
    }
    if (char === '[') depth += 1;
    else if (char === ']') {
      depth -= 1;
      if (depth === 0) break;
    }
  }
  if (close >= text.length || text[close + 1] !== '(') return null;

  let paren = 1;
  let end = close + 2;
  for (; end < text.length; end += 1) {
    const char = text[end] ?? '';
    if (char === '\\') {
      end += 1;
      continue;
    }
    if (char === '(') paren += 1;
    else if (char === ')') {
      paren -= 1;
      if (paren === 0) break;
    }
  }
  if (end >= text.length) return null;

  const href = splitDestination(text.slice(close + 2, end).trim());
  const label = text.slice(open + 1, close);
  const inner = parseInline(label);
  const runs = inner.length > 0 ? inner : [newRun(label)];
  for (const run of runs) {
    run.href = href;
    if (image) run.image = true;
  }
  return { runs: mergeRuns(runs), next: end + 1 };
}

function parseAutolink(text: string, start: number): { run: InlineRun; next: number } | null {
  const match = /^<((?:https?:\/\/|mailto:)[^ >]+|[^ >@]+@[^ >@]+)>/.exec(text.slice(start));
  if (!match) return null;
  const target = match[1] ?? '';
  const href = target.includes('@') && !target.startsWith('mailto:') ? `mailto:${target}` : target;
  const run = newRun(target);
  run.href = href;
  return { run, next: start + match[0].length };
}

function parseBareUrl(text: string, start: number): { run: InlineRun; next: number } | null {
  const match = /^https?:\/\/[^\s<>]+/i.exec(text.slice(start));
  if (!match) return null;
  let url = match[0].replace(/[.,;:!?]+$/, '');
  while (url.endsWith(')') && (url.match(/\(/g)?.length ?? 0) < (url.match(/\)/g)?.length ?? 0)) {
    url = url.slice(0, -1);
  }
  if (url === '') return null;
  const run = newRun(url);
  run.href = url;
  return { run, next: start + url.length };
}

function parseInline(text: string): InlineRun[] {
  const out: InlineRun[] = [];
  let buffer = '';
  let index = 0;

  const flush = (): void => {
    if (buffer === '') return;
    appendRun(out, newRun(buffer));
    buffer = '';
  };

  while (index < text.length) {
    const char = text[index] ?? '';

    if (char === '\\') {
      const next = text[index + 1];
      if (next === '\n') {
        buffer = buffer.replace(/ +$/, '');
        flush();
        appendRun(out, lineBreakRun());
        index += 2;
        continue;
      }
      if (next !== undefined && ESCAPABLE.test(next)) {
        buffer += next;
        index += 2;
        continue;
      }
      buffer += '\\';
      index += 1;
      continue;
    }

    if (char === '\n') {
      if (/ {2,}$/.test(buffer)) {
        buffer = buffer.replace(/ +$/, '');
        flush();
        appendRun(out, lineBreakRun());
      } else {
        buffer = buffer.replace(/ +$/, '');
        flush();
        appendRun(out, newRun(' '));
      }
      index += 1;
      continue;
    }

    if (char === '$') {
      const equation = parseInlineEquation(text, index);
      if (equation) {
        flush();
        appendRun(out, equation.run);
        index = equation.next;
        continue;
      }
      buffer += '$';
      index += 1;
      continue;
    }

    if (char === '`') {
      const span = parseCodeSpan(text, index);
      if (span) {
        flush();
        appendRun(out, span.run);
        index = span.next;
        continue;
      }
      let end = index;
      while (text[end] === '`') end += 1;
      buffer += text.slice(index, end);
      index = end;
      continue;
    }

    if (char === '!' && text[index + 1] === '[') {
      const link = parseLink(text, index, true);
      if (link) {
        flush();
        for (const run of link.runs) appendRun(out, run);
        index = link.next;
        continue;
      }
      buffer += '!';
      index += 1;
      continue;
    }

    if (char === '[') {
      const link = parseLink(text, index, false);
      if (link) {
        flush();
        for (const run of link.runs) appendRun(out, run);
        index = link.next;
        continue;
      }
      buffer += '[';
      index += 1;
      continue;
    }

    if (char === '<') {
      const auto = parseAutolink(text, index);
      if (auto) {
        flush();
        appendRun(out, auto.run);
        index = auto.next;
        continue;
      }
      buffer += '<';
      index += 1;
      continue;
    }

    if ((char === 'h' || char === 'H') && /^https?:\/\//i.test(text.slice(index))) {
      const before = index > 0 ? text[index - 1] ?? '' : '';
      if (before === '' || !/[A-Za-z0-9]/.test(before)) {
        const url = parseBareUrl(text, index);
        if (url) {
          flush();
          appendRun(out, url.run);
          index = url.next;
          continue;
        }
      }
    }

    if (char === '*' || char === '_' || char === '~') {
      const emphasis = parseEmphasis(text, index);
      if (emphasis) {
        flush();
        for (const run of emphasis.runs) appendRun(out, run);
        index = emphasis.next;
        continue;
      }
      buffer += char;
      index += 1;
      continue;
    }

    buffer += char;
    index += 1;
  }

  flush();
  return out;
}

/* ------------------------------------------------------------------ public */

/** Flattens a run of inline nodes (or blocks) back to unformatted text. */
export function plainText(nodes: Array<Block | InlineRun>): string {
  let out = '';
  for (const node of nodes) {
    if (isInline(node)) out += node.lineBreak ? '\n' : node.text;
    else out += plainText(node.children);
  }
  return out;
}

/** Parses a complete Markdown string in one go. */
export function parseMarkdown(source: string): Block[] {
  const text = source.replace(/\r\n?/g, '\n');
  return parseBlocks(text.split('\n'));
}

/**
 * Accumulates streamed Markdown and exposes the parsed tree.
 *
 *   const md = new MarkdownStream();
 *   md.write('# hi');
 *   md.end();
 *   md.blocks(); // [{ kind: 'heading', level: 1, children: [{ text: 'hi' }] }]
 *
 * `blocks()` re-parses the accumulated source and memoizes the result, so
 * repeated reads without a write are free and partial input renders as soon as
 * it is written.
 */
export class MarkdownStream {
  private source = '';
  private cachedKey: string | null = null;
  private cached: Block[] = [];

  write(chunk: string): void {
    if (!chunk) return;
    this.source += chunk;
  }

  /** Replaces the accumulated source, for edits and stored-message reloads. */
  reset(source: string): void {
    this.source = source;
  }

  end(): void {
    // Kept for API symmetry with the old streaming parser; nothing to flush.
  }

  blocks(): Block[] {
    if (this.cachedKey !== this.source) {
      this.cached = parseMarkdown(this.source);
      this.cachedKey = this.source;
    }
    return this.cached;
  }
}
