// Folds the renderer-neutral Markdown tree into terminal components.
//
// The grammar lives in `#core/markdown`, shared with the web client. This
// module only chooses terminal primitives: paragraphs wrap, fenced code sits in
// a box, quotes get a left bar, and tables are padded to column widths.

import { isInline, parseMarkdown, plainText, type Block, type InlineRun } from '#core/markdown';
import { Box, Heading, Indent, Line, LineBreak, Paragraph, Row, Stack, Text } from './components.ts';
import { h, type Props, type VNode } from './h.ts';
import { textWidth } from './text.ts';

function propsForRun(run: InlineRun): Props {
  const props: Props = {};
  if (run.bold) props.bold = true;
  if (run.italic) props.italic = true;
  if (run.strike) props.strike = true;
  if (run.code || run.equation) props.color = 'magenta';
  return props;
}

/** Inline runs become spans; links and images are flattened for a terminal. */
function inlineNodes(runs: InlineRun[]): VNode[] {
  const out: VNode[] = [];
  let index = 0;

  while (index < runs.length) {
    const run = runs[index];
    if (run === undefined) break;

    if (run.lineBreak) {
      out.push(LineBreak());
      index += 1;
      continue;
    }

    if (run.image) {
      const group: InlineRun[] = [];
      while (index < runs.length) {
        const current = runs[index];
        if (current === undefined || !current.image || current.href !== run.href) break;
        group.push(current);
        index += 1;
      }
      const alt = plainText(group).trim();
      const href = run.href ?? '';
      out.push(Text(`image: ${alt || href || '?'}`, { dim: true }));
      if (alt !== '' && href !== '' && href !== alt) out.push(Text(` (${href})`, { dim: true, color: 'blue' }));
      continue;
    }

    if (run.href !== null) {
      const group: InlineRun[] = [];
      while (index < runs.length) {
        const current = runs[index];
        if (current === undefined || current.href !== run.href || current.image) break;
        group.push(current);
        index += 1;
      }
      const label = plainText(group);
      if (label === '') out.push(Text(run.href, { color: 'blue' }));
      else {
        for (const item of group) out.push(Text(item.text, { ...propsForRun(item), color: 'blue' }));
        if (label !== run.href) out.push(Text(` (${run.href})`, { dim: true, color: 'blue' }));
      }
      continue;
    }

    out.push(Text(run.text, propsForRun(run)));
    index += 1;
  }

  return out;
}

function inlineContent(children: Array<Block | InlineRun>): VNode[] {
  return inlineNodes(children.filter(isInline));
}

function blockContent(children: Array<Block | InlineRun>): VNode[] {
  const out: VNode[] = [];
  let runs: InlineRun[] = [];
  const flush = (): void => {
    if (runs.length === 0) return;
    out.push(Paragraph(...inlineNodes(runs)));
    runs = [];
  };
  for (const child of children) {
    if (isInline(child)) {
      runs.push(child);
      continue;
    }
    flush();
    out.push(blockNode(child));
  }
  flush();
  return out;
}

function codeBody(block: Block): string[] {
  const text = block.children.length > 0 ? plainText(block.children) : '';
  return text.split('\n');
}

function codeBlock(block: Block): VNode {
  return Box(
    { title: block.lang || 'code' },
    ...codeBody(block).map((line) => Text(line === '' ? ' ' : line, { color: 'gray' })),
  );
}

function markerFor(list: Block, item: Block, index: number): string {
  if (item.checked !== null) return item.checked ? '[x] ' : '[ ] ';
  if (list.start !== null) return `${list.start + index}. `;
  return '• ';
}

function listItemNodes(item: Block, list: Block, index: number): VNode[] {
  const marker = markerFor(list, item, index);
  const markerProps: Props = { dim: true };
  if (item.checked === true) markerProps.color = 'green';
  const markerNode = Text(marker, markerProps);

  const inline = item.children.filter(isInline);
  if (inline.length === item.children.length) {
    return [Paragraph(markerNode, ...inlineNodes(inline))];
  }

  const out: VNode[] = [];
  const indent = ' '.repeat(textWidth(marker));
  const blocks = item.children.filter((child): child is Block => !isInline(child));
  let first = true;

  for (const child of blocks) {
    if (first && child.kind === 'paragraph') {
      out.push(Paragraph(markerNode, ...inlineContent(child.children)));
    } else if (first) {
      out.push(Paragraph(markerNode));
      out.push(Indent(indent, blockNode(child)));
    } else {
      out.push(Indent(indent, blockNode(child)));
    }
    first = false;
  }

  if (out.length === 0) out.push(Paragraph(markerNode));
  return out;
}

function listNode(block: Block): VNode {
  const items = block.children.filter((child): child is Block => !isInline(child) && child.kind === 'list_item');
  return Stack(...items.map((item, index) => Stack(...listItemNodes(item, block, index))));
}

function tableNode(block: Block): VNode {
  const rows = block.children
    .filter((child): child is Block => !isInline(child) && child.kind === 'table_row')
    .map((row) => row.children
      .filter((cell): cell is Block => !isInline(cell) && cell.kind === 'table_cell')
      .map((cell) => plainText(cell.children).replace(/\n/g, ' ')));

  const columns = rows.reduce((max, row) => Math.max(max, row.length), 0);
  if (columns === 0) return Stack();

  const widths: number[] = [];
  for (let column = 0; column < columns; column += 1) {
    widths.push(rows.reduce((max, row) => Math.max(max, textWidth(row[column] ?? '')), 0));
  }

  const renderRow = (cells: string[], header: boolean): VNode => {
    const parts: VNode[] = [];
    for (let column = 0; column < columns; column += 1) {
      const text = cells[column] ?? '';
      const padding = ' '.repeat(Math.max(0, (widths[column] ?? 0) - textWidth(text)));
      parts.push(Text(text + padding, header ? { bold: true } : {}));
      if (column < columns - 1) parts.push(Text(' │ ', { dim: true }));
    }
    return Row(...parts);
  };

  const out: VNode[] = [renderRow(rows[0] ?? [], true)];
  out.push(Text(widths.map((width) => '─'.repeat(width)).join('─┼─'), { dim: true }));
  for (const row of rows.slice(1)) out.push(renderRow(row, false));
  return Stack(...out);
}

function blockNode(block: Block): VNode {
  switch (block.kind) {
    case 'paragraph': {
      const inline = block.children.filter(isInline);
      if (inline.length === block.children.length) return Paragraph(...inlineNodes(inline));
      return Stack(...blockContent(block.children));
    }
    case 'heading': {
      const nodes = inlineContent(block.children);
      return Paragraph(Heading(nodes));
    }
    case 'code':
      return codeBlock(block);
    case 'list':
      return listNode(block);
    case 'list_item':
      return Stack(...listItemNodes(block, { ...block, start: null, checked: null }, 0));
    case 'blockquote':
      return Indent('│ ', ...blockContent(block.children));
    case 'rule':
      return Line();
    case 'table':
      return tableNode(block);
    case 'equation':
      return Box(
        { title: 'math' },
        ...codeBody(block).map((line) => Text(line === '' ? ' ' : line, { color: 'magenta' })),
      );
    case 'table_row':
    case 'table_cell':
      return Stack(...blockContent(block.children));
    default:
      return h('stack', {});
  }
}

const CACHE_LIMIT = 128;
const cache = new Map<string, VNode[]>();

/** Parses and renders a stored or streamed assistant message. */
export function markdownNodes(source: string): VNode[] {
  const cached = cache.get(source);
  if (cached !== undefined) {
    cache.delete(source);
    cache.set(source, cached);
    return cached;
  }
  const nodes = parseMarkdown(source).map(blockNode);
  cache.set(source, nodes);
  if (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  return nodes;
}
