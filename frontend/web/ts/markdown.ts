// Folds the renderer-neutral Markdown tree into DOM nodes.
//
// The grammar lives in `core/markdown.ts` and is shared with the terminal
// client. This module only maps blocks to elements, so raw HTML never reaches
// the browser: text is always appended as text nodes, and link targets are
// restricted to safe schemes.

import { el } from './dom.js';
import { MarkdownStream, isInline, type Block, type InlineRun } from './core/markdown.js';

const SAFE_SCHEME = /^(?:https?:\/\/|mailto:)/i;
const ANY_SCHEME = /^[a-z][a-z0-9+.-]*:/i;

/**
 * Returns a URL the browser may navigate to, or null for anything that could
 * execute script. Relative paths and fragments are allowed; every other
 * scheme (javascript:, data:, vbscript:, …) is dropped.
 */
function safeHref(value: string | null): string | null {
  if (value === null) return null;
  const href = value.trim();
  if (href === '') return null;
  if (SAFE_SCHEME.test(href)) return href;
  if (ANY_SCHEME.test(href)) return null;
  return href;
}

function styledRun(run: InlineRun): Node {
  let node: Node = document.createTextNode(run.text);
  if (run.code) node = el('code', {}, node);
  if (run.equation) node = el('code', { class: 'equation' }, node);
  if (run.strike) node = el('del', {}, node);
  if (run.italic) node = el('em', {}, node);
  if (run.bold) node = el('strong', {}, node);
  return node;
}

function imageAlt(runs: InlineRun[]): string {
  let alt = '';
  for (const run of runs) alt += run.lineBreak ? ' ' : run.text;
  return alt;
}

/**
 * Appends inline runs, grouping a link's (possibly styled) label into one
 * anchor and an image's label into one `img`.
 */
function appendInline(parent: Node, runs: InlineRun[]): void {
  let index = 0;
  while (index < runs.length) {
    const run = runs[index];
    if (run === undefined) break;

    if (run.lineBreak) {
      parent.appendChild(el('br'));
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
      const href = safeHref(run.href);
      const alt = imageAlt(group);
      if (href !== null) parent.appendChild(el('img', { src: href, alt, class: 'markdown-image' }));
      else parent.appendChild(document.createTextNode(alt));
      continue;
    }

    if (run.href !== null) {
      const href = safeHref(run.href);
      const group: InlineRun[] = [];
      while (index < runs.length) {
        const current = runs[index];
        if (current === undefined || current.href !== run.href || current.image) break;
        group.push(current);
        index += 1;
      }
      if (href === null) {
        for (const item of group) parent.appendChild(styledRun(item));
        continue;
      }
      const anchor = el('a', { href, target: '_blank', rel: 'noopener noreferrer' });
      for (const item of group) anchor.appendChild(styledRun(item));
      parent.appendChild(anchor);
      continue;
    }

    parent.appendChild(styledRun(run));
    index += 1;
  }
}

function appendChildren(parent: Node, children: Array<Block | InlineRun>): void {
  const inline: InlineRun[] = [];
  const flush = (): void => {
    if (inline.length === 0) return;
    appendInline(parent, inline.splice(0));
  };
  for (const child of children) {
    if (isInline(child)) {
      inline.push(child);
      continue;
    }
    flush();
    parent.appendChild(renderBlock(child));
  }
  flush();
}

function headingTag(level: number): 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' {
  switch (level) {
    case 1:
      return 'h1';
    case 2:
      return 'h2';
    case 3:
      return 'h3';
    case 4:
      return 'h4';
    case 5:
      return 'h5';
    default:
      return 'h6';
  }
}

function renderList(block: Block): Node {
  const list = block.start === null
    ? el('ul')
    : el('ol', { start: block.start });
  for (const child of block.children) {
    if (!isInline(child) && child.kind === 'list_item') list.appendChild(renderBlock(child));
  }
  return list;
}

function renderListItem(block: Block): Node {
  const item = el('li');
  if (block.checked !== null) {
    item.classList.add('task-item');
    item.appendChild(el('input', { type: 'checkbox', checked: block.checked, disabled: true }));
    item.appendChild(document.createTextNode(' '));
  }
  appendChildren(item, block.children);
  return item;
}

function renderTable(block: Block): Node {
  const rows = block.children.filter((child): child is Block => !isInline(child) && child.kind === 'table_row');
  const table = el('table');
  const head = rows[0];
  if (head === undefined) return table;

  const headRow = el('tr');
  for (const cell of head.children) {
    if (isInline(cell)) continue;
    const th = el('th');
    appendChildren(th, cell.children);
    headRow.appendChild(th);
  }
  table.appendChild(el('thead', {}, headRow));

  const body = el('tbody');
  for (const row of rows.slice(1)) {
    const tr = el('tr');
    for (const cell of row.children) {
      if (isInline(cell)) continue;
      const td = el('td');
      appendChildren(td, cell.children);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  table.appendChild(body);
  return table;
}

function renderCode(block: Block): Node {
  const code = block.children.length === 1 && block.children[0] !== undefined && isInline(block.children[0])
    ? block.children[0].text
    : '';
  const className = block.lang === '' ? undefined : `language-${block.lang.replace(/[^A-Za-z0-9+#._-]/g, '')}`;
  return el('pre', {}, el('code', { class: className }, code));
}

function renderBlock(block: Block): Node {
  switch (block.kind) {
    case 'paragraph': {
      const paragraph = el('p');
      appendChildren(paragraph, block.children);
      return paragraph;
    }
    case 'heading': {
      const heading = el(headingTag(block.level));
      appendChildren(heading, block.children);
      return heading;
    }
    case 'code':
      return renderCode(block);
    case 'list':
      return renderList(block);
    case 'list_item':
      return renderListItem(block);
    case 'blockquote': {
      const quote = el('blockquote');
      appendChildren(quote, block.children);
      return quote;
    }
    case 'rule':
      return el('hr');
    case 'table':
      return renderTable(block);
    case 'equation': {
      const body = block.children.length === 1 && block.children[0] !== undefined && isInline(block.children[0])
        ? block.children[0].text
        : '';
      return el('pre', { class: 'equation' }, el('code', {}, body));
    }
    case 'table_row': {
      const tr = el('tr');
      appendChildren(tr, block.children);
      return tr;
    }
    case 'table_cell': {
      const td = el('td');
      appendChildren(td, block.children);
      return td;
    }
    default:
      return el('div');
  }
}

export function renderMarkdown(blocks: Block[]): DocumentFragment {
  const fragment = document.createDocumentFragment();
  for (const block of blocks) fragment.appendChild(renderBlock(block));
  return fragment;
}

export function renderMarkdownInto(container: Element, blocks: Block[]): void {
  container.replaceChildren(renderMarkdown(blocks));
}

export interface MarkdownWriter {
  write(chunk: string): void;
  set(source: string): void;
  end(): void;
}

/**
 * Streaming renderer. Writes are coalesced to one animation frame so a token
 * stream does not rebuild the DOM for every chunk; `end()` paints the final
 * state synchronously.
 */
export function createMarkdown(container: Element): MarkdownWriter {
  const stream = new MarkdownStream();
  let frame: number | null = null;
  let ended = false;

  const render = (): void => {
    frame = null;
    renderMarkdownInto(container, stream.blocks());
  };

  const schedule = (): void => {
    if (ended || frame !== null) return;
    frame = requestAnimationFrame(render);
  };

  return {
    write(chunk: string) {
      if (ended || !chunk) return;
      stream.write(chunk);
      schedule();
    },
    set(source: string) {
      if (ended) return;
      stream.reset(source);
      schedule();
    },
    end() {
      if (ended) return;
      ended = true;
      stream.end();
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      render();
    },
  };
}
