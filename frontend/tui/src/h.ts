// Tiny component tree for the terminal. Same idea as frontend/web/ts/dom.ts:
// components return plain nodes, render.ts turns a tree into lines of text.
// Children are normalised at construction, so render never has to flatten.

export type Color = 'red' | 'green' | 'yellow' | 'blue' | 'magenta' | 'cyan' | 'gray';

export interface Style {
  bold: boolean;
  dim: boolean;
  reverse: boolean;
  color: Color | null;
}

export const plain: Style = { bold: false, dim: false, reverse: false, color: null };

export interface Props {
  bold?: boolean;
  dim?: boolean;
  reverse?: boolean;
  color?: Color;
  width?: number;
  title?: string;
}

/** Tags the terminal layout knows how to place. */
export type Tag = 'row' | 'stack' | 'p' | 'span' | 'text' | 'spacer' | 'line' | 'box';

export interface VNode {
  tag: Tag;
  props: Props;
  children: VNode[];
  text?: string;
}

export type Child = VNode | string | number | boolean | null | undefined;
export type ChildList = Child | ChildList[];

/**
 * Control characters would let server content drive the terminal (clear the
 * screen, set the title, OSC 52 clipboard writes). They are neutralised before
 * layout, so painted width and layout width always agree.
 */
// oxlint-disable-next-line eslint/no-control-regex -- stripping control characters is the point
const ESCAPE_SEQUENCE = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g;
// oxlint-disable-next-line eslint/no-control-regex -- stripping control characters is the point
const CONTROL_CHARACTER = /[\u0000-\u001f\u007f-\u009f]/g;

export function sanitizeText(value: string): string {
  return value.replace(ESCAPE_SEQUENCE, '').replace(CONTROL_CHARACTER, ' ');
}

function textNode(value: string): VNode {
  return { tag: 'text', props: {}, children: [], text: sanitizeText(value) };
}

function toNodes(children: ChildList[]): VNode[] {
  const out: VNode[] = [];
  for (const child of children) {
    if (Array.isArray(child)) {
      out.push(...toNodes(child));
      continue;
    }
    if (child === null || child === undefined || typeof child === 'boolean') continue;
    if (typeof child === 'string' || typeof child === 'number') out.push(textNode(String(child)));
    else out.push(child);
  }
  return out;
}

export function h(tag: Tag, props: Props = {}, ...children: ChildList[]): VNode {
  return { tag, props, children: toNodes(children) };
}

/** Inline style: a span's props apply to its own text children. */
export function styleOf(props: Props, parent: Style = plain): Style {
  return {
    bold: props.bold ?? parent.bold,
    dim: props.dim ?? parent.dim,
    reverse: props.reverse ?? parent.reverse,
    color: props.color ?? parent.color,
  };
}
