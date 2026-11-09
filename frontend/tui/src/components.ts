// Components mirroring frontend/web/ts/components.ts, reduced to what a
// terminal can show. Same names, same mental model, no CSS.

import { h, type ChildList, type Props, type VNode } from './h.ts';

type Children = ChildList[];

export function Row(...children: Children): VNode {
  return h('row', {}, ...children);
}

export function Stack(...children: Children): VNode {
  return h('stack', {}, ...children);
}

export function Paragraph(...children: Children): VNode {
  return h('p', {}, ...children);
}

export function Text(value: ChildList, props: Props = {}): VNode {
  return h('span', props, value);
}

export function Heading(value: ChildList): VNode {
  return h('span', { bold: true, color: 'cyan' }, value);
}

export function Hint(value: ChildList, props: Props = {}): VNode {
  return h('span', { dim: true, ...props }, value);
}

export function Badge(value: string, props: Props = {}): VNode {
  return h('span', { reverse: true, ...props }, ` ${value} `);
}

export function Spacer(): VNode {
  return h('spacer', {});
}

export function Line(): VNode {
  return h('line', {});
}

export function Box(props: Props = {}, ...children: Children): VNode {
  return h('box', props, ...children);
}

export function KeyHint(key: string, label: string): VNode {
  return h('span', {}, h('span', { bold: true }, `[${key}]`), h('span', { dim: true }, ` ${label}`));
}
