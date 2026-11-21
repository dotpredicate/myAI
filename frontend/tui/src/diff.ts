// Diff rendering for the terminal.
//
// `#core/diff` decides what each line is; this module decides how it looks.
// A line is a `pre` block, so layout wraps it at the terminal edge and keeps
// every character: the same content the web client shows with `pre-wrap`.

import { type DiffLineKind, parseDiff } from '#core/diff';
import { Paragraph, Text } from './components.ts';
import { h, type Props, type VNode } from './h.ts';

const STYLE: Record<DiffLineKind, Props> = {
  add: { color: 'green' },
  remove: { color: 'red' },
  meta: { dim: true },
  context: {},
};

/** The `target  /path` line of a proposal, or nothing when it names no file. */
export function targetNodes(target: string | null): VNode[] {
  return target === null ? [] : [Paragraph(Text('target  ', { dim: true }), Text(target, { color: 'cyan' }))];
}

/** Every line of the diff, coloured. Layout wraps them; nothing is cut. */
export function diffNodes(diff: string): VNode[] {
  return parseDiff(diff).map((line) => h('pre', STYLE[line.kind], line.text));
}
