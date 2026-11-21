// Unified diffs: one set of rules for the browser and the terminal.
//
// Both clients show the same diff, so what a line *is* lives here and each
// client only maps `kind` to its own styling (CSS classes in the web client,
// colours in the terminal). The maps on both sides are exhaustive
// `Record<DiffLineKind, ...>`, so adding a kind here breaks both builds instead
// of silently rendering it differently in one of them.
//
// A diff body has three line shapes: ' ' context, '+' addition, '-' removal.
// Everything else is a header: the '---'/'+++' file markers, '@@' hunk
// headers, git's 'diff --git'/'index ...' lines, and the
// '\ No newline at end of file' marker. Headers are classified first, because
// '+++' also starts with '+' and '---' with '-'.

import type { ToolOutcome } from './api.js';

export type DiffLineKind = 'meta' | 'add' | 'remove' | 'context';

export interface DiffLine {
  kind: DiffLineKind;
  /** The line as it appears in the diff, without a trailing CR. */
  text: string;
}

/** Classifies one line of a unified diff; the order of the checks matters. */
export function classifyDiffLine(line: string): DiffLineKind {
  if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@')) return 'meta';
  if (line.startsWith('+')) return 'add';
  if (line.startsWith('-')) return 'remove';
  if (line === '' || line.startsWith(' ')) return 'context';
  return 'meta';
}

/**
 * Splits a diff into classified lines. A trailing newline keeps its empty last
 * line, which is what a diff that ends with "\ No newline at end of file" needs.
 */
export function parseDiff(text: string | null | undefined): DiffLine[] {
  return (text ?? '').split('\n').map((raw) => {
    const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
    return { kind: classifyDiffLine(line), text: line };
  });
}

export interface ProposalDiff {
  /** The file the diff applies to; empty when the proposal names none. */
  target: string;
  diff: string;
}

/** The diff a tool proposes, or null for every other outcome. */
export function proposalDiff(outcome: ToolOutcome | null | undefined): ProposalDiff | null {
  if (!outcome || outcome.type !== 'proposal') return null;
  return { target: outcome.target, diff: outcome.diff };
}
