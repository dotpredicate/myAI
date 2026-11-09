// Terminal text metrics: grapheme clusters and display width (wcwidth-ish).
//
// A grapheme cluster is what the user perceives as one character, so it is also
// the smallest unit we are allowed to cut or backspace over. Width is measured
// in terminal cells: East Asian wide characters and emoji take two.

const SEGMENTER = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

const WIDE_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x1100, 0x115f],
  [0x2e80, 0x303e],
  [0x3041, 0x33ff],
  [0x3400, 0x4dbf],
  [0x4e00, 0x9fff],
  [0xa000, 0xa4cf],
  [0xac00, 0xd7a3],
  [0xf900, 0xfaff],
  [0xfe10, 0xfe19],
  [0xfe30, 0xfe6f],
  [0xff00, 0xff60],
  [0xffe0, 0xffe6],
  [0x1f300, 0x1f64f],
  [0x1f900, 0x1f9ff],
  [0x1fa70, 0x1faff],
  [0x20000, 0x3fffd],
];

function isWide(code: number): boolean {
  for (const [start, end] of WIDE_RANGES) {
    if (code >= start && code <= end) return true;
  }
  return false;
}

export function graphemes(text: string): string[] {
  return Array.from(SEGMENTER.segment(text), (part) => part.segment);
}

export function graphemeWidth(grapheme: string): number {
  const code = grapheme.codePointAt(0);
  if (code === undefined || code === 0) return 0;
  // A cluster that starts with a combining mark or a zero-width character
  // occupies no cells of its own.
  const head = String.fromCodePoint(code);
  if (/\p{Mark}|[\u200b-\u200f\u2060]/u.test(head)) return 0;
  return isWide(code) ? 2 : 1;
}

export function textWidth(text: string): number {
  let total = 0;
  for (const grapheme of graphemes(text)) total += graphemeWidth(grapheme);
  return total;
}

/** Cuts text to fit `max` cells, never splitting a grapheme cluster. */
export function truncateToWidth(text: string, max: number): string {
  if (max <= 0) return '';
  let used = 0;
  const kept: string[] = [];
  for (const grapheme of graphemes(text)) {
    const width = graphemeWidth(grapheme);
    if (used + width > max) break;
    kept.push(grapheme);
    used += width;
  }
  return kept.join('');
}

export function dropLastGrapheme(text: string): string {
  const parts = graphemes(text);
  parts.pop();
  return parts.join('');
}
