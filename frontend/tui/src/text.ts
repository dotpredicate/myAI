// Terminal text metrics: grapheme clusters and display width (wcwidth-ish).
//
// A grapheme cluster is what the user perceives as one character, so it is also
// the smallest unit we are allowed to cut or backspace over. Width is measured
// in terminal cells: East Asian wide characters and emoji take two.
//
// Almost all terminal text is printable ASCII, where one character is one cell,
// so that case skips the segmenter entirely.

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

/** Printable ASCII: one character, one cell, no clustering to consider. */
export const PRINTABLE_ASCII = /^[\x20-\x7e]*$/;

export function graphemes(text: string): string[] {
  return Array.from(SEGMENTER.segment(text), (part) => part.segment);
}

export function graphemeWidth(grapheme: string): number {
  const code = grapheme.codePointAt(0);
  if (code === undefined || code === 0) return 0;
  // Printable ASCII is by far the common case, and is always one cell.
  if (code >= 0x20 && code < 0x7f) return 1;
  // A cluster that starts with a combining mark or a zero-width character
  // occupies no cells of its own.
  const head = String.fromCodePoint(code);
  if (/\p{Mark}|[\u200b-\u200f\u2060]/u.test(head)) return 0;
  return isWide(code) ? 2 : 1;
}

// Widths are asked for again and again for the same strings (the same row is
// measured by layout, then by paint, then again next frame), so short strings
// keep their answer. The key is the text itself, so only bounded-size entries
// are remembered to keep this from holding a transcript's worth of memory.
const widthCache = new Map<string, number>();
const CACHE_LIMIT = 4096;
const CACHEABLE_LENGTH = 128;

export function textWidth(text: string): number {
  if (PRINTABLE_ASCII.test(text)) return text.length;
  if (text.length <= CACHEABLE_LENGTH) {
    const cached = widthCache.get(text);
    if (cached !== undefined) return cached;
  }
  let total = 0;
  for (const part of SEGMENTER.segment(text)) total += graphemeWidth(part.segment);
  if (text.length <= CACHEABLE_LENGTH) {
    if (widthCache.size >= CACHE_LIMIT) widthCache.clear();
    widthCache.set(text, total);
  }
  return total;
}

/** Cuts text to fit `max` cells, never splitting a grapheme cluster. */
export function truncateToWidth(text: string, max: number): string {
  if (max <= 0) return '';
  if (text.length <= max && PRINTABLE_ASCII.test(text)) return text;
  // Segment lazily: a long string only has to be walked until the line is full.
  let used = 0;
  let end = 0;
  for (const part of SEGMENTER.segment(text)) {
    const width = graphemeWidth(part.segment);
    if (used + width > max) break;
    used += width;
    end = part.index + part.segment.length;
  }
  return text.slice(0, end);
}

export function dropLastGrapheme(text: string): string {
  const parts = graphemes(text);
  parts.pop();
  return parts.join('');
}
