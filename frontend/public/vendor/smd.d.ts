// Minimal typings for the untyped smd.js. Without this file tsc refuses to emit
// because smd.js would be an input (TS5055), and `allowJs: true` would type-check
// 1600 lines of third-party code we do not want to own.

export function default_renderer(root: Element): unknown;
export function parser(renderer: unknown): unknown;
export function parser_write(parser: unknown, chunk: string): void;
export function parser_end(parser: unknown): void;
