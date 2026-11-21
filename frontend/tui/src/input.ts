// Raw-mode keyboard input. One callback, no widgets.
//
// node:readline never emits a lone ESC, so it cannot drive a modal overlay:
// it waits for the rest of an escape sequence that never comes. This parses
// raw stdin instead, with a short timer that resolves an incomplete escape
// into ESC while still recognising alt+key, CSI sequences and bracketed paste.

export type Key =
  | { name: 'enter' }
  | { name: 'newline' }
  | { name: 'esc' }
  | { name: 'backspace' }
  | { name: 'delete' }
  | { name: 'left' }
  | { name: 'right' }
  | { name: 'home' }
  | { name: 'end' }
  | { name: 'up' }
  | { name: 'down' }
  | { name: 'ctrl-c' }
  | { name: 'paste'; text: string }
  | { name: 'alt'; char: string }
  | { name: 'char'; char: string };

export interface InputHandle {
  stop: () => void;
}

const ESCAPE_TIMEOUT_MS = 25;
const CSI_FINAL = /[A-Za-z~]/;
const PASTE_START = '\x1b[200~';
const PASTE_END = '\x1b[201~';

const CSI_KEYS: Record<string, Key> = {
  A: { name: 'up' },
  B: { name: 'down' },
  C: { name: 'right' },
  D: { name: 'left' },
  H: { name: 'home' },
  F: { name: 'end' },
};

const CSI_TILDE: Record<string, Key> = {
  '1': { name: 'home' },
  '3': { name: 'delete' },
  '4': { name: 'end' },
  '7': { name: 'home' },
  '8': { name: 'end' },
};

/** Kitty encodes modifiers as 1 + a bitmask whose low bit is shift. */
function shiftHeld(modifier: number): boolean {
  return ((modifier - 1) & 1) === 1;
}

/** A key reported by modifyOtherKeys, which spells them CSI 27 ; modifier ; code ~. */
function modifiedKey(code: number, modifier: number): Key | null {
  const bits = modifier - 1;
  const alt = (bits & 2) === 2;
  const ctrl = (bits & 4) === 4;
  if (code === 13) return shiftHeld(modifier) ? { name: 'newline' } : { name: 'enter' };
  if (code === 27) return { name: 'esc' };
  if (code === 127 || code === 8) return { name: 'backspace' };
  // Level 2 reports control characters and alt+letter the same way, so ctrl+c
  // has to stay reachable and alt+m / alt+p have to keep opening the pickers.
  if (ctrl) return code === 99 ? { name: 'ctrl-c' } : null;
  if (code < 32) return null;
  const char = String.fromCodePoint(code);
  if (alt) return { name: 'alt', char: char.toLowerCase() };
  return { name: 'char', char: shiftHeld(modifier) ? char.toUpperCase() : char };
}

/** A complete CSI sequence (`\x1b[...` + final byte) as a key, if it is one we know. */
export function parseCsi(sequence: string): Key | null {
  const final = sequence[sequence.length - 1] ?? '';
  const [head = '', modifier = '1'] = sequence.slice(2, -1).split(';');
  if (final === '~') {
    const extended = sequence.slice(2, -1).split(';');
    if (extended[0] === '27') return modifiedKey(Number(extended[2] ?? ''), Number(extended[1] ?? '1'));
    return CSI_TILDE[head] ?? null;
  }
  // The kitty protocol reports return, escape and backspace with their modifiers.
  if (final === 'u') {
    if (head === '13') return shiftHeld(Number(modifier)) ? { name: 'newline' } : { name: 'enter' };
    if (head === '27') return { name: 'esc' };
    if (head === '127') return { name: 'backspace' };
    return null;
  }
  return CSI_KEYS[final] ?? null;
}

export function startInput(handler: (key: Key) => void): InputHandle {
  const stdin = process.stdin;
  if (stdin.isTTY) stdin.setRawMode(true);
  stdin.setEncoding('utf8');
  stdin.resume();

  let pending = '';
  let timer: ReturnType<typeof setTimeout> | null = null;

  function clearTimer(): void {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  }

  /** An escape that never completes inside the timeout is a plain ESC. */
  function scheduleEscape(): void {
    clearTimer();
    timer = setTimeout(() => {
      timer = null;
      pending = '';
      handler({ name: 'esc' });
    }, ESCAPE_TIMEOUT_MS);
  }

  function emitChar(char: string): void {
    // Enter is CR; a terminal that sends LF for it still submits the prompt.
    if (char === '\r' || char === '\n') handler({ name: 'enter' });
    else if (char === '\x7f' || char === '\b') handler({ name: 'backspace' });
    else if (char === '\x03') handler({ name: 'ctrl-c' });
    else if (char >= ' ') handler({ name: 'char', char });
  }

  function drain(): void {
    while (pending !== '') {
      if (pending.startsWith(PASTE_START)) {
        const end = pending.indexOf(PASTE_END, PASTE_START.length);
        if (end < 0) return;
        const text = pending.slice(PASTE_START.length, end);
        pending = pending.slice(end + PASTE_END.length);
        handler({ name: 'paste', text });
        continue;
      }
      // A split start marker falls through to the CSI branch, whose timer resolves it.
      const char = pending[0] ?? '';
      if (char !== '\x1b') {
        const first = pending[Symbol.iterator]().next().value ?? char;
        pending = pending.slice(first.length);
        emitChar(first);
        continue;
      }

      const second = pending[1];
      if (second === undefined) {
        scheduleEscape();
        return;
      }
      if (second === '[' || second === 'O') {
        let end = -1;
        for (let current = 2; current < pending.length; current += 1) {
          if (CSI_FINAL.test(pending[current] ?? '')) {
            end = current;
            break;
          }
        }
        if (end < 0) {
          scheduleEscape();
          return;
        }
        const sequence = pending.slice(0, end + 1);
        pending = pending.slice(end + 1);
        const key = parseCsi(sequence);
        if (key) handler(key);
        continue;
      }
      // Two escapes in one read are two presses, not an alt+escape.
      if (second === '\x1b') {
        pending = pending.slice(1);
        handler({ name: 'esc' });
        continue;
      }
      const alt = pending.slice(1)[Symbol.iterator]().next().value ?? second;
      pending = pending.slice(1 + alt.length);
      if (alt >= ' ') handler({ name: 'alt', char: alt });
    }
  }

  const onData = (chunk: string | Buffer): void => {
    clearTimer();
    pending += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    drain();
  };

  stdin.on('data', onData);
  return {
    stop: () => {
      clearTimer();
      stdin.off('data', onData);
      if (stdin.isTTY) stdin.setRawMode(false);
      stdin.pause();
    },
  };
}
