// Raw-mode keyboard input. One callback, no widgets.
//
// node:readline never emits a lone ESC, so it cannot drive a modal overlay:
// it waits for the rest of an escape sequence that never comes. This parses
// raw stdin instead, with a short timer that resolves an incomplete escape
// into ESC while still recognising alt+key and CSI sequences.

export type Key =
  | { name: 'enter' }
  | { name: 'esc' }
  | { name: 'backspace' }
  | { name: 'up' }
  | { name: 'down' }
  | { name: 'ctrl-c' }
  | { name: 'alt'; char: string }
  | { name: 'char'; char: string };

export interface InputHandle {
  stop: () => void;
}

const ESCAPE_TIMEOUT_MS = 25;
const CSI_FINAL = /[A-Za-z~]/;
const CSI_KEYS: Record<string, Key> = {
  A: { name: 'up' },
  B: { name: 'down' },
};

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
    if (char === '\r' || char === '\n') handler({ name: 'enter' });
    else if (char === '\x7f' || char === '\b') handler({ name: 'backspace' });
    else if (char === '\x03') handler({ name: 'ctrl-c' });
    else if (char >= ' ') handler({ name: 'char', char });
  }

  function drain(): void {
    const chars = Array.from(pending);
    let index = 0;
    while (index < chars.length) {
      const char = chars[index] ?? '';
      if (char !== '\x1b') {
        index += 1;
        emitChar(char);
        continue;
      }
      const second = chars[index + 1];
      if (second === undefined) {
        pending = chars.slice(index).join('');
        scheduleEscape();
        return;
      }
      if (second === '[' || second === 'O') {
        let end = -1;
        for (let current = index + 2; current < chars.length; current += 1) {
          if (CSI_FINAL.test(chars[current] ?? '')) {
            end = current;
            break;
          }
        }
        if (end < 0) {
          pending = chars.slice(index).join('');
          scheduleEscape();
          return;
        }
        const key = CSI_KEYS[chars[end] ?? ''];
        index = end + 1;
        if (key) handler(key);
        continue;
      }
      index += 2;
      if (second >= ' ') handler({ name: 'alt', char: second });
    }
    pending = '';
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
