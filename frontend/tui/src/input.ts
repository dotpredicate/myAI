// Raw-mode keyboard input. One callback, no widgets.

import { emitKeypressEvents } from 'node:readline';

export type Key =
  | { name: 'enter' }
  | { name: 'esc' }
  | { name: 'backspace' }
  | { name: 'up' }
  | { name: 'down' }
  | { name: 'left' }
  | { name: 'right' }
  | { name: 'tab' }
  | { name: 'ctrl-c' }
  | { name: 'char'; char: string };

interface KeypressEvent {
  name?: string;
  ctrl?: boolean;
  shift?: boolean;
  sequence?: string;
}

export interface InputHandle {
  stop: () => void;
}

export function startInput(handler: (key: Key) => void): InputHandle {
  const stdin = process.stdin;
  emitKeypressEvents(stdin);
  if (stdin.isTTY) stdin.setRawMode(true);
  stdin.resume();

  const onKeypress = (_text: string | undefined, event: KeypressEvent | undefined): void => {
    if (!event) return;
    if (event.ctrl === true && event.name === 'c') {
      handler({ name: 'ctrl-c' });
      return;
    }
    if (event.ctrl === true) return;
    switch (event.name) {
      case 'return':
      case 'enter':
        handler({ name: 'enter' });
        return;
      case 'escape':
        handler({ name: 'esc' });
        return;
      case 'backspace':
        handler({ name: 'backspace' });
        return;
      case 'up':
        handler({ name: 'up' });
        return;
      case 'down':
        handler({ name: 'down' });
        return;
      case 'left':
        handler({ name: 'left' });
        return;
      case 'right':
        handler({ name: 'right' });
        return;
      case 'tab':
        handler({ name: 'tab' });
        return;
      case 'space':
        handler({ name: 'char', char: ' ' });
        return;
      default:
        break;
    }
    const sequence = event.sequence ?? '';
    if (sequence.length === 1 && sequence >= ' ') handler({ name: 'char', char: sequence });
  };

  stdin.on('keypress', onKeypress);
  return {
    stop: () => {
      stdin.off('keypress', onKeypress);
      if (stdin.isTTY) stdin.setRawMode(false);
      stdin.pause();
    },
  };
}
