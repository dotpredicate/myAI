import assert from 'node:assert/strict';
import test from 'node:test';
import { parseCsi } from '../src/input.ts';

test('only shift+enter opens a new line', () => {
  assert.deepEqual(parseCsi('\x1b[13;2u'), { name: 'newline' }); // kitty, shift
  assert.deepEqual(parseCsi('\x1b[13u'), { name: 'enter' }); // kitty, plain
  assert.deepEqual(parseCsi('\x1b[13;3u'), { name: 'enter' }); // kitty, alt
  assert.deepEqual(parseCsi('\x1b[13;5u'), { name: 'enter' }); // kitty, ctrl
  assert.deepEqual(parseCsi('\x1b[27;2;13~'), { name: 'newline' }); // modifyOtherKeys, shift
  assert.deepEqual(parseCsi('\x1b[27;5;13~'), { name: 'enter' }); // modifyOtherKeys, ctrl
});

test('modifyOtherKeys level 2 keeps ctrl+c and backspace working', () => {
  assert.deepEqual(parseCsi('\x1b[27;5;99~'), { name: 'ctrl-c' });
  assert.deepEqual(parseCsi('\x1b[27;2;99~'), { name: 'char', char: 'C' }); // shift never drops text
  assert.deepEqual(parseCsi('\x1b[27;5;13~'), { name: 'enter' });
  assert.deepEqual(parseCsi('\x1b[27;5;127~'), { name: 'backspace' });
  assert.deepEqual(parseCsi('\x1b[27;2;27~'), { name: 'esc' });
  assert.deepEqual(parseCsi('\x1b[27;5;106~'), null); // ctrl+j is not newline
});

test('alt+letter survives modifyOtherKeys level 2', () => {
  assert.deepEqual(parseCsi('\x1b[27;3;109~'), { name: 'alt', char: 'm' });
  assert.deepEqual(parseCsi('\x1b[27;3;112~'), { name: 'alt', char: 'p' });
  assert.deepEqual(parseCsi('\x1b[27;4;109~'), { name: 'alt', char: 'm' }); // shift+alt
  assert.deepEqual(parseCsi('\x1b[27;5;109~'), null); // ctrl+m stays unbound
  assert.deepEqual(parseCsi('\x1b[27;2;77~'), { name: 'char', char: 'M' }); // shift never eats text
});

test('the other sequences still decode', () => {
  assert.deepEqual(parseCsi('\x1b[A'), { name: 'up' });
  assert.deepEqual(parseCsi('\x1b[B'), { name: 'down' });
  assert.deepEqual(parseCsi('\x1b[H'), { name: 'home' });
  assert.deepEqual(parseCsi('\x1b[3~'), { name: 'delete' });
  assert.deepEqual(parseCsi('\x1b[200~'), null);
});
