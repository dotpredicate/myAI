// Searchable list overlay. Used for provider and model selection so a long
// list is reached by typing instead of cycling.

import { filterItems, type PickItem } from '#core/models';
import { Heading, Hint, KeyHint, Line, Row, Spacer, Stack, Text } from './components.ts';
import { layout, paint } from './render.ts';
import type { Key } from './input.ts';
import { type VNode } from './h.ts';

export interface Picker {
  title: string;
  items: PickItem[];
  query: string;
  index: number;
}

export type PickerResult = { kind: 'open' } | { kind: 'cancelled' } | { kind: 'chosen'; value: string };

export function openPicker(title: string, items: PickItem[]): Picker {
  return { title, items, query: '', index: 0 };
}

export function pickerItems(picker: Picker): PickItem[] {
  return filterItems(picker.items, picker.query);
}

export function handlePickerKey(picker: Picker, key: Key): PickerResult {
  const visible = pickerItems(picker);
  switch (key.name) {
    case 'esc':
      return { kind: 'cancelled' };
    case 'up':
      picker.index = Math.max(0, picker.index - 1);
      return { kind: 'open' };
    case 'down':
      picker.index = Math.min(Math.max(0, visible.length - 1), picker.index + 1);
      return { kind: 'open' };
    case 'backspace':
      picker.query = Array.from(picker.query).slice(0, -1).join('');
      picker.index = 0;
      return { kind: 'open' };
    case 'char':
      picker.query += key.char;
      picker.index = 0;
      return { kind: 'open' };
    case 'enter': {
      const item = visible[picker.index];
      return item ? { kind: 'chosen', value: item.value } : { kind: 'open' };
    }
    default:
      return { kind: 'open' };
  }
}

export function pickerFrame(picker: Picker, width: number, height: number): string[] {
  const visible = pickerItems(picker);
  const rows = Math.max(1, height - 6);
  const start = Math.max(0, Math.min(picker.index - Math.floor(rows / 2), Math.max(0, visible.length - rows)));

  const body: VNode[] = [
    Row(Heading(picker.title), Spacer(), Hint(`${visible.length} / ${picker.items.length}`)),
    Line(),
    Row(Text('search ', { dim: true }), Text(picker.query || ' ')),
    Line(),
  ];

  if (visible.length === 0) {
    body.push(Text(picker.items.length === 0 ? 'nothing to choose from' : 'no matches', { dim: true }));
  }
  for (const [offset, item] of visible.slice(start, start + rows).entries()) {
    const selected = start + offset === picker.index;
    body.push(Row(
      Text(selected ? '❯ ' : '  ', { reverse: selected }),
      Text(item.label, { reverse: selected }),
      item.hint ? Text(`  ${item.hint}`, { dim: true }) : null,
    ));
  }

  body.push(
    Line(),
    Row(KeyHint('↑↓', 'move'), Text('  '), KeyHint('enter', 'choose'), Text('  '), KeyHint('esc', 'cancel')),
  );

  return paint(layout(Stack(...body), width), width, height);
}
