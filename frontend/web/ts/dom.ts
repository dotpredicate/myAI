// DOM primitives shared by ui helpers and components. `el()` is the escape
// hatch for one-off markup; anything used in more than a couple of places gets
// a component in components.ts.

let sequence = 0;

export function uid(prefix = 'id'): string {
  sequence += 1;
  return `${prefix}-${sequence}`;
}

type Handler<E extends Event> = (event: E) => void;

/** Anything `el()` can append as a child. Arrays are flattened. */
export type Child = Node | string | number | boolean | null | undefined;
export type ChildList = Child | ChildList[];

/** Props accepted by every element. */
export interface BaseProps {
  class?: string;
  dataset?: Record<string, string>;
  id?: string;
  hidden?: boolean;
  title?: string;
  onClick?: Handler<MouseEvent>;
}

/**
 * Attributes accepted per tag, on top of the tag's own DOM type. Deliberately
 * narrow: an unknown attribute is a compile error, which also keeps `style`
 * and `innerHTML` out of the UI. Tags without extra attributes need no entry.
 */
export interface TagProps {
  a: { href?: string; target?: string; rel?: string; download?: string };
  button: { type?: 'button' | 'submit' | 'reset'; disabled?: boolean };
  form: { novalidate?: boolean };
  img: { src?: string; alt?: string };
  input: {
    type?: string;
    name?: string;
    placeholder?: string;
    autocomplete?: string;
    required?: boolean;
    readonly?: boolean;
    disabled?: boolean;
    checked?: boolean;
    min?: string | number;
    max?: string | number;
    step?: string | number;
    value?: string | number;
  };
  label: { for?: string };
  option: { value?: string; disabled?: boolean; selected?: boolean };
  progress: { value?: string | number; max?: string | number };
  select: { name?: string; required?: boolean; disabled?: boolean; multiple?: boolean };
  textarea: {
    name?: string;
    placeholder?: string;
    autocomplete?: string;
    required?: boolean;
    readonly?: boolean;
    disabled?: boolean;
    rows?: number;
    value?: string | number;
  };
}

export type ElProps<K extends keyof HTMLElementTagNameMap> =
  BaseProps & (K extends keyof TagProps ? TagProps[K] : {});

function isEventListener(value: unknown): value is EventListener {
  return typeof value === 'function';
}

function appendChild(parent: Node, child: ChildList): void {
  if (child === null || child === undefined || child === false) return;
  if (Array.isArray(child)) {
    for (const item of child) appendChild(parent, item);
    return;
  }
  parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
}

/**
 * Creates an element. Props starting with `on` become event listeners,
 * `dataset` is assigned to the dataset, and the rest are either DOM properties
 * (booleans, `value`, `rows`) or attributes.
 */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props?: ElProps<K>,
  ...children: ChildList[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  const entries = Object.entries(props || {});
  for (const [key, value] of entries) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') {
      node.className = String(value);
      continue;
    }
    if (key === 'dataset') {
      if (typeof value === 'object' && value !== null) Object.assign(node.dataset, value);
      continue;
    }
    if (key.startsWith('on') && isEventListener(value)) {
      node.addEventListener(key.slice(2).toLowerCase(), value);
      continue;
    }
    if (key in node && typeof Reflect.get(node, key) === 'boolean') {
      Reflect.set(node, key, Boolean(value));
      continue;
    }
    if (key === 'value' || key === 'rows') {
      Reflect.set(node, key, value);
      continue;
    }
    if (value === true) {
      node.setAttribute(key, '');
      continue;
    }
    node.setAttribute(key, String(value));
  }
  appendChild(node, children);
  return node;
}

export function clear(node: Element): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}
