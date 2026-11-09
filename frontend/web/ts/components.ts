import { el, type ChildList, type ElProps } from './dom.js';

type Children = ChildList[];

/* --------------------------------------------------------------- layout */

export function Row(...children: Children): HTMLDivElement {
  return el('div', { class: 'row' }, ...children);
}

export function RowRight(...children: Children): HTMLDivElement {
  return el('div', { class: 'row row--right' }, ...children);
}

export function RowEnd(...children: Children): HTMLDivElement {
  return el('div', { class: 'row row--end' }, ...children);
}

export function Stack(...children: Children): HTMLDivElement {
  return el('div', { class: 'stack' }, ...children);
}

export function Split(...children: Children): HTMLDivElement {
  return el('div', { class: 'split' }, ...children);
}

export interface GridProps {
  columns?: 2 | 3;
}

export function Grid(props: GridProps = {}, ...children: Children): HTMLDivElement {
  const extra = props.columns ? ` grid--${props.columns}` : '';
  return el('div', { class: `grid${extra}` }, ...children);
}

export interface PageProps {
  narrow?: boolean;
}

export function Page(props: PageProps = {}, ...children: Children): HTMLElement {
  return el('section', { class: props.narrow ? 'page page--narrow' : 'page' }, ...children);
}

const HEADINGS = { 1: 'h1', 2: 'h2', 3: 'h3', 4: 'h4' } as const;

export function Heading(level: 1 | 2 | 3 | 4, ...children: Children): HTMLHeadingElement {
  return el(HEADINGS[level], {}, ...children);
}

export function Card(...children: Children): HTMLDivElement {
  return el('div', { class: 'card' }, ...children);
}

export function CardHeader(title: ChildList, ...rest: Children): HTMLElement {
  return el('header', {}, Heading(2, title), ...rest);
}

/* ---------------------------------------------------------------- text */

export function Paragraph(...children: Children): HTMLParagraphElement {
  return el('p', {}, ...children);
}

export function FieldGroup(...children: Children): HTMLDivElement {
  return el('div', { class: 'field' }, ...children);
}

export function FieldLabel(text: ChildList, forId: string): HTMLLabelElement {
  return el('label', { class: 'field-label', for: forId }, text);
}

export function Hint(...children: Children): HTMLDivElement {
  return el('div', { class: 'hint' }, ...children);
}

export function InlineHint(...children: Children): HTMLSpanElement {
  return el('span', { class: 'hint' }, ...children);
}

export function LineHint(...children: Children): HTMLParagraphElement {
  return el('p', { class: 'hint' }, ...children);
}

export function FieldError(text?: ChildList, props: { hidden?: boolean } = {}): HTMLSpanElement {
  return el('span', { class: 'field-error', hidden: props.hidden }, text);
}

export interface BadgeProps {
  accent?: boolean;
}

export function Badge(text: ChildList, props: BadgeProps = {}): HTMLSpanElement {
  return el('span', { class: props.accent ? 'badge badge--accent' : 'badge' }, text);
}

export interface NoticeProps {
  level?: 'error' | 'warning' | 'state' | 'success';
  tight?: boolean;
  stack?: boolean;
  class?: string;
}

export function Notice(props: NoticeProps = {}, ...children: Children): HTMLDivElement {
  const classes = ['notice'];
  if (props.tight) classes.push('notice--tight');
  if (props.level) classes.push(`notice--${props.level}`);
  if (props.stack) classes.push('stack');
  if (props.class) classes.push(props.class);
  return el('div', { class: classes.join(' ') }, ...children);
}

/* ------------------------------------------------------------- controls */

export interface ButtonProps {
  variant?: 'primary' | 'danger' | 'quiet';
  small?: boolean;
  submit?: boolean;
  disabled?: boolean;
  title?: string;
  id?: string;
  hidden?: boolean;
  class?: string;
  onClick?: (event: MouseEvent) => void;
}

export function Button(label: ChildList, props: ButtonProps = {}): HTMLButtonElement {
  const classes = ['btn'];
  if (props.small) classes.push('btn-sm');
  if (props.variant) classes.push(`btn-${props.variant}`);
  if (props.class) classes.push(props.class);
  return el('button', {
    type: props.submit ? 'submit' : 'button',
    class: classes.join(' '),
    disabled: props.disabled,
    title: props.title,
    id: props.id,
    hidden: props.hidden,
    onClick: props.onClick,
  }, label);
}

export interface LinkButtonProps {
  variant?: 'primary' | 'danger' | 'quiet';
  small?: boolean;
  class?: string;
  download?: string;
  target?: string;
  rel?: string;
  title?: string;
  hidden?: boolean;
}

export function LinkButton(label: ChildList, href: string, props: LinkButtonProps = {}): HTMLAnchorElement {
  const classes = ['btn'];
  if (props.small) classes.push('btn-sm');
  if (props.variant) classes.push(`btn-${props.variant}`);
  if (props.class) classes.push(props.class);
  return el('a', {
    class: classes.join(' '),
    href,
    download: props.download,
    target: props.target,
    rel: props.rel,
    title: props.title,
    hidden: props.hidden,
  }, label);
}

export function Input(props: ElProps<'input'> = {}): HTMLInputElement {
  return el('input', props);
}

export function TextArea(props: ElProps<'textarea'> = {}): HTMLTextAreaElement {
  return el('textarea', props);
}

export function Select(props: ElProps<'select'> = {}, ...children: Children): HTMLSelectElement {
  return el('select', props, ...children);
}

export function Option(value: string | number, label: ChildList): HTMLOptionElement {
  return el('option', { value: String(value) }, label);
}

export function Spinner(): HTMLSpanElement {
  return el('span', { class: 'spinner' });
}

export function Spacer(): HTMLSpanElement {
  return el('span', { class: 'spacer' });
}

export function InlineLoading(text: ChildList): HTMLSpanElement {
  return el('span', { class: 'inline-loading' }, Spinner(), text);
}

/* ---------------------------------------------------------------- forms */

export interface FormProps {
  novalidate?: boolean;
  hidden?: boolean;
  class?: string;
  id?: string;
  onSubmit?: (event: SubmitEvent) => void;
}

export function Form(props: FormProps = {}, ...children: Children): HTMLFormElement {
  return el('form', props, ...children);
}

export function Summary(...children: Children): HTMLElement {
  return el('summary', {}, ...children);
}

export function IconButton(label: ChildList, props: { onClick?: (event: MouseEvent) => void; title?: string; hidden?: boolean } = {}): HTMLButtonElement {
  return el('button', { type: 'button', class: 'icon-btn', ...props }, label);
}

/* ---------------------------------------------------------------- tables */

export function Table(...children: Children): HTMLTableElement {
  return el('table', {}, ...children);
}

export function TableHead(...children: Children): HTMLTableSectionElement {
  return el('thead', {}, ...children);
}

export function TableRow(...children: Children): HTMLTableRowElement {
  return el('tr', {}, ...children);
}

export function TableHeaderCell(...children: Children): HTMLTableCellElement {
  return el('th', {}, ...children);
}

export function TableCell(...children: Children): HTMLTableCellElement {
  return el('td', {}, ...children);
}

/* ---------------------------------------------------------------- lists */

export function List(...items: Children): HTMLUListElement {
  return el('ul', { class: 'list' }, ...items);
}

export function ListItem(...children: Children): HTMLLIElement {
  return el('li', { class: 'list-item' }, ...children);
}
