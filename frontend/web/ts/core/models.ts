// Search shared by the web client and the TUI. Presentation stays per client,
// matching and ordering live here so both behave the same.

export interface SearchItem {
  label: string;
  hint?: string;
}

export interface PickItem extends SearchItem {
  value: string;
}

/** A model as the pickers need it: the identifier is the value, the name is the label. */
export function modelChoices(models: ReadonlyArray<{ id: string; name: string }>): PickItem[] {
  return models.map((model) => ({ value: model.id, label: model.name || model.id }));
}

/**
 * Items whose label or hint contains the query, in their original order.
 * Matching is case-insensitive and trimmed; an empty query keeps everything.
 */
export function filterItems<T extends SearchItem>(items: T[], query: string): T[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return items;
  return items.filter((item) => {
    return item.label.toLowerCase().includes(needle)
      || (item.hint ?? '').toLowerCase().includes(needle);
  });
}
