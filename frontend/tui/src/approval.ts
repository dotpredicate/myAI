// Shortcuts while a tool call waits for approval.
//
// Letters always type: y and n only decide while the composer is still empty,
// and Enter declines with whatever was typed.

export interface Approval {
  decision: 'approve' | 'reject';
  comment: string;
}

export function approvalForShortcut(char: string, composerText: string): Approval | null {
  if (composerText !== '') return null;
  if (char === 'y') return { decision: 'approve', comment: '' };
  if (char === 'n') return { decision: 'reject', comment: '' };
  return null;
}
