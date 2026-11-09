import { el, clear, labeledField, allowedPolicies, asPolicy, policyLabel, statusMessage as setStatus } from '../ui.js';
import type { RepositoryConfig, SecurityPolicy } from '../api.js';
import { RowEnd, Stack, Heading, Hint, InlineHint, LineHint, Button, Select, Option } from '../components.js';

export interface RepositoryAccessEntry {
  repository_id: number;
  repository_internal_name: string;
  security_policy_override: SecurityPolicy | null;
}

export interface RepositoryAccessEditorOptions {
  initial: RepositoryAccessEntry[];
  onChange: () => void;
  subject?: string;
}

export interface RepositoryAccessEditor {
  section: HTMLDivElement;
  entries: RepositoryAccessEntry[];
  setRepositories(repositories: RepositoryConfig[], loaded: boolean): void;
  setError(error: Error, retry: () => void): void;
}

export function createRepositoryAccessEditor(options: RepositoryAccessEditorOptions): RepositoryAccessEditor {
  const entries = [...options.initial];
  let repositories: RepositoryConfig[] = [];

  const repoSelect = Select({});
  const policySelect = Select({});
  const repoAdd = Button('Add');
  const repoPills = RowEnd();
  const repoStatus = Hint();
  const repoSelectField = labeledField('Repository', repoSelect);
  const policyField = labeledField('Permissions', policySelect);
  const repoAddBar = RowEnd(repoSelectField.wrapper, policyField.wrapper, repoAdd);
  const subject = options.subject || 'agent';
  const section = Stack(
    Heading(3, 'Repository access'),
    LineHint(`Access added here is available to the ${subject}. Permissions stay within each repository’s base policy.`),
    repoAddBar,
    repoPills,
    repoStatus,
  );

  function updatePolicyOptions(): void {
    clear(policySelect);
    const repo = repositories.find((item) => String(item.id) === repoSelect.value);
    if (!repo) {
      policySelect.appendChild(Option('', 'Default'));
      policySelect.disabled = true;
      return;
    }
    policySelect.disabled = false;
    policySelect.appendChild(Option('', `Repository default (${policyLabel(repo.security_policy)})`));
    for (const policy of allowedPolicies(repo.security_policy)) {
      policySelect.appendChild(Option(policy.value, policy.label));
    }
  }

  function render(): void {
    clear(repoPills);
    for (const entry of entries) {
      const repo = repositories.find((item) => item.id === entry.repository_id);
      const name = repo?.display_name || entry.repository_internal_name || `Repository #${entry.repository_id}`;
      const policy = repo
        ? policyLabel(entry.security_policy_override || repo.security_policy)
        : policyLabel(entry.security_policy_override || '');
      const pill = el('span', { class: 'badge', dataset: { repo: String(entry.repository_id) } },
        `${name} · ${policy}`,
      );
      const remove = el('button', {
        type: 'button',
        title: 'Remove repository access',
      }, '×');
      remove.addEventListener('click', () => {
        const index = entries.indexOf(entry);
        if (index >= 0) entries.splice(index, 1);
        options.onChange();
        render();
      });
      pill.appendChild(remove);
      repoPills.appendChild(pill);
    }
  }

  function setRepositories(next: RepositoryConfig[], loaded: boolean): void {
    repositories = next;
    render();
    clear(repoSelect);
    if (!loaded) {
      repoSelect.appendChild(Option('', 'Loading repositories…'));
      repoSelect.disabled = true;
      repoAdd.disabled = true;
      clear(repoStatus);
      return;
    }
    if (repositories.length === 0) {
      repoSelect.appendChild(Option('', 'No repositories'));
      repoSelect.disabled = true;
      repoAdd.disabled = true;
      clear(repoStatus);
      repoStatus.appendChild(InlineHint('Add a repository on the Repositories screen first.'));
      return;
    }
    repoSelect.disabled = false;
    repoAdd.disabled = false;
    repoSelect.appendChild(Option('', 'Select a repository'));
    for (const repo of repositories) {
      repoSelect.appendChild(Option(String(repo.id), repo.display_name || repo.internal_name));
    }
    updatePolicyOptions();
    clear(repoStatus);
    render();
  }

  function setError(error: Error, retry: () => void): void {
    render();
    setStatus(repoStatus, `Could not load repositories: ${error.message}`, { error: true, retry });
  }

  repoSelect.addEventListener('change', updatePolicyOptions);
  repoAdd.addEventListener('click', () => {
    const repo = repositories.find((item) => String(item.id) === repoSelect.value);
    if (!repo) return;
    const existing = entries.find((entry) => entry.repository_id === repo.id);
    if (existing) {
      const pill = repoPills.querySelector(`[data-repo="${String(repo.id)}"]`);
      if (pill) pill.scrollIntoView({ block: 'nearest' });
      repoStatus.textContent = 'That repository is already available.';
      return;
    }
    entries.push({
      repository_id: repo.id,
      repository_internal_name: repo.internal_name,
      security_policy_override: asPolicy(policySelect.value),
    });
    options.onChange();
    render();
    repoStatus.textContent = '';
  });

  render();

  return {
    section,
    entries,
    setRepositories,
    setError,
  };
}
