import {
  el, clear, loadingState, errorState, notify, confirmAction,
  initNotifications, parseRoute, onRoute, navigate, readString, writeValue,
  conversationLabel, messageOf, type MountOptions, type Route, type ScreenName,
} from './ui.js';
import { api, type ConversationSummary, type DeleteResponse } from './core/api.js';
import { mount as mountChat } from './screens/chat.js';
import { mount as mountImages } from './screens/images.js';
import { mount as mountInference } from './screens/inference.js';
import { mount as mountAgents } from './screens/agents.js';
import { mount as mountProjects } from './screens/projects.js';
import { mount as mountRepositories } from './screens/repositories.js';
import { LineHint, IconButton } from './components.js';

interface Screen {
  element: HTMLElement;
  destroy?: () => void;
  focus?: () => void;
  newChat?: () => void;
}

type ScreenMount = (route: Route, options?: MountOptions) => Screen;

function requireElement(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
}

const main = requireElement('main');
const conversationList = requireElement('conversation-list');
const newChatButton = requireElement('new-chat');

const workspaces: Record<ScreenName, ScreenMount> = {
  chat: mountChat,
  images: mountImages,
  inference: mountInference,
  agents: mountAgents,
  projects: mountProjects,
  repositories: mountRepositories,
};

let activeScreen: Screen | null = null;
let conversationsLoading = false;
let conversationsDirty = false;

initNotifications(document.getElementById('notifications'));

function updateNavigation(route: Route): void {
  for (const link of document.querySelectorAll<HTMLElement>('[data-nav]')) {
    const name = link.dataset.nav;
    link.dataset.active = name === route.name ? 'true' : 'false';
  }
  const openId = parseRoute().id ?? null;
  for (const row of conversationList.querySelectorAll<HTMLElement>('.list-item')) {
    const id = Number(row.dataset.conversationId);
    const selected = openId !== null && id === openId;
    row.dataset.selected = selected ? 'true' : 'false';
  }
}

function renderRoute(route: Route): void {
  activeScreen?.destroy?.();
  activeScreen = null;
  clear(main);
  main.focus({ preventScroll: true });
  const mount = workspaces[route.name];
  activeScreen = mount(route, { onConversationsChanged: loadConversations });
  main.appendChild(activeScreen.element);
  activeScreen.focus?.();
  updateNavigation(route);
  writeValue('lastHash', window.location.hash || '#/chat');
}

function applyRoute(route: Route): void {
  let target = route;
  if (target.invalid) {
    window.history.replaceState(null, '', '#/chat');
    target = parseRoute('#/chat');
  }
  renderRoute(target);
}

onRoute(applyRoute);

async function loadConversations(): Promise<void> {
  if (conversationsLoading) {
    conversationsDirty = true;
    return;
  }
  conversationsLoading = true;
  const hadContent = conversationList.childElementCount > 0;
  if (!hadContent) {
    clear(conversationList);
    conversationList.appendChild(loadingState('Loading conversations…'));
  }
  try {
    const data = await api.get<ConversationSummary[]>('/api/conversations');
    const conversations = Array.isArray(data) ? data : [];
    clear(conversationList);
    if (conversations.length === 0) {
      conversationList.appendChild(LineHint('No conversations yet. Start one below.'));
      updateNavigation(parseRoute());
      return;
    }
    for (const conversation of conversations) {
      const link = el('a', {
        href: `#/chat/${conversation.id}`,
        title: conversationLabel(conversation),
      }, conversationLabel(conversation));
      const remove = IconButton('×', { title: 'Delete conversation' });
      remove.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        deleteConversation(conversation);
      });
      const row = el('div', {
        class: 'list-item',
        dataset: { conversationId: String(conversation.id) },
      }, el('div', { class: 'list-item-main' }, link), remove);
      conversationList.appendChild(row);
    }
    updateNavigation(parseRoute());
  } catch (error) {
    clear(conversationList);
    conversationList.appendChild(errorState(messageOf(error) || 'Could not load conversations.', loadConversations));
  } finally {
    conversationsLoading = false;
    if (conversationsDirty) {
      conversationsDirty = false;
      loadConversations();
    }
  }
}

async function deleteConversation(conversation: ConversationSummary): Promise<void> {
  const confirmed = await confirmAction({
    title: 'Delete conversation?',
    message: `“${conversationLabel(conversation)}” and its messages will be deleted. This cannot be undone.`,
    confirmLabel: 'Delete conversation',
  });
  if (!confirmed) return;
  try {
    await api.del<DeleteResponse>(`/api/conversations/${conversation.id}`);
    notify('Conversation deleted.', { level: 'success' });
    if (parseRoute().id === conversation.id) navigate('#/chat');
    await loadConversations();
  } catch (error) {
    notify(messageOf(error) || 'Could not delete the conversation.', { level: 'error' });
  }
}

newChatButton.addEventListener('click', () => {
  if (window.location.hash === '#/chat' && activeScreen?.newChat) {
    activeScreen.newChat();
    return;
  }
  navigate('#/chat');
});

window.addEventListener('hashchange', () => {
  writeValue('lastHash', window.location.hash);
});

const initialHash = window.location.hash || readString('lastHash', '#/chat') || '#/chat';
if (initialHash !== window.location.hash) {
  window.history.replaceState(null, '', initialHash);
}
loadConversations();
applyRoute(parseRoute());
