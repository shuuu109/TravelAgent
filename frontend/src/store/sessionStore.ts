/**
 * 会话身份与 session 列表的全局状态。
 *
 * 与 chatStore 的关系：
 *   - sessionStore 管：user_id、session 列表、当前 session_id（这些有跨 session 生命周期）
 *   - chatStore 管：当前会话的 messages / finalData / isStreaming（切 session 时被 reset 再回填）
 *   切 session 时由调用方负责协调（loadHistory → chatStore.setMessages/finalData）
 */
import { create } from 'zustand';
import { API_ENDPOINTS, LS_USER_HISTORY, LS_USER_ID, buildApiUrl } from '../config/api';
import type { SessionInfo } from '../types/session';

interface SessionStore {
  userId: string | null;
  sessionList: SessionInfo[];
  currentSessionId: string | null;
  loadingList: boolean;

  // 身份
  setUser: (userId: string) => void;
  logout: () => void;

  // 列表
  refreshList: () => Promise<void>;
  // 切换当前 session（仅设置 id；历史回填由调用方接管）
  setCurrentSession: (sessionId: string | null) => void;

  // CRUD
  createSession: () => Promise<SessionInfo | null>;
  deleteSession: (sessionId: string) => Promise<void>;
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  userId: localStorage.getItem(LS_USER_ID),
  sessionList: [],
  currentSessionId: null,
  loadingList: false,

  setUser: (userId) => {
    localStorage.setItem(LS_USER_ID, userId);
    // 历史 user_id 列表（登录页下拉用）
    const raw = localStorage.getItem(LS_USER_HISTORY);
    const history: string[] = raw ? JSON.parse(raw) : [];
    if (!history.includes(userId)) {
      history.unshift(userId);
      localStorage.setItem(LS_USER_HISTORY, JSON.stringify(history.slice(0, 10)));
    }
    set({ userId, sessionList: [], currentSessionId: null });
  },

  logout: () => {
    localStorage.removeItem(LS_USER_ID);
    set({ userId: null, sessionList: [], currentSessionId: null });
  },

  refreshList: async () => {
    const { userId } = get();
    if (!userId) return;
    set({ loadingList: true });
    try {
      const url = `${buildApiUrl(API_ENDPOINTS.SESSIONS)}?user_id=${encodeURIComponent(userId)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list: SessionInfo[] = await res.json();
      set({ sessionList: list });
    } catch (err) {
      console.error('[sessionStore] refreshList failed', err);
    } finally {
      set({ loadingList: false });
    }
  },

  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),

  createSession: async () => {
    const { userId } = get();
    if (!userId) return null;
    try {
      const res = await fetch(buildApiUrl(API_ENDPOINTS.SESSIONS), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const created: SessionInfo = await res.json();
      // 乐观插入到列表顶部，避免再请求一次
      set((s) => ({ sessionList: [created, ...s.sessionList] }));
      return created;
    } catch (err) {
      console.error('[sessionStore] createSession failed', err);
      return null;
    }
  },

  deleteSession: async (sessionId) => {
    try {
      const res = await fetch(
        `${buildApiUrl(API_ENDPOINTS.SESSIONS)}/${encodeURIComponent(sessionId)}`,
        { method: 'DELETE' },
      );
      if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
      set((s) => ({
        sessionList: s.sessionList.filter((it) => it.session_id !== sessionId),
        currentSessionId: s.currentSessionId === sessionId ? null : s.currentSessionId,
      }));
    } catch (err) {
      console.error('[sessionStore] deleteSession failed', err);
    }
  },
}));
