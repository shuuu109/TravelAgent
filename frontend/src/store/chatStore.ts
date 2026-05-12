import { create } from 'zustand';
import type { FinalData, TimelineDay } from '../types/sse';

export type Message =
  | { id: string; role: 'user'; text: string }
  | { id: string; role: 'agent'; kind: 'text'; text: string }
  | {
      id: string;
      role: 'agent';
      kind: 'progress';
      nodes: Array<{
        node: string;
        phase: string;
        label: string;
        status: 'running' | 'done';
        data?: any;
      }>;
      collapsed: boolean;
      // failed=true 时进度块用红色样式（用户取消 / 超时 / 后端 error 事件）
      failed?: boolean;
      summary?: string;
      startedAt: number;
    }
  | {
      id: string;
      role: 'agent';
      kind: 'needs_input';
      question: string;
      missing: string[];
    }
  | {
      id: string;
      role: 'agent';
      kind: 'timeline';
      days: TimelineDay[];
    };

interface ChatStore {
  messages: Message[];
  isStreaming: boolean;
  finalData: FinalData | null;
  errorText: string | null;

  appendMessage: (m: Message) => void;
  updateMessage: (id: string, patch: (prev: Message) => Message) => void;
  removeMessage: (id: string) => void;
  setMessages: (msgs: Message[]) => void;
  setStreaming: (v: boolean) => void;
  setFinalData: (d: FinalData | null) => void;
  setError: (e: string | null) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isStreaming: false,
  finalData: null,
  errorText: null,

  appendMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
  updateMessage: (id, patch) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? patch(m) : m)),
    })),
  removeMessage: (id) =>
    set((s) => ({ messages: s.messages.filter((m) => m.id !== id) })),
  setMessages: (msgs) => set({ messages: msgs }),
  setStreaming: (v) => set({ isStreaming: v }),
  setFinalData: (d) => set({ finalData: d }),
  setError: (e) => set({ errorText: e }),
  reset: () =>
    set({ messages: [], isStreaming: false, finalData: null, errorText: null }),
}));

export const newId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
