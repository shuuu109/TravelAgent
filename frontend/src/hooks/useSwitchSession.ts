/**
 * 切换 session 的协调逻辑（跨 sessionStore + chatStore）。
 *
 * 为什么不放进 sessionStore：保持 store 之间无耦合，store 只管自己的状态；
 * 跨 store 的副作用编排（reset chat → fetch history → fill chat）放在 hook 层。
 */
import { useCallback } from 'react';
import { API_ENDPOINTS, buildApiUrl } from '../config/api';
import { useChatStore, newId, type Message } from '../store/chatStore';
import { useSessionStore } from '../store/sessionStore';
import type { ChatHistoryResponse } from '../types/session';

export function useSwitchSession() {
  return useCallback(async (sessionId: string) => {
    const sessionStore = useSessionStore.getState();
    const chatStore = useChatStore.getState();

    if (sessionStore.currentSessionId === sessionId) return;
    if (chatStore.isStreaming) {
      chatStore.setError('请先等待当前请求结束再切换会话');
      return;
    }

    // 切换前先清空，避免新会话短暂展示旧内容
    chatStore.reset();
    sessionStore.setCurrentSession(sessionId);

    try {
      const url = `${buildApiUrl(API_ENDPOINTS.CHAT_HISTORY)}?session_id=${encodeURIComponent(sessionId)}`;
      const res = await fetch(url);
      if (!res.ok) {
        // 404：可能是刚创建还没发过消息 — 不算错误
        if (res.status === 404) return;
        throw new Error(`HTTP ${res.status}`);
      }
      const body: ChatHistoryResponse = await res.json();

      const messages: Message[] = body.messages.map((h) =>
        h.role === 'user'
          ? { id: newId(), role: 'user', text: h.text }
          : { id: newId(), role: 'agent', kind: 'text', text: h.text },
      );

      // 最近一次 planning 的 timeline 气泡：实时流里是在 final 事件那一刻 append 的，
      // 不会写进 checkpointer 的 messages，所以历史回填要从 last_final_data 重建一条。
      const lastFinal = body.last_final_data;
      if (lastFinal?.result_type === 'planning') {
        const days = lastFinal.chat_summary?.timeline;
        if (days && days.length > 0) {
          messages.push({ id: newId(), role: 'agent', kind: 'timeline', days });
        }
      }

      chatStore.setMessages(messages);
      chatStore.setFinalData(lastFinal);
    } catch (err) {
      console.error('[useSwitchSession] load history failed', err);
      chatStore.setError(`加载历史失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }, []);
}
