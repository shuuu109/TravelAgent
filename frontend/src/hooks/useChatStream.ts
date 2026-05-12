import { useCallback } from 'react';
import { buildApiUrl, API_ENDPOINTS } from '../config/api';
import { useChatStore, newId, type Message } from '../store/chatStore';
import { useSessionStore } from '../store/sessionStore';
import type { SSEEnvelope, FinalData, NeedsInputData } from '../types/sse';

// 同一时间只允许一条流（由 isStreaming 兜底）；模块级 controller 即可。
let currentAbort: AbortController | null = null;
type AbortReason = 'user' | 'timeout' | null;
let abortReason: AbortReason = null;

// 兜底超时：planning 实测 5-8 分钟，留出 2x 余量。
const REQUEST_TIMEOUT_MS = 15 * 60 * 1000;

export function useChatStream() {
  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const store = useChatStore.getState();
    if (store.isStreaming) return;

    // 必须先登录 + 选中（或自动创建）session
    const { userId, currentSessionId } = useSessionStore.getState();
    if (!userId || !currentSessionId) {
      store.setError('请先选择或新建一个会话');
      return;
    }

    // 1) 用户消息
    const userMsg: Message = { id: newId(), role: 'user', text: trimmed };
    store.appendMessage(userMsg);

    // 2) 进度占位（每轮独占一条；id 随该轮 send 闭包共享）
    const progressId = newId();
    store.appendMessage({
      id: progressId,
      role: 'agent',
      kind: 'progress',
      nodes: [],
      collapsed: false,
      startedAt: Date.now(),
    });

    store.setStreaming(true);
    store.setError(null);

    // needs_input 已经把 final_response 渲染成黄色追问气泡，
    // 后续 final 事件携带的同一段文案就不应再追加白色文本气泡，否则同样内容会出现两次。
    const streamFlags = { needsInputShown: false };

    // 3) 准备 abort + 兜底超时
    const controller = new AbortController();
    currentAbort = controller;
    abortReason = null;
    const timeoutId = window.setTimeout(() => {
      if (!controller.signal.aborted) {
        abortReason = 'timeout';
        controller.abort();
      }
    }, REQUEST_TIMEOUT_MS);

    let res: Response;
    try {
      res = await fetch(buildApiUrl(API_ENDPOINTS.CHAT_STREAM), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: currentSessionId,
          message: trimmed,
        }),
        signal: controller.signal,
      });
    } catch (e) {
      handleAbortOrError(e, progressId);
      cleanup(timeoutId);
      return;
    }

    if (!res.ok || !res.body) {
      store.setError(`后端响应异常：HTTP ${res.status}`);
      failProgress(progressId, `请求失败（HTTP ${res.status}）`);
      cleanup(timeoutId);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // sse-starlette 默认用 \r\n 作为行分隔符，事件之间是 \r\n\r\n；
        // 这里用正则同时兼容 \n\n / \r\n\r\n / \r\r。
        const parts = buffer.split(/\r?\n\r?\n|\r\r/);
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part
            .split(/\r?\n/)
            .find((l) => l.startsWith('data:'));
          if (!line) continue;
          // data: 后面通常有一个空格，但规范允许没有
          const payload = line.slice(5).trimStart();
          try {
            const env: SSEEnvelope = JSON.parse(payload);
            handleEnvelope(env, progressId, streamFlags);
          } catch (err) {
            // 不再静默吞 — 帮助排查后端推送格式问题
            console.warn('[SSE] parse failed', err, payload);
          }
        }
      }
    } catch (e) {
      handleAbortOrError(e, progressId);
    } finally {
      cleanup(timeoutId);
    }
  }, []);

  const stop = useCallback(() => {
    if (currentAbort && !currentAbort.signal.aborted) {
      abortReason = 'user';
      currentAbort.abort();
    }
  }, []);

  return { send, stop };
}

// ─────────────────────────────────────────────────────────────────────────────
// abort / 错误统一收口
// ─────────────────────────────────────────────────────────────────────────────

function isAbortError(e: unknown): boolean {
  return e instanceof Error && (e.name === 'AbortError' || e.name === 'TypeError' && /abort/i.test(e.message));
}

function handleAbortOrError(e: unknown, progressId: string) {
  const store = useChatStore.getState();
  if (isAbortError(e)) {
    if (abortReason === 'user') {
      failProgress(progressId, '已取消');
      // 取消是用户主动行为，不弹红条提示
    } else if (abortReason === 'timeout') {
      failProgress(progressId, '请求超时');
      store.setError('请求超时（15 分钟内无响应）');
    } else {
      // 不应发生：abort 但没有 reason
      failProgress(progressId, '已中断');
    }
    return;
  }
  const msg = e instanceof Error ? e.message : String(e);
  store.setError(`流读取失败：${msg}`);
  failProgress(progressId, '执行失败');
}

function cleanup(timeoutId: number) {
  window.clearTimeout(timeoutId);
  currentAbort = null;
  abortReason = null;
  useChatStore.getState().setStreaming(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE 事件分发
// ─────────────────────────────────────────────────────────────────────────────

type StreamFlags = { needsInputShown: boolean };

function handleEnvelope(
  env: SSEEnvelope,
  progressId: string,
  flags: StreamFlags,
): void {
  const store = useChatStore.getState();

  switch (env.event) {
    case 'node_start':
      mutateProgressNodes(progressId, (nodes) =>
        upsertNode(nodes, {
          node: env.node || '',
          phase: env.phase || '',
          label: env.label || env.node || '',
          status: 'running',
        }),
      );
      return;

    case 'node_complete':
      mutateProgressNodes(progressId, (nodes) =>
        upsertNode(nodes, {
          node: env.node || '',
          phase: env.phase || '',
          label: env.label || env.node || '',
          status: 'done',
          data: env.data,
        }),
      );
      return;

    case 'needs_input': {
      const d = (env.data || {}) as NeedsInputData;
      collapseProgress(progressId, '需要您补充信息');
      store.appendMessage({
        id: newId(),
        role: 'agent',
        kind: 'needs_input',
        question: d.question || '',
        missing: d.missing_info || [],
      });
      flags.needsInputShown = true;
      return;
    }

    case 'final': {
      const data = env.data as FinalData | undefined;
      if (!data) return;
      store.setFinalData(data);
      finalizeProgress(progressId, data);
      // needs_input 已渲染同一段追问文案，避免重复白色气泡
      if (data.final_response && !flags.needsInputShown) {
        store.appendMessage({
          id: newId(),
          role: 'agent',
          kind: 'text',
          text: data.final_response,
        });
      }
      // planning 路径：headline 文本气泡之后，追加一条 timeline 表格气泡
      if (data.result_type === 'planning') {
        const days = data.chat_summary?.timeline;
        if (days && days.length > 0) {
          store.appendMessage({
            id: newId(),
            role: 'agent',
            kind: 'timeline',
            days,
          });
        }
      }
      return;
    }

    case 'error':
      store.setError((env.data?.message as string) || '未知错误');
      failProgress(progressId, '执行失败');
      return;

    case 'node_retry':
      // 后端目前不推送；预留位，等回环事件接上后再加重试角标
      return;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// progress 消息辅助操作
// ─────────────────────────────────────────────────────────────────────────────

type ProgressNode = Extract<Message, { kind: 'progress' }>['nodes'][number];

function upsertNode(nodes: ProgressNode[], next: ProgressNode): ProgressNode[] {
  const idx = nodes.findIndex((n) => n.node === next.node);
  if (idx === -1) return [...nodes, next];
  // 已存在：保留先到达的字段，用 next 覆盖 status / data / label
  const merged: ProgressNode = { ...nodes[idx], ...next };
  const copy = nodes.slice();
  copy[idx] = merged;
  return copy;
}

function mutateProgressNodes(
  id: string,
  fn: (nodes: ProgressNode[]) => ProgressNode[],
): void {
  useChatStore.getState().updateMessage(id, (prev) => {
    if (prev.role !== 'agent' || prev.kind !== 'progress') return prev;
    return { ...prev, nodes: fn(prev.nodes) };
  });
}

function collapseProgress(id: string, summary: string): void {
  useChatStore.getState().updateMessage(id, (prev) => {
    if (prev.role !== 'agent' || prev.kind !== 'progress') return prev;
    return { ...prev, collapsed: true, summary };
  });
}

// 异常路径专用：折叠 + summary + 红色样式
function failProgress(id: string, summary: string): void {
  useChatStore.getState().updateMessage(id, (prev) => {
    if (prev.role !== 'agent' || prev.kind !== 'progress') return prev;
    return { ...prev, collapsed: true, failed: true, summary };
  });
}

function finalizeProgress(id: string, data: FinalData): void {
  const store = useChatStore.getState();
  const msg = store.messages.find((m) => m.id === id);
  if (!msg || msg.role !== 'agent' || msg.kind !== 'progress') return;

  // 非 planning 且 progress 只跑了 intent 一个节点 -> 直接删除占位，
  // 与产品决策一致：preference / memory / info / unknown 不显示进度块
  const isTrivialTextOnly =
    data.result_type === 'text_only' &&
    msg.nodes.length <= 1 &&
    (msg.nodes[0]?.node === 'intent' || msg.nodes.length === 0);

  if (isTrivialTextOnly) {
    store.removeMessage(id);
    return;
  }

  const elapsedSec = ((Date.now() - msg.startedAt) / 1000).toFixed(1);
  const summary = `已规划完成（${msg.nodes.length} 节点 · ${elapsedSec}s）`;
  collapseProgress(id, summary);
}
