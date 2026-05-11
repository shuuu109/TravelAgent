const RAW_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
export const API_BASE_URL = RAW_BASE.replace(/\/$/, '');

export const API_ENDPOINTS = {
  CHAT_STREAM: '/api/chat/stream',
  CHAT_HISTORY: '/api/chat/history',
  SESSIONS: '/api/sessions',
  HEALTH: '/health',
} as const;

export const buildApiUrl = (endpoint: string): string => `${API_BASE_URL}${endpoint}`;

// localStorage 键
export const LS_USER_ID = 'aligo_user_id';
export const LS_USER_HISTORY = 'aligo_user_history'; // 登录页用过的 user_id 列表
