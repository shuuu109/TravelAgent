// 与后端 app/schemas.py 的 SessionInfo / ChatHistoryResponse 对齐
import type { FinalData } from './sse';

export interface SessionInfo {
  session_id: string;
  user_id: string;
  title: string;
  created_at: number; // 服务端秒级时间戳
  updated_at: number;
  has_final: boolean;
}

export interface HistoryMessage {
  role: 'user' | 'agent';
  text: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: HistoryMessage[];
  last_final_data: FinalData | null;
}
