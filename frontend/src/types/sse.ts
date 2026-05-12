// 与后端 app/schemas.py / app/event_mapper.py 对齐的 SSE 协议类型。

export type EventKind =
  | 'node_start'
  | 'node_complete'
  | 'node_retry'
  | 'needs_input'
  | 'final'
  | 'error';

export interface SSEEnvelope {
  event: EventKind;
  node?: string;
  phase?: string;
  label?: string;
  data?: Record<string, any>;
  ts: number;
}

export interface NeedsInputData {
  question: string;
  missing_info: string[];
}

export interface PlanningFinalData {
  result_type: 'planning';
  final_response: string;
  chat_summary?: ChatSummary | null;
  current_plan: any;
  daily_routes: DailyRoute[];
  daily_restaurants: any[];
  daily_options_by_tier: any[];
  poi_descriptions: Record<string, string>;
  poi_photos: Record<string, string[]>;
  rag_context: any;
  transport_options: any[];
  transport_return_options?: any[];
  budget_fit_message?: string;
}

// 与后端 graph.nodes._respond.chat_summary._build_chat_summary 输出对齐
export interface ChatSummary {
  headline: {
    origin: string;
    destination: string;
    start_date: string;
    end_date: string;
    travel_days: number;
    pax: number;
  };
  timeline: TimelineDay[];
  budget: {
    currency: string;
    total: number;
    limit: number | null;
    fit: string;
    items: any[];
  };
  tips: string[];
  risks: string[];
}

export interface TimelineDay {
  date: string;     // "YYYY-MM-DD" 或空串
  label: string;    // "05-13 周三" 或 "第 N 天"
  events: TimelineEvent[];
}

export interface TimelineEvent {
  type: 'transport_outbound' | 'transport_return' | 'poi' | 'hotel';
  icon: string;
  title: string;
  detail: string;
  time?: string;    // 仅 transport 事件携带
  action?: string;  // 仅 hotel 事件携带（"入住" / "换住"）
}

export interface TextOnlyFinalData {
  result_type: 'text_only';
  final_response: string;
}

export type FinalData = PlanningFinalData | TextOnlyFinalData;

export interface DailyRoute {
  day: number;
  ordered_pois: Array<{
    name: string;
    lng: number;
    lat: number;
    [key: string]: any;
  }>;
  legs: Array<{
    from: string;
    to: string;
    duration: number;
    mode: string;
    steps?: any[];
  }>;
  total_duration: number;
}
