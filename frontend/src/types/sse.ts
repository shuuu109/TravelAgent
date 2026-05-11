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
  current_plan: any;
  daily_routes: DailyRoute[];
  daily_restaurants: any[];
  daily_options_by_tier: any[];
  poi_descriptions: Record<string, string>;
  poi_photos: Record<string, string[]>;
  rag_context: any;
  transport_options: any[];
  budget_fit_message?: string;
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
