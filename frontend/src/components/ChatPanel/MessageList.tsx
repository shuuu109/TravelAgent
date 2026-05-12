import { useEffect, useRef } from 'react';
import { Alert, Timeline } from 'antd';
import { useChatStore, type Message } from '../../store/chatStore';
import ProgressBubble from './ProgressBubble';
import type { TimelineDay, TimelineEvent } from '../../types/sse';

export default function MessageList() {
  const messages = useChatStore((s) => s.messages);
  const errorText = useChatStore((s) => s.errorText);
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, errorText]);

  return (
    <div
      ref={scrollerRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 16px 24px',
        background: '#fafafa',
      }}
    >
      {messages.length === 0 && (
        <div style={{ color: '#999', textAlign: 'center', marginTop: 80 }}>
          告诉我你的旅行计划，例如出发地、目的地、天数和预算。
        </div>
      )}
      {messages.map((m) => (
        <Bubble key={m.id} message={m} />
      ))}
      {errorText && (
        <Alert
          type="error"
          showIcon
          message={errorText}
          style={{ marginTop: 12 }}
        />
      )}
    </div>
  );
}

function Bubble({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <div style={bubbleStyle('#1677ff', '#fff')}>{message.text}</div>
      </div>
    );
  }

  if (message.kind === 'text') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
        <div style={bubbleStyle('#fff', '#222')}>
          <pre style={preStyle}>{message.text}</pre>
        </div>
      </div>
    );
  }

  if (message.kind === 'progress') {
    return <ProgressBubble message={message} />;
  }

  if (message.kind === 'needs_input') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
        <div style={bubbleStyle('#fffbe6', '#7c5b00')}>
          <pre style={preStyle}>{message.question}</pre>
          {message.missing.length > 0 && (
            <div style={{ marginTop: 6, fontSize: 12, color: '#a37a00' }}>
              缺：{message.missing.join('、')}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.kind === 'timeline') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
        <div style={{ ...bubbleStyle('#fff', '#222'), maxWidth: '92%', paddingTop: 14 }}>
          <TimelineView days={message.days} />
        </div>
      </div>
    );
  }

  if (message.kind === 'weather') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
        <div style={{ ...bubbleStyle('#f0f8ff', '#222'), maxWidth: '92%' }}>
          <WeatherCard summary={message.summary} advice={message.advice} />
        </div>
      </div>
    );
  }

  return null;
}

function WeatherCard({ summary, advice }: { summary: string; advice: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 18 }} aria-hidden>{pickWeatherEmoji(summary)}</span>
        <span style={{ fontWeight: 600 }}>出发当天天气</span>
        <span style={{ color: '#1677ff', fontWeight: 500 }}>{summary}</span>
      </div>
      {advice && (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, color: '#444' }}>
          <span style={{ fontSize: 16, lineHeight: '22px' }} aria-hidden>🧥</span>
          <span style={{ lineHeight: 1.6 }}>{advice}</span>
        </div>
      )}
    </div>
  );
}

// 根据天气简报关键字粗选一个 emoji（前端展示，纯装饰）
function pickWeatherEmoji(summary: string): string {
  const s = summary || '';
  if (/雷|暴雨/.test(s)) return '⛈️';
  if (/雨/.test(s)) return '🌧️';
  if (/雪/.test(s)) return '❄️';
  if (/雾|霾/.test(s)) return '🌫️';
  if (/多云/.test(s)) return '⛅';
  if (/阴/.test(s)) return '☁️';
  if (/晴/.test(s)) return '☀️';
  return '🌤️';
}

function TimelineView({ days }: { days: TimelineDay[] }) {
  const items = days
    .filter((d) => (d.events?.length ?? 0) > 0)
    .map((d) => ({
      color: dotColorForDay(d),
      children: (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ fontWeight: 600, color: '#222', marginBottom: 2 }}>
            {d.label}
          </div>
          {d.events.map((ev, i) => (
            <EventLine key={i} ev={ev} />
          ))}
        </div>
      ),
    }));
  return (
    <Timeline className="chat-timeline" items={items} style={{ marginTop: 4 }} />
  );
}

function EventLine({ ev }: { ev: TimelineEvent }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.7 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        <span aria-hidden>{ev.icon}</span>
        {ev.time && <span style={{ color: '#888' }}>{ev.time}</span>}
        {ev.action && <span style={{ color: '#666' }}>{ev.action}</span>}
        <span style={{ fontWeight: 500 }}>{ev.title}</span>
        {ev.detail && <span style={{ color: '#666' }}>{ev.detail}</span>}
      </div>
      {ev.type === 'hotel' && ev.address && (
        <div
          style={{
            marginLeft: 22,
            color: '#888',
            fontSize: 12,
            wordBreak: 'break-word',
          }}
        >
          <span aria-hidden>📍</span> {ev.address}
        </div>
      )}
    </div>
  );
}

// 当天首个事件类型决定时间轴节点颜色（交通=蓝、酒店=橙、POI=绿）
function dotColorForDay(d: TimelineDay): string {
  const first = d.events?.[0]?.type;
  if (first === 'transport_outbound' || first === 'transport_return') return 'blue';
  if (first === 'hotel') return 'orange';
  return 'green';
}

function bubbleStyle(bg: string, color: string): React.CSSProperties {
  return {
    maxWidth: '78%',
    padding: '10px 14px',
    borderRadius: 12,
    background: bg,
    color,
    boxShadow: '0 1px 2px rgba(0,0,0,0.06)',
    lineHeight: 1.6,
    wordBreak: 'break-word',
  };
}

const preStyle: React.CSSProperties = {
  margin: 0,
  fontFamily: 'inherit',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
};
