import { useEffect, useRef } from 'react';
import { Alert } from 'antd';
import { useChatStore, type Message } from '../../store/chatStore';
import ProgressBubble from './ProgressBubble';

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

  return null;
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
