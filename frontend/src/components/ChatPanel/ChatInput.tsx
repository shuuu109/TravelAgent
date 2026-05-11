import { useState, type KeyboardEvent } from 'react';
import { Input, Button, Space } from 'antd';
import { SendOutlined, StopOutlined } from '@ant-design/icons';
import { useChatStream } from '../../hooks/useChatStream';
import { useChatStore } from '../../store/chatStore';

const { TextArea } = Input;

export default function ChatInput() {
  const [value, setValue] = useState('');
  const { send, stop } = useChatStream();
  const isStreaming = useChatStore((s) => s.isStreaming);

  const submit = async () => {
    if (!value.trim() || isStreaming) return;
    const text = value;
    setValue('');
    await send(text);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div style={{ padding: 12, borderTop: '1px solid #eee', background: '#fff' }}>
      <Space.Compact style={{ width: '100%' }}>
        <TextArea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="例如：我想去成都玩 4 天，预算 5000，喜欢人文景点"
          autoSize={{ minRows: 1, maxRows: 5 }}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <Button danger icon={<StopOutlined />} onClick={stop}>
            停止
          </Button>
        ) : (
          <Button type="primary" icon={<SendOutlined />} onClick={submit}>
            发送
          </Button>
        )}
      </Space.Compact>
    </div>
  );
}
