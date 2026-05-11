import MessageList from './MessageList';
import ChatInput from './ChatInput';

export default function ChatPanel() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#fff',
        borderRight: '1px solid #eee',
      }}
    >
      <MessageList />
      <ChatInput />
    </div>
  );
}
