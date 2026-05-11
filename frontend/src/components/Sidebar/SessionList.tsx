import { useMemo, useState } from 'react';
import { Button, Empty, Modal, Tooltip } from 'antd';
import {
  DeleteOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MessageOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import { useSessionStore } from '../../store/sessionStore';
import { useSwitchSession } from '../../hooks/useSwitchSession';
import { useChatStore } from '../../store/chatStore';
import type { SessionInfo } from '../../types/session';

interface SessionListProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export default function SessionList({ collapsed = false, onToggleCollapse }: SessionListProps) {
  const sessionList = useSessionStore((s) => s.sessionList);
  const currentSessionId = useSessionStore((s) => s.currentSessionId);
  const createSession = useSessionStore((s) => s.createSession);
  const deleteSession = useSessionStore((s) => s.deleteSession);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const switchSession = useSwitchSession();

  const onCreate = async () => {
    if (isStreaming) return;
    const created = await createSession();
    if (created) await switchSession(created.session_id);
  };

  const onDelete = (item: SessionInfo) => {
    Modal.confirm({
      title: '删除会话',
      content: `确认删除「${item.title}」？该会话的全部历史与规划结果将被清除。`,
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        const wasCurrent = item.session_id === currentSessionId;
        await deleteSession(item.session_id);
        if (wasCurrent) {
          // 切到第一个剩下的；都没了就新建一个
          const { sessionList: rest } = useSessionStore.getState();
          if (rest.length > 0) {
            await switchSession(rest[0].session_id);
          } else {
            const created = await createSession();
            if (created) await switchSession(created.session_id);
          }
        }
      },
    });
  };

  if (collapsed) {
    return (
      <div style={collapsedContainerStyle}>
        <Tooltip title="展开会话列表" placement="right">
          <Button
            type="text"
            icon={<MenuUnfoldOutlined />}
            onClick={onToggleCollapse}
            style={{ marginBottom: 4 }}
          />
        </Tooltip>
        <Tooltip title="新对话" placement="right">
          <Button
            type="text"
            icon={<PlusOutlined />}
            disabled={isStreaming}
            onClick={onCreate}
          />
        </Tooltip>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={onCreate}
            disabled={isStreaming}
            style={{ flex: 1, minWidth: 0 }}
          >
            新对话
          </Button>
          {onToggleCollapse && (
            <Tooltip title="折叠">
              <Button
                type="text"
                icon={<MenuFoldOutlined />}
                onClick={onToggleCollapse}
              />
            </Tooltip>
          )}
        </div>
      </div>

      <div style={listScrollStyle}>
        {sessionList.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无会话"
            style={{ marginTop: 40 }}
          />
        ) : (
          sessionList.map((item) => (
            <SessionRow
              key={item.session_id}
              item={item}
              active={item.session_id === currentSessionId}
              disabled={isStreaming && item.session_id !== currentSessionId}
              onClick={() => switchSession(item.session_id)}
              onDelete={() => onDelete(item)}
            />
          ))
        )}
      </div>
    </div>
  );
}

interface RowProps {
  item: SessionInfo;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  onDelete: () => void;
}

function SessionRow({ item, active, disabled, onClick, onDelete }: RowProps) {
  const [hover, setHover] = useState(false);
  const timeStr = useMemo(() => formatTime(item.updated_at), [item.updated_at]);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={disabled ? undefined : onClick}
      style={rowStyle(active, disabled)}
    >
      <MessageOutlined style={{ color: active ? '#1677ff' : '#999', flexShrink: 0 }} />
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <div
          style={{
            color: active ? '#1677ff' : '#333',
            fontSize: 13,
            fontWeight: active ? 500 : 400,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
          title={item.title}
        >
          {item.title}
        </div>
        <div style={{ color: '#999', fontSize: 11, marginTop: 2 }}>{timeStr}</div>
      </div>
      {hover && !disabled && (
        <Tooltip title="删除">
          <Button
            type="text"
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          />
        </Tooltip>
      )}
    </div>
  );
}

// updated_at 是秒级 epoch；同一天显示 HH:mm，否则 MM-DD
function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) {
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const pad = (n: number) => n.toString().padStart(2, '0');

const containerStyle: React.CSSProperties = {
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
  background: '#fafafa',
  borderRight: '1px solid #eee',
  overflow: 'hidden',
};

const collapsedContainerStyle: React.CSSProperties = {
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '8px 0',
  background: '#fafafa',
  borderRight: '1px solid #eee',
};

const headerStyle: React.CSSProperties = {
  padding: 12,
  borderBottom: '1px solid #eee',
};

const listScrollStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '8px 6px',
};

function rowStyle(active: boolean, disabled: boolean): React.CSSProperties {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '8px 10px',
    borderRadius: 6,
    marginBottom: 2,
    cursor: disabled ? 'not-allowed' : 'pointer',
    background: active ? '#e6f4ff' : 'transparent',
    opacity: disabled ? 0.5 : 1,
    transition: 'background 0.15s',
  };
}
