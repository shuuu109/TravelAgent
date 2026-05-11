import { useState } from 'react';
import {
  LoadingOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  RightOutlined,
  DownOutlined,
} from '@ant-design/icons';
import type { Message } from '../../store/chatStore';

type ProgressMessage = Extract<Message, { kind: 'progress' }>;

interface Props {
  message: ProgressMessage;
}

// 单行展示策略：
//   进行中：标题=最新一条 running 节点的 label（如"正在理解您的需求..."）+ 转圈
//   完成态：标题=summary（如"已规划完成（16 节点 · 480s）"）+ 绿勾
//   两种状态默认都折叠；点击展开后展示完整节点列表（仅 label，无 phase / data）
export default function ProgressBubble({ message }: Props) {
  const [open, setOpen] = useState(false);
  const finished = message.collapsed;
  const failed = !!message.failed;

  const headerText = finished
    ? message.summary || (failed ? '执行失败' : '已完成')
    : currentLabel(message);

  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
      <div style={containerStyle(failed)}>
        <div style={headerStyle} onClick={() => setOpen((v) => !v)}>
          {failed ? (
            <CloseCircleFilled style={{ color: '#ff4d4f' }} />
          ) : finished ? (
            <CheckCircleFilled style={{ color: '#52c41a' }} />
          ) : (
            <LoadingOutlined style={{ color: '#1677ff' }} spin />
          )}
          <span style={{ flex: 1, color: failed ? '#a8071a' : undefined }}>{headerText}</span>
          <span style={{ color: '#999', fontSize: 12 }}>
            {open ? <DownOutlined /> : <RightOutlined />}
          </span>
        </div>

        {open && message.nodes.length > 0 && (
          <ul style={listStyle}>
            {message.nodes.map((n) => (
              <li key={`${n.phase}-${n.node}`} style={itemStyle}>
                <span style={{ width: 16, display: 'inline-flex' }}>
                  {n.status === 'done' ? (
                    <CheckCircleFilled style={{ color: '#52c41a', fontSize: 13 }} />
                  ) : (
                    <LoadingOutlined style={{ color: '#1677ff', fontSize: 13 }} spin />
                  )}
                </span>
                <span style={{ color: '#333' }}>{n.label}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// 取最新一条 running 节点；若没有 running（极短窗口），取最后一条 done 节点
function currentLabel(msg: ProgressMessage): string {
  const running = [...msg.nodes].reverse().find((n) => n.status === 'running');
  if (running) return `${running.label}...`;
  const lastDone = msg.nodes[msg.nodes.length - 1];
  if (lastDone) return `${lastDone.label}...`;
  return '正在为您规划...';
}

function containerStyle(failed: boolean): React.CSSProperties {
  return {
    maxWidth: '78%',
    padding: '10px 14px',
    borderRadius: 12,
    background: failed ? '#fff1f0' : '#f5f7fa',
    border: `1px solid ${failed ? '#ffa39e' : '#e6e8eb'}`,
    color: '#333',
    lineHeight: 1.6,
  };
}

const headerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  cursor: 'pointer',
  fontSize: 14,
};

const listStyle: React.CSSProperties = {
  margin: '8px 0 0',
  padding: '8px 0 0',
  listStyle: 'none',
  borderTop: '1px dashed #e6e8eb',
};

const itemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '3px 0',
  fontSize: 13,
};
