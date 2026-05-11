import { useEffect, useRef, useState } from 'react';
import { Button, Drawer, Layout, Tag, Typography } from 'antd';
import { LogoutOutlined, MenuOutlined } from '@ant-design/icons';
import ChatPanel from '../components/ChatPanel';
import ResultPanel from '../components/ResultPanel';
import Sidebar from '../components/Sidebar';
import { useSessionStore } from '../store/sessionStore';
import { useChatStore } from '../store/chatStore';
import { useSwitchSession } from '../hooks/useSwitchSession';
import { useIsNarrow } from '../hooks/useIsNarrow';

const { Header, Content } = Layout;
const { Title } = Typography;

const SIDEBAR_W_EXPANDED = 220;
const SIDEBAR_W_COLLAPSED = 48;
const CHAT_W = 480;
const DRAWER_W = 260;

export default function ChatPage() {
  const userId = useSessionStore((s) => s.userId);
  const currentSessionId = useSessionStore((s) => s.currentSessionId);
  const sessionList = useSessionStore((s) => s.sessionList);
  const refreshList = useSessionStore((s) => s.refreshList);
  const createSession = useSessionStore((s) => s.createSession);
  const logout = useSessionStore((s) => s.logout);
  const resetChat = useChatStore((s) => s.reset);
  const switchSession = useSwitchSession();
  const initRef = useRef(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const isNarrow = useIsNarrow(1024);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // 用户登录后首次进入：拉列表 → 选首个（或自动建一个）→ 回填历史
  useEffect(() => {
    if (!userId || initRef.current) return;
    initRef.current = true;
    (async () => {
      await refreshList();
      const { sessionList: list, currentSessionId: cur } = useSessionStore.getState();
      if (cur) return;
      if (list.length > 0) {
        await switchSession(list[0].session_id);
      } else {
        const created = await createSession();
        if (created) await switchSession(created.session_id);
      }
    })();
  }, [userId, refreshList, createSession, switchSession]);

  // 切换 session 后自动收起 Drawer（仅窄屏抽屉态下）
  useEffect(() => {
    if (isNarrow && drawerOpen) setDrawerOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId]);

  // 从窄屏切回宽屏时关掉残留的 Drawer
  useEffect(() => {
    if (!isNarrow) setDrawerOpen(false);
  }, [isNarrow]);

  const onLogout = () => {
    initRef.current = false;
    resetChat();
    logout();
  };

  const currentTitle =
    sessionList.find((s) => s.session_id === currentSessionId)?.title || '—';

  return (
    <Layout style={{ height: '100vh' }}>
      <Header
        style={{
          background: '#fff',
          borderBottom: '1px solid #eee',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 16px',
          gap: 8,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          {isNarrow && (
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setDrawerOpen(true)}
            />
          )}
          <Title
            level={4}
            style={{
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            Aligo 旅行规划
          </Title>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <Tag color="blue">user: {userId}</Tag>
          {!isNarrow && <Tag>session: {currentTitle}</Tag>}
          <Button size="small" icon={<LogoutOutlined />} onClick={onLogout}>
            登出
          </Button>
        </div>
      </Header>
      <Content
        style={{
          display: 'flex',
          flexDirection: isNarrow ? 'column' : 'row',
          height: 'calc(100vh - 64px)',
        }}
      >
        {!isNarrow && (
          <div
            style={{
              width: sidebarCollapsed ? SIDEBAR_W_COLLAPSED : SIDEBAR_W_EXPANDED,
              flexShrink: 0,
              height: '100%',
              overflow: 'hidden',
              transition: 'width 0.18s ease',
            }}
          >
            <Sidebar
              collapsed={sidebarCollapsed}
              onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
            />
          </div>
        )}
        <div
          style={
            isNarrow
              ? {
                  width: '100%',
                  flex: '1 1 0',
                  minHeight: 0,
                  borderBottom: '1px solid #eee',
                  overflow: 'hidden',
                }
              : {
                  width: CHAT_W,
                  flexShrink: 0,
                  height: '100%',
                  borderRight: '1px solid #eee',
                  overflow: 'hidden',
                }
          }
        >
          <ChatPanel />
        </div>
        <div
          style={
            isNarrow
              ? {
                  width: '100%',
                  flex: '1 1 0',
                  minHeight: 0,
                  background: '#f5f6fa',
                  overflow: 'hidden',
                }
              : { flex: 1, minWidth: 0, height: '100%', background: '#f5f6fa' }
          }
        >
          <ResultPanel />
        </div>
      </Content>

      <Drawer
        placement="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        width={DRAWER_W}
        styles={{ body: { padding: 0 } }}
        title="会话列表"
      >
        <Sidebar />
      </Drawer>
    </Layout>
  );
}
