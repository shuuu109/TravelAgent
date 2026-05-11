import { useMemo, useState } from 'react';
import { AutoComplete, Button, Layout, Typography, message } from 'antd';
import { DownOutlined, LoginOutlined, UserOutlined } from '@ant-design/icons';
import { LS_USER_HISTORY } from '../config/api';
import { useSessionStore } from '../store/sessionStore';

const { Content } = Layout;
const { Title, Text } = Typography;

/**
 * 伪登录：输入 user_id 即进入主页。不做密码 / token，靠 localStorage 记住。
 * 历史 user_id 通过 AutoComplete 下拉箭头复用，避免与输入框堆叠占两行。
 */
export default function LoginPage() {
  const setUser = useSessionStore((s) => s.setUser);

  const options = useMemo(() => {
    const raw = localStorage.getItem(LS_USER_HISTORY);
    const history: string[] = raw ? JSON.parse(raw) : [];
    return history.map((h) => ({ value: h }));
  }, []);

  const [value, setValue] = useState<string>('');

  const submit = () => {
    const id = value.trim();
    if (!id) {
      message.warning('请输入用户 ID');
      return;
    }
    setUser(id);
  };

  return (
    <Layout
      style={{
        minHeight: '100vh',
        background:
          'linear-gradient(135deg, #eef2ff 0%, #f5f7fb 45%, #e6f0ff 100%)',
      }}
    >
      <Content
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 24,
        }}
      >
        <div
          style={{
            width: 420,
            padding: '48px 40px 36px',
            background: '#ffffff',
            borderRadius: 16,
            boxShadow: '0 12px 40px rgba(31, 56, 120, 0.10)',
            border: '1px solid rgba(31, 56, 120, 0.06)',
          }}
        >
          <div style={{ textAlign: 'center', marginBottom: 32 }}>
            <Title
              level={2}
              style={{
                marginBottom: 8,
                fontWeight: 700,
                letterSpacing: 1,
              }}
            >
              Aligo 旅行规划
            </Title>
            <Text type="secondary" style={{ fontSize: 14 }}>
              登录 | 注册
            </Text>
          </div>

          <div style={{ marginBottom: 20 }}>
            <Text
              style={{
                display: 'block',
                marginBottom: 8,
                fontSize: 13,
                color: '#6b7280',
              }}
            >
              用户 ID
            </Text>
            <div style={{ position: 'relative' }}>
              <UserOutlined
                style={{
                  position: 'absolute',
                  left: 14,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: '#9ca3af',
                  fontSize: 16,
                  zIndex: 1,
                  pointerEvents: 'none',
                }}
              />
              <AutoComplete
                value={value}
                options={options}
                onChange={(v) => setValue(v)}
                onSelect={(v) => setValue(v)}
                placeholder="例如：u1"
                allowClear
                suffixIcon={
                  options.length > 0 ? (
                    <DownOutlined style={{ color: '#9ca3af' }} />
                  ) : undefined
                }
                filterOption={(input, option) =>
                  (option?.value as string).toLowerCase().includes(input.toLowerCase())
                }
                onKeyDown={(e) => {
                  if (e.key === 'Enter') submit();
                }}
                style={{ width: '100%' }}
                size="large"
                className="login-autocomplete"
              />
            </div>
          </div>

          <Button
            type="primary"
            icon={<LoginOutlined />}
            block
            size="large"
            onClick={submit}
            style={{
              height: 48,
              fontSize: 16,
              fontWeight: 600,
              borderRadius: 10,
              boxShadow: '0 6px 16px rgba(22, 119, 255, 0.25)',
            }}
          >
            进入
          </Button>
        </div>
      </Content>
    </Layout>
  );
}
