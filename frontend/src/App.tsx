import ChatPage from './pages/ChatPage';
import LoginPage from './pages/LoginPage';
import { useSessionStore } from './store/sessionStore';

export default function App() {
  const userId = useSessionStore((s) => s.userId);
  return userId ? <ChatPage /> : <LoginPage />;
}
