import { useState, useEffect } from 'react';
import Login from './components/Login';
import AdminLogin from './components/AdminLogin';
import Admin from './components/Admin';
import Chat from './components/Chat';
import Sidebar from './components/Sidebar';
import { api, Session } from './api';
import './App.css';

type ViewMode = 'user-login' | 'admin-login' | 'user-chat' | 'admin-panel';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('user-login');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const apiKey = api.getApiKey();
    const adminToken = api.getAdminToken();
    
    if (adminToken) {
      setViewMode('admin-panel');
      setIsLoading(false);
    } else if (apiKey) {
      setViewMode('user-chat');
      loadSessions();
    } else {
      setIsLoading(false);
    }
  }, []);

  const loadSessions = async () => {
    try {
      const sessionList = await api.listSessions();
      setSessions(sessionList);
      if (sessionList.length > 0 && !activeSessionId) {
        setActiveSessionId(sessionList[0].session_id);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (apiKey: string) => {
    api.setApiKey(apiKey);
    setViewMode('user-chat');
    await loadSessions();
  };

  const handleAdminLogin = async (username: string, password: string) => {
    const response = await api.adminLogin(username, password);
    api.setAdminToken(response.access_token);
    setViewMode('admin-panel');
  };

  const handleLogout = () => {
    api.clearApiKey();
    setSessions([]);
    setActiveSessionId(null);
    setViewMode('user-login');
  };

  const handleAdminLogout = () => {
    api.clearAdminToken();
    setViewMode('user-login');
  };

  const handleNewChat = async () => {
    try {
      const session = await api.createSession('New chat');
      setSessions([session, ...sessions]);
      setActiveSessionId(session.session_id);
    } catch (error) {
      console.error('Failed to create session:', error);
      alert('Failed to create new chat. Please check your API key.');
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await api.deleteSession(sessionId);
      setSessions(sessions.filter((s) => s.session_id !== sessionId));
      if (activeSessionId === sessionId) {
        setActiveSessionId(sessions[0]?.session_id || null);
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  if (viewMode === 'admin-login') {
    return (
      <AdminLogin
        onLogin={handleAdminLogin}
        onBackToUser={() => setViewMode('user-login')}
      />
    );
  }

  if (viewMode === 'admin-panel') {
    return <Admin onLogout={handleAdminLogout} />;
  }

  if (viewMode === 'user-login') {
    return (
      <Login
        onLogin={handleLogin}
        onSwitchToAdmin={() => setViewMode('admin-login')}
      />
    );
  }

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onLogout={handleLogout}
      />
      <div className="main">
        {activeSessionId ? (
          <Chat sessionId={activeSessionId} />
        ) : (
          <div className="empty-state">
            <h2>No active chat</h2>
            <p>Create a new chat to get started</p>
            <button onClick={handleNewChat}>New Chat</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
