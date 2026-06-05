import { useState, useEffect } from 'react';
import Login from './components/Login';
import Chat from './components/Chat';
import Sidebar from './components/Sidebar';
import { api, Session } from './api';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const apiKey = api.getApiKey();
    if (apiKey) {
      setIsAuthenticated(true);
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
    setIsAuthenticated(true);
    await loadSessions();
  };

  const handleLogout = () => {
    api.clearApiKey();
    setIsAuthenticated(false);
    setSessions([]);
    setActiveSessionId(null);
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

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
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
