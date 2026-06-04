import { Session } from '../api';
import './Sidebar.css';

interface SidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteSession: (sessionId: string) => void;
  onLogout: () => void;
}

function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onLogout,
}: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>Ollama Gateway</h2>
        <button onClick={onNewChat} className="new-chat-button">
          + New Chat
        </button>
      </div>

      <div className="sessions-list">
        {sessions.map((session) => (
          <div
            key={session.session_id}
            className={`session-item ${
              session.session_id === activeSessionId ? 'active' : ''
            }`}
            onClick={() => onSelectSession(session.session_id)}
          >
            <div className="session-title">{session.title}</div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(session.session_id);
              }}
              className="delete-button"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <button onClick={onLogout} className="logout-button">
          Logout
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
