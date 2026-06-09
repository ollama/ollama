import { useState, useEffect } from 'react';
import { api, Message } from '../api';
import MessageRenderer from './MessageRenderer';
import './Chat.css';

interface ChatProps {
  sessionId: string;
}

function Chat({ sessionId }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [codeMode, setCodeMode] = useState(false);

  useEffect(() => {
    loadMessages();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const loadMessages = async () => {
    try {
      const data = await api.getSessionMessages(sessionId);
      setMessages(data.messages);
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setIsLoading(true);

    try {
      await api.sendMessage(sessionId, userMessage);
      await loadMessages();
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <button
          className={`code-mode-toggle ${codeMode ? 'active' : ''}`}
          onClick={() => setCodeMode(!codeMode)}
          title={codeMode ? 'Disable Code Mode' : 'Enable Code Mode'}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="16 18 22 12 16 6"></polyline>
            <polyline points="8 6 2 12 8 18"></polyline>
          </svg>
          <span>Code Mode</span>
        </button>
      </div>

      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <MessageRenderer content={msg.content} codeMode={codeMode} />
          </div>
        ))}
        {isLoading && <div className="message assistant loading">Thinking...</div>}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;
