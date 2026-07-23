import { useState, useEffect } from 'react';
import { api, Message, GitHubContext } from '../api';
import MessageRenderer from './MessageRenderer';
import './Chat.css';

interface ChatProps {
  sessionId: string;
}

function Chat({ sessionId }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [codeMode, setCodeMode] = useState(false);
  const [githubContext, setGithubContext] = useState<GitHubContext | null>(null);

  useEffect(() => {
    loadMessages();
    loadGithubContext();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const loadMessages = async () => {
    try {
      const data = await api.getSessionMessages(sessionId);
      setMessages(data.messages.filter((m) => m.role !== 'system'));
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  const loadGithubContext = async () => {
    try {
      const ctx = await api.getSessionGithubContext(sessionId);
      setGithubContext(ctx);
    } catch (error) {
      console.error('Failed to load GitHub context:', error);
    }
  };

  const handleClearGithubContext = async () => {
    try {
      await api.clearSessionGithubContext(sessionId);
      setGithubContext({ linked: false });
    } catch (error) {
      console.error('Failed to clear GitHub context:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setIsLoading(true);
    setStreamingMessage('');

    // Add user message to local state immediately
    const newUserMessage: Message = { role: 'user', content: userMessage };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      let accumulatedContent = '';
      
      // Use streaming API
      for await (const chunk of api.sendMessageStreaming(sessionId, userMessage)) {
        accumulatedContent += chunk;
        setStreamingMessage(accumulatedContent);
      }

      // Clear streaming message
      setStreamingMessage('');
      
      // Reload messages from server to ensure sync with Redis
      await loadMessages();
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove user message on error
      setMessages(prev => prev.slice(0, -1));
      setStreamingMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        {githubContext?.linked && githubContext.full_name && (
          <div className="repo-context-badge">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span>
              {githubContext.full_name}
              {githubContext.ref && <span className="repo-ref">@{githubContext.ref}</span>}
              {githubContext.paths && githubContext.paths.length > 0 && (
                <span className="repo-files">+{githubContext.paths.length} files</span>
              )}
            </span>
            <button
              type="button"
              className="clear-repo-btn"
              onClick={handleClearGithubContext}
              title="Unlink repository"
            >
              ×
            </button>
          </div>
        )}
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
        {githubContext?.linked && messages.length === 0 && (
          <div className="repo-chat-hint">
            Repository <strong>{githubContext.full_name}</strong> is linked.
            Ask questions about its code, structure, or how to change it.
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <MessageRenderer content={msg.content} codeMode={codeMode} />
          </div>
        ))}
        {streamingMessage && (
          <div className="message assistant">
            <MessageRenderer content={streamingMessage} codeMode={codeMode} />
          </div>
        )}
        {isLoading && !streamingMessage && <div className="message assistant loading">Loading...</div>}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder={
            githubContext?.linked
              ? `Ask about ${githubContext.full_name}...`
              : 'Type a message...'
          }
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
