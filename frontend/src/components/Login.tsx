import { useState, FormEvent } from 'react';
import './Login.css';

interface LoginProps {
  onLogin: (apiKey: string) => void;
  onSwitchToAdmin: () => void;
}

function Login({ onLogin, onSwitchToAdmin }: LoginProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }
    onLogin(apiKey);
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Ollama Gateway</h1>
        <p className="subtitle">Enter your API key to continue</p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="apiKey">API Key</label>
            <input
              type="password"
              id="apiKey"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              autoFocus
            />
          </div>

          {error && <div className="error">{error}</div>}

          <button type="submit" className="login-button">
            Sign In
          </button>
        </form>

        <div className="info">
          <p>Don't have an API key? Contact your administrator.</p>
          <button onClick={onSwitchToAdmin} className="link-button">
            Admin login →
          </button>
        </div>
      </div>
    </div>
  );
}

export default Login;
