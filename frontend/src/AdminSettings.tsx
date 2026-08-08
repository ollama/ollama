import { useState, useEffect } from 'react';
import { adminApi, AdminConfigResponse, GeneratedApiKey } from './adminApi';
import './Admin.css';

interface AdminSettingsProps {
  onLogout: () => void;
}

function AdminSettings({ onLogout }: AdminSettingsProps) {
  const [config, setConfig] = useState<AdminConfigResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [generatedKey, setGeneratedKey] = useState<GeneratedApiKey | null>(null);
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // Form fields
  const [githubClientId, setGithubClientId] = useState('');
  const [githubClientSecret, setGithubClientSecret] = useState('');
  const [githubRedirectUri, setGithubRedirectUri] = useState('');

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      setError('');
      const data = await adminApi.getConfig();
      setConfig(data);
      setGithubClientId(data.github_client_id);
      setGithubRedirectUri(data.github_redirect_uri || 'http://localhost:8080/api/auth/github/callback');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load config');
      if (err instanceof Error && err.message.includes('401')) {
        onLogout();
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveGithub = async () => {
    setIsSaving(true);
    setError('');
    setSuccess('');

    try {
      await adminApi.updateConfig({
        github_client_id: githubClientId,
        github_client_secret: githubClientSecret || undefined,
        github_redirect_uri: githubRedirectUri,
      });
      
      setSuccess('GitHub settings saved successfully');
      setGithubClientSecret(''); // Clear secret field after save
      await loadConfig();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  const handleGenerateKey = async () => {
    setError('');
    setSuccess('');
    setGeneratedKey(null);
    setIsGenerating(true);

    try {
      const result = await adminApi.generateApiKey();
      setGeneratedKey(result);
      setShowKeyModal(true); // Show modal immediately
      await loadConfig();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate key');
      console.error('Generate key error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRevokeKey = async (index: number) => {
    if (!confirm('Are you sure you want to revoke this API key? Users using this key will lose access.')) {
      return;
    }

    setError('');
    setSuccess('');

    try {
      await adminApi.revokeApiKey(index);
      setSuccess('API key revoked successfully');
      await loadConfig();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke key');
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      // Try modern clipboard API first
      await navigator.clipboard.writeText(text);
      setSuccess('✅ Copied to clipboard!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      // Fallback for browsers that block clipboard API (HTTP non-localhost)
      try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        textarea.style.top = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        
        if (successful) {
          setSuccess('✅ Copied to clipboard!');
          setTimeout(() => setSuccess(''), 3000);
        } else {
          setError('Failed to copy. Please select and copy manually.');
        }
      } catch (fallbackErr) {
        setError('Clipboard access denied. Please copy manually.');
      }
    }
  };

  const closeKeyModal = () => {
    setShowKeyModal(false);
    setGeneratedKey(null);
  };

  if (isLoading) {
    return (
      <div className="admin-loading">
        <div className="spinner"></div>
        <p>Loading configuration...</p>
      </div>
    );
  }

  return (
    <div className="admin-settings">
      <header className="admin-header">
        <h1>Super Admin Settings</h1>
        <button onClick={onLogout} className="admin-logout-button">
          Logout
        </button>
      </header>

      {error && <div className="admin-error">{error}</div>}
      {success && <div className="admin-success">{success}</div>}

      {/* API Key Modal */}
      {showKeyModal && generatedKey && (
        <div className="modal-overlay" onClick={closeKeyModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>🔑 New API Key Generated</h2>
              <button onClick={closeKeyModal} className="modal-close">×</button>
            </div>
            <div className="modal-body">
              <div className="key-warning">
                ⚠️ <strong>IMPORTANT:</strong> Copy this key now! It will not be shown again.
              </div>
              <div className="key-display-modal">
                <input
                  type="text"
                  readOnly
                  value={generatedKey.api_key}
                  onClick={(e) => e.currentTarget.select()}
                  className="key-input-large"
                  autoFocus
                />
              </div>
              <div className="modal-actions">
                <button onClick={() => copyToClipboard(generatedKey.api_key)} className="copy-button-large">
                  📋 Copy to Clipboard
                </button>
                <button onClick={closeKeyModal} className="close-button-large">
                  I've Saved It
                </button>
              </div>
              <div className="key-info">
                <p><strong>Preview:</strong> <code>{generatedKey.preview}</code></p>
                <p className="help-text">Share this key with users who need access to the chat interface at <code>/</code></p>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="admin-content">
        {/* GitHub OAuth Settings */}
        <section className="admin-section">
          <h2>GitHub OAuth Settings</h2>
          <p className="section-description">
            Configure GitHub OAuth for repository integration. Register your app at{' '}
            <a href="https://github.com/settings/developers" target="_blank" rel="noreferrer">
              GitHub Developer Settings
            </a>
          </p>

          <div className="form-group">
            <label htmlFor="github-client-id">Client ID</label>
            <input
              type="text"
              id="github-client-id"
              value={githubClientId}
              onChange={(e) => setGithubClientId(e.target.value)}
              placeholder="Enter GitHub Client ID"
            />
          </div>

          <div className="form-group">
            <label htmlFor="github-client-secret">
              Client Secret
              {config?.github_client_secret_set && (
                <span className="hint"> (currently set)</span>
              )}
            </label>
            <input
              type="password"
              id="github-client-secret"
              value={githubClientSecret}
              onChange={(e) => setGithubClientSecret(e.target.value)}
              placeholder={config?.github_client_secret_set ? 'Leave blank to keep existing' : 'Enter GitHub Client Secret'}
            />
          </div>

          <div className="form-group">
            <label htmlFor="github-redirect-uri">Redirect URI</label>
            <input
              type="text"
              id="github-redirect-uri"
              value={githubRedirectUri}
              onChange={(e) => setGithubRedirectUri(e.target.value)}
              placeholder="http://localhost:8080/api/auth/github/callback"
            />
            <small>Must match exactly in your GitHub OAuth app settings</small>
          </div>

          <button
            onClick={handleSaveGithub}
            className="admin-save-button"
            disabled={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save GitHub Settings'}
          </button>
        </section>

        {/* API Keys Management */}
        <section className="admin-section">
          <h2>API Keys Management</h2>
          <p className="section-description">
            Generate and manage API keys for users to access the gateway
          </p>

          <button onClick={handleGenerateKey} className="admin-generate-button" disabled={isGenerating}>
            {isGenerating ? '⏳ Generating...' : '+ Generate New API Key'}
          </button>

          <div className="api-keys-list">
            <h3>Active API Keys ({config?.api_keys.length || 0})</h3>
            {config?.api_keys && config.api_keys.length > 0 ? (
              <table className="keys-table">
                <thead>
                  <tr>
                    <th>Key Preview</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {config.api_keys.map((key, index) => (
                    <tr key={key.id}>
                      <td>
                        <code>{key.preview}</code>
                      </td>
                      <td>
                        <div className="key-actions">
                          <button
                            onClick={() => copyToClipboard(key.preview)}
                            className="admin-copy-preview-button"
                            title="Copy preview (last 4 chars)"
                          >
                            📋
                          </button>
                          <button
                            onClick={() => handleRevokeKey(index)}
                            className="admin-revoke-button"
                          >
                            Revoke
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="no-keys">No API keys configured. Generate one to get started.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default AdminSettings;
