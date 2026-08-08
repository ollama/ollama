import { useState, useEffect } from 'react';
import { api, GitHubRepo, GitHubContent, GitHubStatus } from '../api';
import '../styles/GitHubBrowser.css';

interface GitHubBrowserProps {
  sessionId: string | null;
  onClose: () => void;
  onRepoLinked?: () => void;
}

export default function GitHubBrowser({ sessionId, onClose, onRepoLinked }: GitHubBrowserProps) {
  const [status, setStatus] = useState<GitHubStatus | null>(null);
  const [repos, setRepos] = useState<GitHubRepo[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<GitHubRepo | null>(null);
  const [currentPath, setCurrentPath] = useState<string>('');
  const [contents, setContents] = useState<GitHubContent[]>([]);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const githubStatus = await api.getGithubStatus();
      setStatus(githubStatus);
      if (githubStatus.connected) {
        loadRepos();
      }
    } catch (err) {
      setError('Failed to load GitHub status');
    }
  };

  const loadRepos = async () => {
    setLoading(true);
    setError(null);
    try {
      const reposList = await api.listGithubRepos();
      setRepos(reposList);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load repositories';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const requireSession = (): boolean => {
    if (!sessionId) {
      setError('Create or select a chat session first, then link a repository.');
      return false;
    }
    return true;
  };

  const handleConnect = async () => {
    try {
      await api.connectGithub();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to start GitHub OAuth';
      setError(message);
    }
  };

  const handleDisconnect = async () => {
    try {
      await api.disconnectGithub();
      setStatus({ connected: false });
      setRepos([]);
      setSelectedRepo(null);
      setContents([]);
      setFileContent(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to disconnect GitHub';
      setError(message);
    }
  };

  const linkRepoToChat = async (repo: GitHubRepo, paths: string[] = []) => {
    if (!requireSession()) return;

    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const [owner, repoName] = repo.full_name.split('/');
      await api.setSessionGithubContext(
        sessionId!,
        owner,
        repoName,
        repo.default_branch,
        paths
      );
      onRepoLinked?.();
      onClose();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to link repository to chat';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleChatWithRepo = async (repo: GitHubRepo) => {
    await linkRepoToChat(repo);
  };

  const handleAddFileToChat = async () => {
    if (!selectedRepo || !currentPath) return;
    if (!requireSession()) return;

    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const [owner, repoName] = selectedRepo.full_name.split('/');
      await api.setSessionGithubContext(
        sessionId!,
        owner,
        repoName,
        selectedRepo.default_branch,
        []
      );
      await api.addSessionGithubFiles(sessionId!, [currentPath]);
      setSuccess(`Added ${currentPath} to chat context`);
      onRepoLinked?.();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to add file to chat';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRepo = async (repo: GitHubRepo) => {
    setSelectedRepo(repo);
    setCurrentPath('');
    setFileContent(null);
    await loadContents(repo, '');
  };

  const loadContents = async (repo: GitHubRepo, path: string) => {
    setLoading(true);
    setError(null);
    try {
      const [owner, repoName] = repo.full_name.split('/');
      const contentsList = await api.getGithubContents(owner, repoName, path);
      setContents(contentsList);
      setCurrentPath(path);
      setFileContent(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load contents';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectContent = async (content: GitHubContent) => {
    if (content.type === 'dir') {
      if (selectedRepo) {
        await loadContents(selectedRepo, content.path);
      }
    } else if (content.type === 'file') {
      setLoading(true);
      setError(null);
      try {
        const [owner, repoName] = selectedRepo!.full_name.split('/');
        const file = await api.getGithubFileContent(owner, repoName, content.path);
        setFileContent(file.content);
        setCurrentPath(content.path);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to load file content';
        setError(message);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleBack = () => {
    if (!selectedRepo) return;

    if (fileContent) {
      setFileContent(null);
      const pathParts = currentPath.split('/').filter(Boolean);
      pathParts.pop();
      const parentPath = pathParts.join('/');
      setCurrentPath(parentPath);
      return;
    }

    const pathParts = currentPath.split('/').filter(Boolean);
    pathParts.pop();
    const newPath = pathParts.join('/');
    loadContents(selectedRepo, newPath);
  };

  const handleBackToRepos = () => {
    setSelectedRepo(null);
    setContents([]);
    setCurrentPath('');
    setFileContent(null);
  };

  if (!status) {
    return (
      <div className="github-browser">
        <div className="github-header">
          <h2>GitHub Browser</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </div>
        <div className="github-loading">Loading...</div>
      </div>
    );
  }

  if (!status.connected) {
    return (
      <div className="github-browser">
        <div className="github-header">
          <h2>GitHub Browser</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </div>
        <div className="github-connect">
          <div className="connect-info">
            <svg width="64" height="64" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <h3>Connect Your GitHub Account</h3>
            <p>Access your repositories and browse code directly from the chat interface.</p>
            <button onClick={handleConnect} className="connect-btn">
              Connect GitHub
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="github-browser">
      <div className="github-header">
        <h2>GitHub Browser</h2>
        <div className="header-actions">
          {status.login && (
            <span className="user-info">
              {status.avatar_url && <img src={status.avatar_url} alt={status.login} />}
              <span>{status.login}</span>
            </span>
          )}
          <button onClick={handleDisconnect} className="disconnect-btn">Disconnect</button>
          <button onClick={onClose} className="close-btn">×</button>
        </div>
      </div>

      {error && (
        <div className="github-error">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {success && (
        <div className="github-success">
          {success}
          <button onClick={() => setSuccess(null)}>×</button>
        </div>
      )}

      <div className="github-content">
        {!selectedRepo ? (
          <div className="repos-list">
            <h3>Your Repositories</h3>
            <p className="repos-hint">Browse a repo or start chatting with one directly.</p>
            {loading ? (
              <div className="loading">Loading repositories...</div>
            ) : repos.length === 0 ? (
              <div className="empty-state">No repositories found</div>
            ) : (
              <div className="repos-grid">
                {repos.map((repo) => (
                  <div key={repo.id} className="repo-card">
                    <div className="repo-card-main" onClick={() => handleSelectRepo(repo)}>
                      <div className="repo-name">{repo.name}</div>
                      {repo.description && (
                        <div className="repo-description">{repo.description}</div>
                      )}
                      <div className="repo-meta">
                        {repo.private && <span className="badge">Private</span>}
                        <span className="updated">
                          Updated {new Date(repo.updated_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    <button
                      type="button"
                      className="chat-repo-btn"
                      onClick={() => handleChatWithRepo(repo)}
                      disabled={loading}
                    >
                      Chat with repo
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="repo-browser">
            <div className="browser-header">
              <button onClick={handleBackToRepos} className="back-btn">
                ← Back to Repositories
              </button>
              <div className="repo-title-row">
                <div className="repo-title">
                  <strong>{selectedRepo.name}</strong>
                  {currentPath && !fileContent && (
                    <span className="current-path">/ {currentPath}</span>
                  )}
                </div>
                <button
                  type="button"
                  className="chat-repo-btn"
                  onClick={() => handleChatWithRepo(selectedRepo)}
                  disabled={loading}
                >
                  Chat with repo
                </button>
              </div>
            </div>

            {fileContent ? (
              <div className="file-viewer">
                <div className="file-header">
                  <button onClick={handleBack} className="back-btn">← Back</button>
                  <span className="file-path">{currentPath}</span>
                  <button
                    type="button"
                    className="add-file-btn"
                    onClick={handleAddFileToChat}
                    disabled={loading}
                  >
                    Add to chat
                  </button>
                </div>
                <pre className="file-content">{fileContent}</pre>
              </div>
            ) : (
              <div className="contents-list">
                {currentPath && (
                  <div className="content-item" onClick={handleBack}>
                    <span className="icon">📁</span>
                    <span className="name">..</span>
                  </div>
                )}
                {loading ? (
                  <div className="loading">Loading...</div>
                ) : (
                  contents.map((content) => (
                    <div
                      key={content.sha}
                      className="content-item"
                      onClick={() => handleSelectContent(content)}
                    >
                      <span className="icon">
                        {content.type === 'dir' ? '📁' : '📄'}
                      </span>
                      <span className="name">{content.name}</span>
                      {content.size && (
                        <span className="size">{(content.size / 1024).toFixed(1)} KB</span>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
