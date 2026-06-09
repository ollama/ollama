export interface Message {
  role: string;
  content: string;
}

export interface Session {
  session_id: string;
  user_id: string;
  title: string;
  model: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: Message;
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface GitHubStatus {
  connected: boolean;
  login?: string;
  avatar_url?: string;
}

export interface GitHubRepo {
  id: number;
  name: string;
  full_name: string;
  description: string | null;
  private: boolean;
  html_url: string;
  default_branch: string;
  updated_at: string;
}

export interface GitHubBranch {
  name: string;
  commit: {
    sha: string;
    url: string;
  };
}

export interface GitHubContent {
  name: string;
  path: string;
  type: string;
  size?: number;
  sha: string;
  download_url?: string;
  html_url: string;
}

export interface GitHubFile {
  path: string;
  name: string;
  size: number;
  content: string;
  sha: string;
}

export interface GitHubContext {
  linked: boolean;
  owner?: string;
  repo?: string;
  full_name?: string;
  ref?: string;
  paths?: string[];
}

class ApiClient {
  private apiKey: string | null = null;
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  setApiKey(key: string) {
    this.apiKey = key;
    localStorage.setItem('ollama_api_key', key);
  }

  getApiKey(): string | null {
    if (!this.apiKey) {
      this.apiKey = localStorage.getItem('ollama_api_key');
    }
    return this.apiKey;
  }

  clearApiKey() {
    this.apiKey = null;
    localStorage.removeItem('ollama_api_key');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const apiKey = this.getApiKey();
    if (!apiKey) {
      throw new Error('API key not set');
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async createSession(title?: string, model: string = 'llama3.2:1b'): Promise<Session> {
    return this.request<Session>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ title, model }),
    });
  }

  async listSessions(): Promise<Session[]> {
    return this.request<Session[]>('/sessions');
  }

  async getSessionMessages(sessionId: string): Promise<{ session_id: string; messages: Message[] }> {
    return this.request(`/sessions/${sessionId}/messages`);
  }

  async sendMessage(
    sessionId: string,
    message: string,
    model?: string
  ): Promise<ChatResponse> {
    return this.request<ChatResponse>(`/sessions/${sessionId}/chat`, {
      method: 'POST',
      body: JSON.stringify({ message, model, stream: false }),
    });
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.request(`/sessions/${sessionId}`, {
      method: 'DELETE',
    });
  }

  async getSessionGithubContext(sessionId: string): Promise<GitHubContext> {
    return this.request<GitHubContext>(`/sessions/${sessionId}/github-context`);
  }

  async setSessionGithubContext(
    sessionId: string,
    owner: string,
    repo: string,
    ref?: string,
    paths?: string[]
  ): Promise<GitHubContext> {
    return this.request<GitHubContext>(`/sessions/${sessionId}/github-context`, {
      method: 'POST',
      body: JSON.stringify({ owner, repo, ref, paths: paths || [] }),
    });
  }

  async addSessionGithubFiles(sessionId: string, paths: string[]): Promise<GitHubContext> {
    return this.request<GitHubContext>(`/sessions/${sessionId}/github-context/files`, {
      method: 'POST',
      body: JSON.stringify({ paths }),
    });
  }

  async clearSessionGithubContext(sessionId: string): Promise<void> {
    await this.request(`/sessions/${sessionId}/github-context`, {
      method: 'DELETE',
    });
  }

  // GitHub OAuth methods
  async getGithubStatus(): Promise<GitHubStatus> {
    return this.request<GitHubStatus>('/auth/github/status');
  }

  async startGithubOAuth(): Promise<{ authorization_url: string }> {
    return this.request<{ authorization_url: string }>('/auth/github/login');
  }

  async connectGithub(): Promise<void> {
    const response = await this.startGithubOAuth();
    window.location.href = response.authorization_url;
  }

  async disconnectGithub(): Promise<void> {
    await this.request('/auth/github/disconnect', {
      method: 'DELETE',
    });
  }

  // GitHub Repository methods
  async listGithubRepos(page: number = 1): Promise<GitHubRepo[]> {
    return this.request<GitHubRepo[]>(`/github/repos?per_page=100&page=${page}`);
  }

  async getGithubRepo(owner: string, repo: string): Promise<any> {
    return this.request(`/github/repos/${owner}/${repo}`);
  }

  async listGithubBranches(owner: string, repo: string): Promise<GitHubBranch[]> {
    return this.request<GitHubBranch[]>(`/github/repos/${owner}/${repo}/branches`);
  }

  async getGithubContents(owner: string, repo: string, path: string = '', ref?: string): Promise<GitHubContent[]> {
    const params = ref ? `?ref=${ref}` : '';
    const result = await this.request<GitHubContent | GitHubContent[]>(
      `/github/repos/${owner}/${repo}/contents/${path}${params}`
    );
    return Array.isArray(result) ? result : [result];
  }

  async getGithubFileContent(owner: string, repo: string, path: string, ref?: string): Promise<GitHubFile> {
    const params = ref ? `?ref=${ref}` : '';
    return this.request<GitHubFile>(`/github/repos/${owner}/${repo}/file/${path}${params}`);
  }
}

export const api = new ApiClient();
