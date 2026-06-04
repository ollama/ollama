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

export interface User {
  user_id: string;
  display_name: string;
  role: string;
  created_at: string;
  disabled: string;
}

export interface ApiKey {
  key_id: string;
  user_id: string;
  label: string;
  prefix: string;
  created_by: string;
  created_at: string;
  last_used_at: string;
  revoked: string;
}

export interface AdminLoginResponse {
  access_token: string;
  token_type: string;
}

export interface ApiKeyCreateResponse {
  api_key: string;
  metadata: ApiKey;
}

class ApiClient {
  private apiKey: string | null = null;
  private adminToken: string | null = null;
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

  setAdminToken(token: string) {
    this.adminToken = token;
    localStorage.setItem('ollama_admin_token', token);
  }

  getAdminToken(): string | null {
    if (!this.adminToken) {
      this.adminToken = localStorage.getItem('ollama_admin_token');
    }
    return this.adminToken;
  }

  clearAdminToken() {
    this.adminToken = null;
    localStorage.removeItem('ollama_admin_token');
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

  private async adminRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const token = this.getAdminToken();
    if (!token) {
      throw new Error('Admin token not set');
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    if (response.status === 204) {
      return {} as T;
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

  // Admin endpoints
  async adminLogin(username: string, password: string): Promise<AdminLoginResponse> {
    const response = await fetch(`${this.baseUrl}/admin/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Login failed: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async adminListUsers(): Promise<{ users: User[] }> {
    return this.adminRequest<{ users: User[] }>('/admin/users');
  }

  async adminCreateUser(userId: string, displayName?: string): Promise<User> {
    return this.adminRequest<User>('/admin/users', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, display_name: displayName }),
    });
  }

  async adminDisableUser(userId: string): Promise<void> {
    await this.adminRequest(`/admin/users/${userId}/disable`, {
      method: 'PATCH',
    });
  }

  async adminListApiKeys(userId?: string): Promise<{ api_keys: ApiKey[] }> {
    const query = userId ? `?user_id=${userId}` : '';
    return this.adminRequest<{ api_keys: ApiKey[] }>(`/admin/api-keys${query}`);
  }

  async adminCreateApiKey(userId: string, label?: string): Promise<ApiKeyCreateResponse> {
    return this.adminRequest<ApiKeyCreateResponse>('/admin/api-keys', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, label }),
    });
  }

  async adminRevokeApiKey(keyId: string): Promise<void> {
    await this.adminRequest(`/admin/api-keys/${keyId}`, {
      method: 'DELETE',
    });
  }
}

export const api = new ApiClient();
