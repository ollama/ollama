export interface AdminConfigResponse {
  github_client_id: string;
  github_redirect_uri: string;
  github_client_secret_set: boolean;
  api_keys: Array<{ id: string; preview: string }>;
}

export interface AdminConfigUpdate {
  github_client_id?: string;
  github_client_secret?: string;
  github_redirect_uri?: string;
}

export interface GeneratedApiKey {
  api_key: string;
  preview: string;
}

class AdminApiClient {
  private token: string | null = null;
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('admin_token');
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('admin_token', token);
  }

  getToken(): string | null {
    return this.token;
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('admin_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async login(password: string): Promise<{ admin_token: string; expires_in: number }> {
    return this.request('/admin/login', {
      method: 'POST',
      body: JSON.stringify({ password }),
    });
  }

  async getConfig(): Promise<AdminConfigResponse> {
    return this.request('/admin/config');
  }

  async updateConfig(data: AdminConfigUpdate): Promise<{ status: string }> {
    return this.request('/admin/config', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async generateApiKey(): Promise<GeneratedApiKey> {
    return this.request('/admin/api-keys/generate', {
      method: 'POST',
    });
  }

  async revokeApiKey(index: number): Promise<{ status: string }> {
    return this.request(`/admin/api-keys/${index}`, {
      method: 'DELETE',
    });
  }
}

export const adminApi = new AdminApiClient();
