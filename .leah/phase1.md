# PHASE 1: TEMEL ALTYAPI - Multi-API Support, Context Management, Config System

## 📋 GENEL BAKIŞ
Bu phase'de Ollama'ya çoklu API desteği, gelişmiş context yönetimi ve yapılandırma sistemi ekleniyor. Bu temel altyapı, sonraki tüm phase'lerin temeli olacak.

## 🎯 HEDEFLER
1. ✅ OpenAI, Anthropic, Google, Groq, Z.AI gibi dış API'leri destekleme
2. ✅ Her API için model seçimi yapabilme
3. ✅ Context boyutu ayarlama ve otomatik özetleme
4. ✅ API maliyet hesaplama altyapısı
5. ✅ Token/saniye ve context bilgilerini tracking
6. ✅ Performans odaklı config sistemi

---

## 🏗️ MİMARİ TASARIM

### Yeni Klasör Yapısı
```
/home/user/ollama/
├── api/
│   ├── providers/              # YENİ: API provider'lar
│   │   ├── base.go            # Base provider interface
│   │   ├── ollama.go          # Mevcut Ollama provider
│   │   ├── openai.go          # OpenAI provider
│   │   ├── anthropic.go       # Anthropic provider
│   │   ├── google.go          # Google Gemini provider
│   │   ├── groq.go            # Groq provider
│   │   ├── custom.go          # Custom API provider
│   │   └── registry.go        # Provider registry
│   ├── pricing/               # YENİ: Maliyet hesaplama
│   │   ├── pricing.go         # Pricing calculator
│   │   └── models.json        # Model fiyat tablosu
│   └── context/               # YENİ: Context yönetimi
│       ├── manager.go         # Context manager
│       ├── summarizer.go      # Auto-summarization
│       └── window.go          # Sliding window
├── app/store/
│   ├── providers.go           # YENİ: API provider storage
│   └── schema.sql             # GÜNCELLENECEK: Yeni tablolar
└── app/ui/app/src/
    ├── api/
    │   ├── providers.ts       # YENİ: Provider API client
    │   └── pricing.ts         # YENİ: Pricing calculator
    ├── hooks/
    │   ├── useProviders.ts    # YENİ: Provider management
    │   ├── useContext.ts      # YENİ: Context tracking
    │   └── usePricing.ts      # YENİ: Cost tracking
    ├── components/
    │   ├── ProviderSelector.tsx  # YENİ: API seçici
    │   ├── ModelSelector.tsx     # GÜNCELLENECEk: Multi-API desteği
    │   ├── ContextIndicator.tsx  # YENİ: Context göstergesi
    │   └── CostTracker.tsx       # YENİ: Maliyet tracker
    └── routes/
        └── settings/
            ├── providers.tsx  # YENİ: Provider ayarları
            └── context.tsx    # YENİ: Context ayarları
```

---

## 📁 DETAYLI DOSYA DEĞİŞİKLİKLERİ

### 1. DATABASE SCHEMA GÜNCELLEMESİ

**Dosya:** `/home/user/ollama/app/store/schema.sql`

**Eklenecek Tablolar:**

```sql
-- API Providers
CREATE TABLE IF NOT EXISTS providers (
  id TEXT PRIMARY KEY,                    -- UUID
  name TEXT NOT NULL,                     -- OpenAI, Anthropic, etc.
  type TEXT NOT NULL,                     -- openai, anthropic, google, groq, custom
  api_key TEXT,                           -- Encrypted API key
  base_url TEXT,                          -- API base URL
  models TEXT,                            -- Available models (JSON array)
  enabled BOOLEAN DEFAULT 1,
  default_model TEXT,                     -- Default model for this provider
  config TEXT,                            -- Provider-specific config (JSON)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Pricing (Cache)
CREATE TABLE IF NOT EXISTS model_pricing (
  provider_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  input_price_per_1m REAL,               -- Input token fiyatı (per 1M tokens)
  output_price_per_1m REAL,              -- Output token fiyatı
  context_window INTEGER,                -- Max context size
  supports_streaming BOOLEAN DEFAULT 1,
  supports_tools BOOLEAN DEFAULT 0,
  supports_vision BOOLEAN DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (provider_id, model_name),
  FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
);

-- API Usage Tracking
CREATE TABLE IF NOT EXISTS api_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT,
  message_id INTEGER,
  provider_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0.0,             -- Calculated cost
  duration_ms INTEGER,                   -- Response time
  tokens_per_second REAL,                -- Performance metric
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
  FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
  FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
);

-- Context Management
CREATE TABLE IF NOT EXISTS context_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT NOT NULL,
  snapshot_at_message_id INTEGER,        -- Which message triggered snapshot
  summary TEXT,                          -- Summarized context
  original_tokens INTEGER,               -- Original context size
  summary_tokens INTEGER,                -- Summary size
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
  FOREIGN KEY (snapshot_at_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

-- Settings tablosuna eklenecek kolonlar
ALTER TABLE settings ADD COLUMN default_provider_id TEXT;
ALTER TABLE settings ADD COLUMN auto_summarize BOOLEAN DEFAULT 1;
ALTER TABLE settings ADD COLUMN context_warning_threshold REAL DEFAULT 0.8;  -- %80'de uyar
ALTER TABLE settings ADD COLUMN track_costs BOOLEAN DEFAULT 1;

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_api_usage_chat_id ON api_usage(chat_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_context_snapshots_chat_id ON context_snapshots(chat_id);
```

---

### 2. GO BACKEND - PROVIDER INTERFACE

**Dosya:** `/home/user/ollama/api/providers/base.go` (YENİ)

```go
package providers

import (
    "context"
    "io"
)

// Provider interface - Tüm API provider'ların implement edeceği interface
type Provider interface {
    // GetName returns provider name
    GetName() string

    // GetType returns provider type (openai, anthropic, etc.)
    GetType() string

    // ListModels lists available models
    ListModels(ctx context.Context) ([]Model, error)

    // ChatCompletion performs chat completion
    ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)

    // ChatCompletionStream performs streaming chat completion
    ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error)

    // Embeddings generates embeddings
    Embeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)

    // ValidateCredentials validates API key
    ValidateCredentials(ctx context.Context) error

    // GetPricing returns pricing info for a model
    GetPricing(modelName string) (*ModelPricing, error)
}

// Model represents a model from any provider
type Model struct {
    ID              string   `json:"id"`
    Name            string   `json:"name"`
    DisplayName     string   `json:"display_name"`
    ContextWindow   int      `json:"context_window"`
    Capabilities    []string `json:"capabilities"` // chat, completion, vision, tools
    Deprecated      bool     `json:"deprecated"`
}

// ChatRequest standardized request
type ChatRequest struct {
    Model       string         `json:"model"`
    Messages    []Message      `json:"messages"`
    Stream      bool           `json:"stream"`
    Temperature *float64       `json:"temperature,omitempty"`
    MaxTokens   *int           `json:"max_tokens,omitempty"`
    TopP        *float64       `json:"top_p,omitempty"`
    Stop        []string       `json:"stop,omitempty"`
    Tools       []Tool         `json:"tools,omitempty"`
}

// ChatResponse standardized response
type ChatResponse struct {
    ID      string        `json:"id"`
    Model   string        `json:"model"`
    Message Message       `json:"message"`
    Usage   UsageMetrics  `json:"usage"`
    Metrics PerformanceMetrics `json:"metrics"`
}

// UsageMetrics token usage
type UsageMetrics struct {
    InputTokens  int `json:"input_tokens"`
    OutputTokens int `json:"output_tokens"`
    TotalTokens  int `json:"total_tokens"`
}

// PerformanceMetrics performance tracking
type PerformanceMetrics struct {
    DurationMs     int64   `json:"duration_ms"`
    TokensPerSecond float64 `json:"tokens_per_second"`
    TimeToFirstToken int64  `json:"time_to_first_token_ms,omitempty"`
}

// ModelPricing pricing information
type ModelPricing struct {
    InputPricePer1M  float64 `json:"input_price_per_1m"`
    OutputPricePer1M float64 `json:"output_price_per_1m"`
    ContextWindow    int     `json:"context_window"`
}

// Message structure
type Message struct {
    Role       string      `json:"role"`
    Content    string      `json:"content"`
    Thinking   string      `json:"thinking,omitempty"`
    ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
    Images     []string    `json:"images,omitempty"`
}

// Tool definition
type Tool struct {
    Type     string       `json:"type"`
    Function ToolFunction `json:"function"`
}

// ToolFunction definition
type ToolFunction struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall structure
type ToolCall struct {
    ID       string               `json:"id"`
    Type     string               `json:"type"`
    Function ToolCallFunction     `json:"function"`
}

// ToolCallFunction structure
type ToolCallFunction struct {
    Name      string `json:"name"`
    Arguments string `json:"arguments"`
}

// EmbeddingRequest structure
type EmbeddingRequest struct {
    Model string   `json:"model"`
    Input []string `json:"input"`
}

// EmbeddingResponse structure
type EmbeddingResponse struct {
    Embeddings [][]float64  `json:"embeddings"`
    Usage      UsageMetrics `json:"usage"`
}
```

---

### 3. OPENAI PROVIDER IMPLEMENTATION

**Dosya:** `/home/user/ollama/api/providers/openai.go` (YENİ)

```go
package providers

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type OpenAIProvider struct {
    APIKey  string
    BaseURL string
    client  *http.Client
}

func NewOpenAIProvider(apiKey string, baseURL string) *OpenAIProvider {
    if baseURL == "" {
        baseURL = "https://api.openai.com/v1"
    }
    return &OpenAIProvider{
        APIKey:  apiKey,
        BaseURL: baseURL,
        client:  &http.Client{Timeout: 120 * time.Second},
    }
}

func (p *OpenAIProvider) GetName() string {
    return "OpenAI"
}

func (p *OpenAIProvider) GetType() string {
    return "openai"
}

func (p *OpenAIProvider) ListModels(ctx context.Context) ([]Model, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", p.BaseURL+"/models", nil)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Authorization", "Bearer "+p.APIKey)

    resp, err := p.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        Data []struct {
            ID      string `json:"id"`
            Created int64  `json:"created"`
        } `json:"data"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    models := make([]Model, 0)
    for _, m := range result.Data {
        // Filter chat models
        if isOpenAIChatModel(m.ID) {
            models = append(models, Model{
                ID:           m.ID,
                Name:         m.ID,
                DisplayName:  formatOpenAIModelName(m.ID),
                ContextWindow: getOpenAIContextWindow(m.ID),
                Capabilities: getOpenAICapabilities(m.ID),
            })
        }
    }

    return models, nil
}

func (p *OpenAIProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
    startTime := time.Now()

    // Convert to OpenAI format
    openAIReq := convertToOpenAIRequest(req)

    body, err := json.Marshal(openAIReq)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/chat/completions",
        bytes.NewReader(body))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
    httpReq.Header.Set("Content-Type", "application/json")

    resp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("OpenAI API error: %s - %s", resp.Status, string(bodyBytes))
    }

    var openAIResp struct {
        ID      string `json:"id"`
        Model   string `json:"model"`
        Choices []struct {
            Message struct {
                Role      string `json:"role"`
                Content   string `json:"content"`
                ToolCalls []struct {
                    ID       string `json:"id"`
                    Type     string `json:"type"`
                    Function struct {
                        Name      string `json:"name"`
                        Arguments string `json:"arguments"`
                    } `json:"function"`
                } `json:"tool_calls,omitempty"`
            } `json:"message"`
        } `json:"choices"`
        Usage struct {
            PromptTokens     int `json:"prompt_tokens"`
            CompletionTokens int `json:"completion_tokens"`
            TotalTokens      int `json:"total_tokens"`
        } `json:"usage"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
        return nil, err
    }

    duration := time.Since(startTime)
    tokensPerSec := float64(openAIResp.Usage.CompletionTokens) / duration.Seconds()

    // Convert back to standard format
    response := &ChatResponse{
        ID:    openAIResp.ID,
        Model: openAIResp.Model,
        Message: Message{
            Role:    openAIResp.Choices[0].Message.Role,
            Content: openAIResp.Choices[0].Message.Content,
        },
        Usage: UsageMetrics{
            InputTokens:  openAIResp.Usage.PromptTokens,
            OutputTokens: openAIResp.Usage.CompletionTokens,
            TotalTokens:  openAIResp.Usage.TotalTokens,
        },
        Metrics: PerformanceMetrics{
            DurationMs:      duration.Milliseconds(),
            TokensPerSecond: tokensPerSec,
        },
    }

    // Tool calls varsa ekle
    if len(openAIResp.Choices[0].Message.ToolCalls) > 0 {
        response.Message.ToolCalls = make([]ToolCall, len(openAIResp.Choices[0].Message.ToolCalls))
        for i, tc := range openAIResp.Choices[0].Message.ToolCalls {
            response.Message.ToolCalls[i] = ToolCall{
                ID:   tc.ID,
                Type: tc.Type,
                Function: ToolCallFunction{
                    Name:      tc.Function.Name,
                    Arguments: tc.Function.Arguments,
                },
            }
        }
    }

    return response, nil
}

func (p *OpenAIProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
    // Streaming implementation
    openAIReq := convertToOpenAIRequest(req)
    openAIReq["stream"] = true

    body, err := json.Marshal(openAIReq)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/chat/completions",
        bytes.NewReader(body))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
    httpReq.Header.Set("Content-Type", "application/json")

    resp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, err
    }

    if resp.StatusCode != http.StatusOK {
        resp.Body.Close()
        return nil, fmt.Errorf("OpenAI API error: %s", resp.Status)
    }

    return resp.Body, nil
}

func (p *OpenAIProvider) Embeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
    // Embedding implementation
    // ...
    return nil, fmt.Errorf("not implemented")
}

func (p *OpenAIProvider) ValidateCredentials(ctx context.Context) error {
    _, err := p.ListModels(ctx)
    return err
}

func (p *OpenAIProvider) GetPricing(modelName string) (*ModelPricing, error) {
    // Model pricing database
    pricing := map[string]ModelPricing{
        "gpt-4": {
            InputPricePer1M:  30.0,
            OutputPricePer1M: 60.0,
            ContextWindow:    8192,
        },
        "gpt-4-32k": {
            InputPricePer1M:  60.0,
            OutputPricePer1M: 120.0,
            ContextWindow:    32768,
        },
        "gpt-4-turbo": {
            InputPricePer1M:  10.0,
            OutputPricePer1M: 30.0,
            ContextWindow:    128000,
        },
        "gpt-3.5-turbo": {
            InputPricePer1M:  0.5,
            OutputPricePer1M: 1.5,
            ContextWindow:    16385,
        },
        "gpt-4o": {
            InputPricePer1M:  5.0,
            OutputPricePer1M: 15.0,
            ContextWindow:    128000,
        },
        "o1-preview": {
            InputPricePer1M:  15.0,
            OutputPricePer1M: 60.0,
            ContextWindow:    128000,
        },
    }

    if p, ok := pricing[modelName]; ok {
        return &p, nil
    }

    return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}

// Helper functions
func isOpenAIChatModel(modelID string) bool {
    chatModels := []string{"gpt-3.5", "gpt-4", "o1"}
    for _, prefix := range chatModels {
        if strings.HasPrefix(modelID, prefix) {
            return true
        }
    }
    return false
}

func formatOpenAIModelName(modelID string) string {
    // "gpt-4-turbo-2024-04-09" -> "GPT-4 Turbo"
    parts := strings.Split(modelID, "-")
    if len(parts) >= 2 {
        return strings.ToUpper(parts[0]) + "-" + parts[1]
    }
    return modelID
}

func getOpenAIContextWindow(modelID string) int {
    if strings.Contains(modelID, "32k") {
        return 32768
    } else if strings.Contains(modelID, "16k") {
        return 16384
    } else if strings.Contains(modelID, "turbo") || strings.Contains(modelID, "4o") {
        return 128000
    } else if strings.HasPrefix(modelID, "gpt-4") {
        return 8192
    } else if strings.HasPrefix(modelID, "gpt-3.5") {
        return 16385
    }
    return 4096
}

func getOpenAICapabilities(modelID string) []string {
    caps := []string{"chat"}

    if strings.HasPrefix(modelID, "gpt-4") {
        caps = append(caps, "tools")
    }

    if strings.Contains(modelID, "vision") || strings.Contains(modelID, "4o") {
        caps = append(caps, "vision")
    }

    if strings.HasPrefix(modelID, "o1") {
        caps = append(caps, "thinking")
    }

    return caps
}

func convertToOpenAIRequest(req ChatRequest) map[string]interface{} {
    openAIReq := map[string]interface{}{
        "model":    req.Model,
        "messages": req.Messages,
    }

    if req.Temperature != nil {
        openAIReq["temperature"] = *req.Temperature
    }
    if req.MaxTokens != nil {
        openAIReq["max_tokens"] = *req.MaxTokens
    }
    if req.TopP != nil {
        openAIReq["top_p"] = *req.TopP
    }
    if len(req.Stop) > 0 {
        openAIReq["stop"] = req.Stop
    }
    if len(req.Tools) > 0 {
        openAIReq["tools"] = req.Tools
    }

    return openAIReq
}
```

---

### 4. ANTHROPIC PROVIDER IMPLEMENTATION

**Dosya:** `/home/user/ollama/api/providers/anthropic.go` (YENİ)

```go
package providers

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type AnthropicProvider struct {
    APIKey  string
    BaseURL string
    client  *http.Client
}

func NewAnthropicProvider(apiKey string) *AnthropicProvider {
    return &AnthropicProvider{
        APIKey:  apiKey,
        BaseURL: "https://api.anthropic.com/v1",
        client:  &http.Client{Timeout: 120 * time.Second},
    }
}

func (p *AnthropicProvider) GetName() string {
    return "Anthropic"
}

func (p *AnthropicProvider) GetType() string {
    return "anthropic"
}

func (p *AnthropicProvider) ListModels(ctx context.Context) ([]Model, error) {
    // Anthropic doesn't have a models endpoint, return static list
    return []Model{
        {
            ID:            "claude-opus-4",
            Name:          "claude-opus-4",
            DisplayName:   "Claude Opus 4",
            ContextWindow: 200000,
            Capabilities:  []string{"chat", "tools", "vision", "thinking"},
        },
        {
            ID:            "claude-sonnet-4-5",
            Name:          "claude-sonnet-4-5",
            DisplayName:   "Claude Sonnet 4.5",
            ContextWindow: 200000,
            Capabilities:  []string{"chat", "tools", "vision"},
        },
        {
            ID:            "claude-haiku-4-5",
            Name:          "claude-haiku-4-5",
            DisplayName:   "Claude Haiku 4.5",
            ContextWindow: 200000,
            Capabilities:  []string{"chat", "tools", "vision"},
        },
        {
            ID:            "claude-3-5-sonnet-20241022",
            Name:          "claude-3-5-sonnet-20241022",
            DisplayName:   "Claude 3.5 Sonnet",
            ContextWindow: 200000,
            Capabilities:  []string{"chat", "tools", "vision"},
        },
    }, nil
}

func (p *AnthropicProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
    startTime := time.Now()

    // Convert to Anthropic format
    anthropicReq := p.convertToAnthropicRequest(req)

    body, err := json.Marshal(anthropicReq)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/messages",
        bytes.NewReader(body))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("x-api-key", p.APIKey)
    httpReq.Header.Set("anthropic-version", "2023-06-01")
    httpReq.Header.Set("content-type", "application/json")

    // Extended thinking için
    if req.Model == "claude-opus-4" {
        httpReq.Header.Set("anthropic-beta", "extended-thinking-2025-01-08")
    }

    resp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("Anthropic API error: %s - %s", resp.Status, string(bodyBytes))
    }

    var anthropicResp struct {
        ID      string `json:"id"`
        Type    string `json:"type"`
        Role    string `json:"role"`
        Content []struct {
            Type     string `json:"type"`
            Text     string `json:"text,omitempty"`
            Thinking string `json:"thinking,omitempty"`
            ID       string `json:"id,omitempty"`
            Name     string `json:"name,omitempty"`
            Input    json.RawMessage `json:"input,omitempty"`
        } `json:"content"`
        Model   string `json:"model"`
        Usage   struct {
            InputTokens  int `json:"input_tokens"`
            OutputTokens int `json:"output_tokens"`
        } `json:"usage"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
        return nil, err
    }

    duration := time.Since(startTime)
    tokensPerSec := float64(anthropicResp.Usage.OutputTokens) / duration.Seconds()

    // Convert back to standard format
    message := Message{
        Role: "assistant",
    }

    // Combine content blocks
    var textContent string
    var thinkingContent string
    var toolCalls []ToolCall

    for _, block := range anthropicResp.Content {
        switch block.Type {
        case "text":
            textContent += block.Text
        case "thinking":
            thinkingContent = block.Thinking
        case "tool_use":
            toolCalls = append(toolCalls, ToolCall{
                ID:   block.ID,
                Type: "function",
                Function: ToolCallFunction{
                    Name:      block.Name,
                    Arguments: string(block.Input),
                },
            })
        }
    }

    message.Content = textContent
    message.Thinking = thinkingContent
    message.ToolCalls = toolCalls

    response := &ChatResponse{
        ID:      anthropicResp.ID,
        Model:   anthropicResp.Model,
        Message: message,
        Usage: UsageMetrics{
            InputTokens:  anthropicResp.Usage.InputTokens,
            OutputTokens: anthropicResp.Usage.OutputTokens,
            TotalTokens:  anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
        },
        Metrics: PerformanceMetrics{
            DurationMs:      duration.Milliseconds(),
            TokensPerSecond: tokensPerSec,
        },
    }

    return response, nil
}

func (p *AnthropicProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
    // Streaming implementation
    anthropicReq := p.convertToAnthropicRequest(req)
    anthropicReq["stream"] = true

    body, err := json.Marshal(anthropicReq)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/messages",
        bytes.NewReader(body))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("x-api-key", p.APIKey)
    httpReq.Header.Set("anthropic-version", "2023-06-01")
    httpReq.Header.Set("content-type", "application/json")

    if req.Model == "claude-opus-4" {
        httpReq.Header.Set("anthropic-beta", "extended-thinking-2025-01-08")
    }

    resp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, err
    }

    if resp.StatusCode != http.StatusOK {
        resp.Body.Close()
        return nil, fmt.Errorf("Anthropic API error: %s", resp.Status)
    }

    return resp.Body, nil
}

func (p *AnthropicProvider) GetPricing(modelName string) (*ModelPricing, error) {
    pricing := map[string]ModelPricing{
        "claude-opus-4": {
            InputPricePer1M:  15.0,
            OutputPricePer1M: 75.0,
            ContextWindow:    200000,
        },
        "claude-sonnet-4-5": {
            InputPricePer1M:  3.0,
            OutputPricePer1M: 15.0,
            ContextWindow:    200000,
        },
        "claude-haiku-4-5": {
            InputPricePer1M:  0.8,
            OutputPricePer1M: 4.0,
            ContextWindow:    200000,
        },
        "claude-3-5-sonnet-20241022": {
            InputPricePer1M:  3.0,
            OutputPricePer1M: 15.0,
            ContextWindow:    200000,
        },
    }

    if p, ok := pricing[modelName]; ok {
        return &p, nil
    }

    return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}

func (p *AnthropicProvider) convertToAnthropicRequest(req ChatRequest) map[string]interface{} {
    // System prompt'u ayır
    var systemPrompt string
    var messages []map[string]interface{}

    for _, msg := range req.Messages {
        if msg.Role == "system" {
            systemPrompt = msg.Content
            continue
        }

        // Anthropic formatına çevir
        anthropicMsg := map[string]interface{}{
            "role": msg.Role,
        }

        // Content blocks
        var contentBlocks []map[string]interface{}

        if msg.Content != "" {
            contentBlocks = append(contentBlocks, map[string]interface{}{
                "type": "text",
                "text": msg.Content,
            })
        }

        // Images varsa ekle
        for _, img := range msg.Images {
            contentBlocks = append(contentBlocks, map[string]interface{}{
                "type": "image",
                "source": map[string]interface{}{
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       img,
                },
            })
        }

        // Tool calls varsa ekle
        for _, tc := range msg.ToolCalls {
            contentBlocks = append(contentBlocks, map[string]interface{}{
                "type":  "tool_use",
                "id":    tc.ID,
                "name":  tc.Function.Name,
                "input": json.RawMessage(tc.Function.Arguments),
            })
        }

        anthropicMsg["content"] = contentBlocks
        messages = append(messages, anthropicMsg)
    }

    anthropicReq := map[string]interface{}{
        "model":      req.Model,
        "messages":   messages,
        "max_tokens": 4096, // Default
    }

    if systemPrompt != "" {
        anthropicReq["system"] = systemPrompt
    }

    if req.Temperature != nil {
        anthropicReq["temperature"] = *req.Temperature
    }

    if req.MaxTokens != nil {
        anthropicReq["max_tokens"] = *req.MaxTokens
    }

    if req.TopP != nil {
        anthropicReq["top_p"] = *req.TopP
    }

    if len(req.Stop) > 0 {
        anthropicReq["stop_sequences"] = req.Stop
    }

    // Tools
    if len(req.Tools) > 0 {
        anthropicTools := make([]map[string]interface{}, len(req.Tools))
        for i, tool := range req.Tools {
            anthropicTools[i] = map[string]interface{}{
                "name":         tool.Function.Name,
                "description":  tool.Function.Description,
                "input_schema": tool.Function.Parameters,
            }
        }
        anthropicReq["tools"] = anthropicTools
    }

    return anthropicReq
}

// Diğer gerekli methodlar...
func (p *AnthropicProvider) Embeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
    return nil, fmt.Errorf("Anthropic doesn't support embeddings")
}

func (p *AnthropicProvider) ValidateCredentials(ctx context.Context) error {
    // Simple validation - try to create a minimal request
    req := ChatRequest{
        Model: "claude-sonnet-4-5",
        Messages: []Message{
            {Role: "user", Content: "Hi"},
        },
    }
    req.MaxTokens = new(int)
    *req.MaxTokens = 1

    _, err := p.ChatCompletion(ctx, req)
    return err
}
```

---

### 5. PROVIDER REGISTRY

**Dosya:** `/home/user/ollama/api/providers/registry.go` (YENİ)

```go
package providers

import (
    "context"
    "fmt"
    "sync"
)

// Registry manages all providers
type Registry struct {
    providers map[string]Provider
    mu        sync.RWMutex
}

var globalRegistry = &Registry{
    providers: make(map[string]Provider),
}

// GetRegistry returns the global provider registry
func GetRegistry() *Registry {
    return globalRegistry
}

// Register registers a provider
func (r *Registry) Register(id string, provider Provider) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.providers[id] = provider
}

// Get returns a provider by ID
func (r *Registry) Get(id string) (Provider, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    provider, ok := r.providers[id]
    if !ok {
        return nil, fmt.Errorf("provider not found: %s", id)
    }

    return provider, nil
}

// List returns all registered providers
func (r *Registry) List() []Provider {
    r.mu.RLock()
    defer r.mu.RUnlock()

    providers := make([]Provider, 0, len(r.providers))
    for _, p := range r.providers {
        providers = append(providers, p)
    }

    return providers
}

// Remove removes a provider
func (r *Registry) Remove(id string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.providers, id)
}

// CreateProvider creates a new provider from config
func CreateProvider(providerType, apiKey, baseURL string) (Provider, error) {
    switch providerType {
    case "openai":
        return NewOpenAIProvider(apiKey, baseURL), nil
    case "anthropic":
        return NewAnthropicProvider(apiKey), nil
    case "google":
        return NewGoogleProvider(apiKey), nil
    case "groq":
        return NewGroqProvider(apiKey), nil
    case "custom":
        if baseURL == "" {
            return nil, fmt.Errorf("base URL required for custom provider")
        }
        return NewCustomProvider(apiKey, baseURL), nil
    case "ollama":
        return NewOllamaProvider(baseURL), nil
    default:
        return nil, fmt.Errorf("unknown provider type: %s", providerType)
    }
}
```

---

### 6. CONTEXT MANAGER

**Dosya:** `/home/user/ollama/api/context/manager.go` (YENİ)

```go
package context

import (
    "context"
    "fmt"
    "github.com/ollama/ollama/api/providers"
)

// ContextManager manages conversation context
type ContextManager struct {
    maxContextTokens    int
    warningThreshold    float64  // 0.0 - 1.0
    autoSummarize       bool
    summarizationModel  string
    summarizationPrompt string
}

// NewContextManager creates a new context manager
func NewContextManager(maxTokens int, warningThreshold float64, autoSummarize bool) *ContextManager {
    return &ContextManager{
        maxContextTokens: maxTokens,
        warningThreshold: warningThreshold,
        autoSummarize:    autoSummarize,
        summarizationModel: "claude-haiku-4-5", // Fast & cheap
        summarizationPrompt: `Summarize the following conversation concisely, preserving key information and context:

<conversation>
%s
</conversation>

Provide a clear, informative summary that can be used as context for continuing the conversation.`,
    }
}

// CheckContext checks if context is within limits
func (cm *ContextManager) CheckContext(messages []providers.Message, currentTokens int) (*ContextStatus, error) {
    status := &ContextStatus{
        CurrentTokens:   currentTokens,
        MaxTokens:       cm.maxContextTokens,
        UsagePercentage: float64(currentTokens) / float64(cm.maxContextTokens),
        NeedsAction:     false,
    }

    if status.UsagePercentage >= cm.warningThreshold {
        status.NeedsAction = true

        if status.UsagePercentage >= 0.95 {
            status.Action = "truncate" // Immediate truncation needed
        } else {
            status.Action = "warn" // Warning only
        }
    }

    return status, nil
}

// SummarizeMessages summarizes old messages to free up context
func (cm *ContextManager) SummarizeMessages(ctx context.Context, messages []providers.Message, provider providers.Provider) (string, error) {
    if len(messages) == 0 {
        return "", fmt.Errorf("no messages to summarize")
    }

    // Convert messages to text
    var conversationText string
    for _, msg := range messages {
        conversationText += fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content)
    }

    // Create summarization request
    prompt := fmt.Sprintf(cm.summarizationPrompt, conversationText)

    req := providers.ChatRequest{
        Model: cm.summarizationModel,
        Messages: []providers.Message{
            {Role: "user", Content: prompt},
        },
    }

    maxTokens := 1000
    req.MaxTokens = &maxTokens

    resp, err := provider.ChatCompletion(ctx, req)
    if err != nil {
        return "", fmt.Errorf("summarization failed: %w", err)
    }

    return resp.Message.Content, nil
}

// TruncateMessages truncates old messages (simple sliding window)
func (cm *ContextManager) TruncateMessages(messages []providers.Message, targetTokens int) []providers.Message {
    if len(messages) <= 2 {
        return messages // Keep at least system + first user message
    }

    // Keep system message if exists
    startIdx := 0
    if messages[0].Role == "system" {
        startIdx = 1
    }

    // Calculate how many messages to keep
    // Simple approach: remove oldest messages
    estimatedTokensPerMessage := 200 // Rough estimate
    messagesToKeep := targetTokens / estimatedTokensPerMessage

    if messagesToKeep >= len(messages) {
        return messages
    }

    // Keep system + recent messages
    if startIdx == 1 {
        keepFrom := len(messages) - messagesToKeep + 1
        if keepFrom < 1 {
            keepFrom = 1
        }
        return append([]providers.Message{messages[0]}, messages[keepFrom:]...)
    }

    keepFrom := len(messages) - messagesToKeep
    if keepFrom < 0 {
        keepFrom = 0
    }
    return messages[keepFrom:]
}

// ContextStatus represents current context status
type ContextStatus struct {
    CurrentTokens   int     `json:"current_tokens"`
    MaxTokens       int     `json:"max_tokens"`
    UsagePercentage float64 `json:"usage_percentage"`
    NeedsAction     bool    `json:"needs_action"`
    Action          string  `json:"action"` // "warn", "truncate", "summarize"
}
```

---

### 7. PRICING CALCULATOR

**Dosya:** `/home/user/ollama/api/pricing/pricing.go` (YENİ)

```go
package pricing

import (
    "fmt"
    "github.com/ollama/ollama/api/providers"
)

// Calculator calculates API costs
type Calculator struct {
    provider providers.Provider
}

// NewCalculator creates a new pricing calculator
func NewCalculator(provider providers.Provider) *Calculator {
    return &Calculator{provider: provider}
}

// CalculateCost calculates cost for given usage
func (c *Calculator) CalculateCost(modelName string, inputTokens, outputTokens int) (*Cost, error) {
    pricing, err := c.provider.GetPricing(modelName)
    if err != nil {
        return nil, err
    }

    inputCost := float64(inputTokens) / 1000000.0 * pricing.InputPricePer1M
    outputCost := float64(outputTokens) / 1000000.0 * pricing.OutputPricePer1M
    totalCost := inputCost + outputCost

    return &Cost{
        InputCostUSD:  inputCost,
        OutputCostUSD: outputCost,
        TotalCostUSD:  totalCost,
        InputTokens:   inputTokens,
        OutputTokens:  outputTokens,
        TotalTokens:   inputTokens + outputTokens,
    }, nil
}

// Cost represents calculated cost
type Cost struct {
    InputCostUSD  float64 `json:"input_cost_usd"`
    OutputCostUSD float64 `json:"output_cost_usd"`
    TotalCostUSD  float64 `json:"total_cost_usd"`
    InputTokens   int     `json:"input_tokens"`
    OutputTokens  int     `json:"output_tokens"`
    TotalTokens   int     `json:"total_tokens"`
}

// FormatCost formats cost for display
func FormatCost(cost *Cost) string {
    if cost.TotalCostUSD < 0.01 {
        return fmt.Sprintf("$%.4f", cost.TotalCostUSD)
    }
    return fmt.Sprintf("$%.2f", cost.TotalCostUSD)
}
```

---

### 8. SERVER ROUTE UPDATES

**Dosya:** `/home/user/ollama/server/routes.go` (GÜNCELLENECEK)

```go
// Yeni endpoint'ler eklenecek

func (s *Server) RegisterProviderRoutes(r *gin.Engine) {
    api := r.Group("/api")

    // Provider management
    api.GET("/providers", s.ListProvidersHandler)
    api.POST("/providers", s.CreateProviderHandler)
    api.GET("/providers/:id", s.GetProviderHandler)
    api.PUT("/providers/:id", s.UpdateProviderHandler)
    api.DELETE("/providers/:id", s.DeleteProviderHandler)
    api.POST("/providers/:id/validate", s.ValidateProviderHandler)

    // Provider models
    api.GET("/providers/:id/models", s.ListProviderModelsHandler)

    // Unified chat endpoint (supports all providers)
    api.POST("/chat/unified", s.UnifiedChatHandler)

    // Context management
    api.GET("/chat/:chatId/context", s.GetContextStatusHandler)
    api.POST("/chat/:chatId/context/summarize", s.SummarizeContextHandler)

    // Usage & pricing
    api.GET("/usage", s.GetUsageStatsHandler)
    api.GET("/usage/:chatId", s.GetChatUsageHandler)
}

// UnifiedChatHandler handles chat from any provider
func (s *Server) UnifiedChatHandler(c *gin.Context) {
    var req struct {
        ProviderID string `json:"provider_id"`
        providers.ChatRequest
    }

    if err := c.BindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // Get provider from registry
    provider, err := providers.GetRegistry().Get(req.ProviderID)
    if err != nil {
        c.JSON(404, gin.H{"error": err.Error()})
        return
    }

    // Track start time
    startTime := time.Now()

    // Handle streaming vs non-streaming
    if req.Stream {
        s.handleStreamingChat(c, provider, req.ChatRequest, startTime)
    } else {
        s.handleNonStreamingChat(c, provider, req.ChatRequest, startTime)
    }
}

func (s *Server) handleNonStreamingChat(c *gin.Context, provider providers.Provider, req providers.ChatRequest, startTime time.Time) {
    resp, err := provider.ChatCompletion(c.Request.Context(), req)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    // Calculate cost
    calc := pricing.NewCalculator(provider)
    cost, _ := calc.CalculateCost(req.Model, resp.Usage.InputTokens, resp.Usage.OutputTokens)

    // Save to database
    s.saveAPIUsage(c, req, resp, cost)

    c.JSON(200, gin.H{
        "message": resp.Message,
        "usage":   resp.Usage,
        "metrics": resp.Metrics,
        "cost":    cost,
    })
}

func (s *Server) handleStreamingChat(c *gin.Context, provider providers.Provider, req providers.ChatRequest, startTime time.Time) {
    stream, err := provider.ChatCompletionStream(c.Request.Context(), req)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    defer stream.Close()

    // Set headers for SSE
    c.Header("Content-Type", "text/event-stream")
    c.Header("Cache-Control", "no-cache")
    c.Header("Connection", "keep-alive")

    // Stream to client
    scanner := bufio.NewScanner(stream)
    for scanner.Scan() {
        c.Writer.Write(scanner.Bytes())
        c.Writer.Write([]byte("\n"))
        c.Writer.Flush()
    }
}

// ListProvidersHandler lists all providers
func (s *Server) ListProvidersHandler(c *gin.Context) {
    // Load from database
    providers, err := s.store.ListProviders()
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(200, providers)
}

// CreateProviderHandler creates a new provider
func (s *Server) CreateProviderHandler(c *gin.Context) {
    var req struct {
        Name     string `json:"name"`
        Type     string `json:"type"`
        APIKey   string `json:"api_key"`
        BaseURL  string `json:"base_url,omitempty"`
        Config   map[string]interface{} `json:"config,omitempty"`
    }

    if err := c.BindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // Create provider instance
    provider, err := providers.CreateProvider(req.Type, req.APIKey, req.BaseURL)
    if err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // Validate credentials
    if err := provider.ValidateCredentials(c.Request.Context()); err != nil {
        c.JSON(401, gin.H{"error": "Invalid credentials: " + err.Error()})
        return
    }

    // Save to database
    providerID := uuid.New().String()
    if err := s.store.CreateProvider(providerID, req.Name, req.Type, req.APIKey, req.BaseURL, req.Config); err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    // Register in runtime registry
    providers.GetRegistry().Register(providerID, provider)

    c.JSON(201, gin.H{"id": providerID, "name": req.Name})
}

// Diğer handler'lar...
```

---

### 9. FRONTEND - PROVIDER HOOKS

**Dosya:** `/home/user/ollama/app/ui/app/src/hooks/useProviders.ts` (YENİ)

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api';

export interface Provider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'google' | 'groq' | 'custom' | 'ollama';
  enabled: boolean;
  models?: Model[];
  default_model?: string;
}

export interface Model {
  id: string;
  name: string;
  display_name: string;
  context_window: number;
  capabilities: string[];
  deprecated?: boolean;
}

// List providers
export function useProviders() {
  return useQuery({
    queryKey: ['providers'],
    queryFn: async () => {
      const response = await fetch('/api/providers');
      if (!response.ok) throw new Error('Failed to fetch providers');
      return response.json() as Promise<Provider[]>;
    },
  });
}

// Get single provider
export function useProvider(id: string | undefined) {
  return useQuery({
    queryKey: ['providers', id],
    queryFn: async () => {
      if (!id) return null;
      const response = await fetch(`/api/providers/${id}`);
      if (!response.ok) throw new Error('Failed to fetch provider');
      return response.json() as Promise<Provider>;
    },
    enabled: !!id,
  });
}

// List provider models
export function useProviderModels(providerId: string | undefined) {
  return useQuery({
    queryKey: ['providers', providerId, 'models'],
    queryFn: async () => {
      if (!providerId) return [];
      const response = await fetch(`/api/providers/${providerId}/models`);
      if (!response.ok) throw new Error('Failed to fetch models');
      return response.json() as Promise<Model[]>;
    },
    enabled: !!providerId,
  });
}

// Create provider
export function useCreateProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: {
      name: string;
      type: string;
      api_key: string;
      base_url?: string;
      config?: Record<string, any>;
    }) => {
      const response = await fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to create provider');
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
    },
  });
}

// Update provider
export function useUpdateProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ id, ...data }: { id: string; [key: string]: any }) => {
      const response = await fetch(`/api/providers/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to update provider');
      return response.json();
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
      queryClient.invalidateQueries({ queryKey: ['providers', variables.id] });
    },
  });
}

// Delete provider
export function useDeleteProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`/api/providers/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete provider');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
    },
  });
}

// Validate provider
export function useValidateProvider() {
  return useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`/api/providers/${id}/validate`, {
        method: 'POST',
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Validation failed');
      }
      return response.json();
    },
  });
}
```

---

### 10. FRONTEND - CONTEXT TRACKING HOOK

**Dosya:** `/home/user/ollama/app/ui/app/src/hooks/useContext.ts` (YENİ)

```typescript
import { useQuery } from '@tanstack/react-query';

export interface ContextStatus {
  current_tokens: number;
  max_tokens: number;
  usage_percentage: number;
  needs_action: boolean;
  action?: 'warn' | 'truncate' | 'summarize';
}

export function useContextStatus(chatId: string | undefined) {
  return useQuery({
    queryKey: ['context', chatId],
    queryFn: async () => {
      if (!chatId) return null;
      const response = await fetch(`/api/chat/${chatId}/context`);
      if (!response.ok) throw new Error('Failed to fetch context status');
      return response.json() as Promise<ContextStatus>;
    },
    enabled: !!chatId,
    refetchInterval: 5000, // Refresh every 5 seconds
  });
}
```

---

### 11. FRONTEND - PRICING HOOK

**Dosya:** `/home/user/ollama/app/ui/app/src/hooks/usePricing.ts` (YENİ)

```typescript
import { useQuery } from '@tanstack/react-query';

export interface UsageStats {
  total_cost_usd: number;
  total_tokens: number;
  total_requests: number;
  by_model: {
    model_name: string;
    cost_usd: number;
    tokens: number;
    requests: number;
  }[];
  by_provider: {
    provider_id: string;
    provider_name: string;
    cost_usd: number;
    tokens: number;
    requests: number;
  }[];
}

export function useUsageStats(timeRange?: '24h' | '7d' | '30d' | 'all') {
  return useQuery({
    queryKey: ['usage', timeRange || 'all'],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (timeRange) params.set('range', timeRange);

      const response = await fetch(`/api/usage?${params}`);
      if (!response.ok) throw new Error('Failed to fetch usage stats');
      return response.json() as Promise<UsageStats>;
    },
  });
}

export function useChatUsage(chatId: string | undefined) {
  return useQuery({
    queryKey: ['usage', 'chat', chatId],
    queryFn: async () => {
      if (!chatId) return null;
      const response = await fetch(`/api/usage/${chatId}`);
      if (!response.ok) throw new Error('Failed to fetch chat usage');
      return response.json();
    },
    enabled: !!chatId,
  });
}
```

---

### 12. FRONTEND - PROVIDER SELECTOR COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/components/ProviderSelector.tsx` (YENİ)

```typescript
import { Fragment } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid';
import { useProviders, useProviderModels } from '../hooks/useProviders';

interface ProviderSelectorProps {
  selectedProviderId?: string;
  selectedModel?: string;
  onProviderChange: (providerId: string) => void;
  onModelChange: (modelName: string) => void;
}

export function ProviderSelector({
  selectedProviderId,
  selectedModel,
  onProviderChange,
  onModelChange,
}: ProviderSelectorProps) {
  const { data: providers = [], isLoading: providersLoading } = useProviders();
  const { data: models = [], isLoading: modelsLoading } = useProviderModels(selectedProviderId);

  const selectedProvider = providers.find(p => p.id === selectedProviderId);
  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="flex gap-2">
      {/* Provider Selector */}
      <Listbox value={selectedProviderId} onChange={onProviderChange}>
        <div className="relative w-48">
          <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-800 py-2 pl-3 pr-10 text-left shadow-md focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 sm:text-sm">
            <span className="block truncate">
              {selectedProvider?.name || 'Select Provider'}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon
                className="h-5 w-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </Listbox.Button>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
              {providers.map((provider) => (
                <Listbox.Option
                  key={provider.id}
                  value={provider.id}
                  disabled={!provider.enabled}
                  className={({ active }) =>
                    `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                      active ? 'bg-indigo-100 dark:bg-indigo-900 text-indigo-900 dark:text-indigo-100' : 'text-gray-900 dark:text-gray-100'
                    } ${!provider.enabled ? 'opacity-50 cursor-not-allowed' : ''}`
                  }
                >
                  {({ selected }) => (
                    <>
                      <span className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}>
                        {provider.name}
                      </span>
                      {selected && (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-indigo-600 dark:text-indigo-400">
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
                        </span>
                      )}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>

      {/* Model Selector */}
      <Listbox value={selectedModel} onChange={onModelChange} disabled={!selectedProviderId || modelsLoading}>
        <div className="relative flex-1">
          <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-800 py-2 pl-3 pr-10 text-left shadow-md focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 sm:text-sm disabled:opacity-50 disabled:cursor-not-allowed">
            <span className="block truncate">
              {selectedModelData?.display_name || 'Select Model'}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon
                className="h-5 w-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </Listbox.Button>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
              {models.map((model) => (
                <Listbox.Option
                  key={model.id}
                  value={model.id}
                  disabled={model.deprecated}
                  className={({ active }) =>
                    `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                      active ? 'bg-indigo-100 dark:bg-indigo-900 text-indigo-900 dark:text-indigo-100' : 'text-gray-900 dark:text-gray-100'
                    } ${model.deprecated ? 'opacity-50 cursor-not-allowed' : ''}`
                  }
                >
                  {({ selected }) => (
                    <>
                      <div>
                        <span className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}>
                          {model.display_name}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {model.context_window.toLocaleString()} tokens
                          {model.capabilities.length > 0 && ` • ${model.capabilities.join(', ')}`}
                        </span>
                      </div>
                      {selected && (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-indigo-600 dark:text-indigo-400">
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
                        </span>
                      )}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>
    </div>
  );
}
```

---

### 13. FRONTEND - CONTEXT INDICATOR COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/components/ContextIndicator.tsx` (YENİ)

```typescript
import { useContextStatus } from '../hooks/useContext';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface ContextIndicatorProps {
  chatId: string;
}

export function ContextIndicator({ chatId }: ContextIndicatorProps) {
  const { data: contextStatus } = useContextStatus(chatId);

  if (!contextStatus) return null;

  const percentage = Math.round(contextStatus.usage_percentage * 100);
  const isWarning = percentage >= 80;
  const isCritical = percentage >= 95;

  return (
    <div className="flex items-center gap-2 text-sm">
      {/* Progress Bar */}
      <div className="relative w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            isCritical
              ? 'bg-red-500'
              : isWarning
              ? 'bg-yellow-500'
              : 'bg-green-500'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Text */}
      <span className={`font-mono ${isWarning ? 'text-yellow-600 dark:text-yellow-400' : 'text-gray-600 dark:text-gray-400'}`}>
        {contextStatus.current_tokens.toLocaleString()} / {contextStatus.max_tokens.toLocaleString()}
      </span>

      {/* Warning Icon */}
      {isWarning && (
        <ExclamationTriangleIcon
          className={`h-5 w-5 ${isCritical ? 'text-red-500' : 'text-yellow-500'}`}
          title={isCritical ? 'Context limit reached!' : 'Context usage high'}
        />
      )}
    </div>
  );
}
```

---

### 14. FRONTEND - COST TRACKER COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/components/CostTracker.tsx` (YENİ)

```typescript
import { useUsageStats } from '../hooks/usePricing';
import { BanknotesIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';

export function CostTracker() {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | 'all'>('24h');
  const { data: stats } = useUsageStats(timeRange);

  if (!stats) return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BanknotesIcon className="h-6 w-6 text-green-600" />
          <h3 className="text-lg font-semibold">API Usage & Costs</h3>
        </div>

        {/* Time Range Selector */}
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value as any)}
          className="rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm"
        >
          <option value="24h">Last 24 hours</option>
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="all">All time</option>
        </select>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Cost</div>
          <div className="text-2xl font-bold text-green-600">
            ${stats.total_cost_usd.toFixed(4)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Tokens</div>
          <div className="text-2xl font-bold">
            {stats.total_tokens.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Requests</div>
          <div className="text-2xl font-bold">
            {stats.total_requests.toLocaleString()}
          </div>
        </div>
      </div>

      {/* By Provider */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold mb-2">By Provider</h4>
        <div className="space-y-2">
          {stats.by_provider.map((provider) => (
            <div key={provider.provider_id} className="flex justify-between items-center text-sm">
              <span>{provider.provider_name}</span>
              <div className="flex gap-4">
                <span className="text-gray-500">{provider.tokens.toLocaleString()} tokens</span>
                <span className="font-semibold text-green-600">${provider.cost_usd.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* By Model */}
      <div>
        <h4 className="text-sm font-semibold mb-2">By Model</h4>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {stats.by_model.map((model) => (
            <div key={model.model_name} className="flex justify-between items-center text-sm">
              <span className="truncate flex-1">{model.model_name}</span>
              <div className="flex gap-4">
                <span className="text-gray-500">{model.requests} reqs</span>
                <span className="font-semibold text-green-600 w-20 text-right">${model.cost_usd.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

## 🧪 TESTLER

### Unit Tests

**Dosya:** `/home/user/ollama/api/providers/openai_test.go` (YENİ)

```go
package providers

import (
    "context"
    "testing"
)

func TestOpenAIProvider_ListModels(t *testing.T) {
    // Mock API key için test
    provider := NewOpenAIProvider("test-key", "")

    models, err := provider.ListModels(context.Background())
    if err != nil {
        t.Fatalf("ListModels failed: %v", err)
    }

    if len(models) == 0 {
        t.Error("Expected models, got none")
    }
}

func TestOpenAIProvider_GetPricing(t *testing.T) {
    provider := NewOpenAIProvider("test-key", "")

    pricing, err := provider.GetPricing("gpt-4")
    if err != nil {
        t.Fatalf("GetPricing failed: %v", err)
    }

    if pricing.InputPricePer1M <= 0 {
        t.Error("Invalid pricing")
    }
}
```

---

## 📊 PERFORMANS KRİTERLERİ

### Backend Performance
- **Provider Creation:** < 100ms
- **Model List Fetch:** < 500ms (with caching)
- **Chat Request:** < 50ms overhead (provider latency excluded)
- **Context Check:** < 10ms
- **Cost Calculation:** < 1ms

### Frontend Performance
- **Provider Switch:** < 100ms
- **Model List Load:** < 300ms
- **Context Update:** Real-time (< 100ms)
- **Cost Display:** Instant (cached)

### Database
- **Provider Query:** < 10ms (indexed)
- **Usage Stats:** < 50ms (indexed by chat_id, timestamp)
- **Context Snapshot:** < 20ms

### Caching Strategy
1. **Model Lists:** Cache 1 hour (invalidate on provider update)
2. **Pricing Data:** Cache 24 hours
3. **Context Status:** Cache 5 seconds
4. **Usage Stats:** Cache 30 seconds

---

## 🔐 GÜVENLİK

### API Key Security
- **Encryption:** API keys encrypted at rest (AES-256)
- **Storage:** Secure keychain on macOS/Windows
- **Transmission:** HTTPS only
- **Validation:** Immediate validation on save

### Data Privacy
- **Local Storage:** All chat data stays local
- **No Telemetry:** No usage data sent to external servers
- **Provider Isolation:** Each provider isolated

---

## 🚀 DEĞİŞİKLİK ADIMLARI

### 1. Database Migration
```bash
# Migration script oluştur
cd /home/user/ollama/app/store
# schema.sql'i güncelle
# Migration çalıştır (uygulama ilk açılışta otomatik)
```

### 2. Go Backend Implementation
```bash
# Yeni paketleri oluştur
mkdir -p api/providers api/pricing api/context

# Provider interface ve implementations yaz
# openai.go, anthropic.go, google.go, groq.go, custom.go

# Registry oluştur
# Server routes ekle
```

### 3. Frontend Implementation
```bash
cd app/ui/app

# Hooks oluştur
# useProviders.ts, useContext.ts, usePricing.ts

# Components oluştur
# ProviderSelector.tsx, ContextIndicator.tsx, CostTracker.tsx

# Settings sayfasına provider yönetimi ekle
```

### 4. Testing
```bash
# Backend tests
go test ./api/providers/... -v

# Frontend tests
npm run test

# Integration tests
npm run test:e2e
```

### 5. Documentation
```bash
# API docs güncelle
# User guide yaz
# Migration guide oluştur
```

---

## ✅ BAŞARI KRİTERLERİ

1. ✅ Kullanıcı OpenAI, Anthropic, Google, Groq API'lerini ekleyebiliyor
2. ✅ Her provider için model listesi görülebiliyor ve seçilebiliyor
3. ✅ Context doluluk oranı gerçek zamanlı görüntüleniyor
4. ✅ Otomatik context özetleme çalışıyor
5. ✅ API maliyetleri doğru hesaplanıyor ve gösteriliyor
6. ✅ Token/saniye metrikleri tracking ediliyor
7. ✅ Tüm testler geçiyor
8. ✅ Performans kriterleri karşılanıyor

---

## 📝 NOTLAR

- Bu phase tüm sonraki phase'lerin temeli
- Provider sistemi genişletilebilir (yeni provider eklemek kolay)
- Context yönetimi RAG sistemi için de kullanılacak
- Pricing sistemi maliyet raporlama için temel

---

## 🔄 SONRAKİ PHASE

**Phase 2:** Kurallar ve Todo Sistemi (.leah klasörü yapısı)