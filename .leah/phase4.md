# PHASE 4: ADVANCED CHAT FEATURES (Multi-Model, Streaming, Context)

## 📋 HEDEFLER
1. ✅ Multi-model chat (2-3 model aynı anda)
2. ✅ Model karşılaştırma (yan yana)
3. ✅ Streaming improvements
4. ✅ Context auto-summarization
5. ✅ Message regeneration
6. ✅ Branch conversations

## 🏗️ MİMARİ

### Multi-Model Chat Yapısı
```typescript
interface MultiModelChat {
  models: {
    provider_id: string;
    model_name: string;
    role: 'primary' | 'secondary' | 'comparison';
  }[];
  sync_mode: 'parallel' | 'sequential';
  comparison_mode: boolean;
}
```

### Database Schema
```sql
-- Multi-model message tracking
CREATE TABLE multi_model_responses (
  id INTEGER PRIMARY KEY,
  message_id INTEGER,
  provider_id TEXT,
  model_name TEXT,
  response_content TEXT,
  response_time_ms INTEGER,
  cost_usd REAL,
  FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Conversation branches
CREATE TABLE conversation_branches (
  id INTEGER PRIMARY KEY,
  chat_id TEXT,
  parent_message_id INTEGER,
  branch_name TEXT,
  created_at TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(id),
  FOREIGN KEY (parent_message_id) REFERENCES messages(id)
);
```

## 📁 DOSYALAR

### 1. Multi-Model Manager
**Dosya:** `/home/user/ollama/server/multimodel.go` (YENİ)

```go
type MultiModelManager struct {
    providers map[string]providers.Provider
}

func (m *MultiModelManager) ChatMultiple(ctx context.Context, req MultiModelRequest) ([]*providers.ChatResponse, error) {
    results := make([]*providers.ChatResponse, len(req.Models))
    errs := make([]error, len(req.Models))

    if req.SyncMode == "parallel" {
        var wg sync.WaitGroup
        for i, model := range req.Models {
            wg.Add(1)
            go func(idx int, m ModelConfig) {
                defer wg.Done()
                provider := m.providers[m.ProviderID]
                resp, err := provider.ChatCompletion(ctx, req.ChatRequest)
                results[idx] = resp
                errs[idx] = err
            }(i, model)
        }
        wg.Wait()
    } else {
        // Sequential
        for i, model := range req.Models {
            provider := m.providers[model.ProviderID]
            resp, err := provider.ChatCompletion(ctx, req.ChatRequest)
            results[i] = resp
            errs[i] = err
        }
    }

    return results, combineErrors(errs)
}
```

### 2. Comparison Component
**Dosya:** `/home/user/ollama/app/ui/app/src/components/ComparisonView.tsx` (YENİ)

```typescript
export function ComparisonView({ responses }: { responses: ModelResponse[] }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {responses.map((resp, idx) => (
        <div key={idx} className="glass rounded-lg p-4">
          <div className="flex justify-between mb-2">
            <span className="font-semibold">{resp.model_name}</span>
            <span className="text-sm text-gray-500">{resp.duration_ms}ms</span>
          </div>
          <div className="prose dark:prose-invert">
            <ReactMarkdown>{resp.content}</ReactMarkdown>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Cost: ${resp.cost.toFixed(4)} | Tokens: {resp.total_tokens}
          </div>
        </div>
      ))}
    </div>
  );
}
```

## 📊 PERFORMANS
- **Parallel Request:** 2-3 models simultaneously
- **Streaming:** 60fps smooth updates
- **Context Summarization:** < 2s
- **Branch Creation:** < 100ms

## ✅ BAŞARI KRİTERLERİ
1. ✅ 2-3 model paralel çalışıyor
2. ✅ Karşılaştırma görünümü çalışıyor
3. ✅ Context otomatik özetleniyor
4. ✅ Branch conversations çalışıyor

**SONRAKİ:** Phase 5 - Prompt Templates
