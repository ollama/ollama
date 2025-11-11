# PHASE 8: MODEL MANAGEMENT (Benchmark, Comparison, Fine-tuning)

## 📋 HEDEFLER
1. ✅ Model listesi UI
2. ✅ Model indirme/silme
3. ✅ Benchmark testleri (speed, quality, cost)
4. ✅ Model karşılaştırma tablosu
5. ✅ Fine-tuning job management (future)

## 🏗️ MİMARİ

### Benchmark Schema
```sql
CREATE TABLE model_benchmarks (
  id INTEGER PRIMARY KEY,
  provider_id TEXT,
  model_name TEXT,
  benchmark_type TEXT,  -- speed, quality, reasoning
  test_prompt TEXT,
  response_text TEXT,
  tokens_per_second REAL,
  latency_ms INTEGER,
  quality_score REAL,
  cost_usd REAL,
  timestamp TIMESTAMP
);
```

## 📁 DOSYALAR

### 1. Benchmark Runner
**Dosya:** `/home/user/ollama/benchmark/runner.go` (YENİ)

```go
type BenchmarkRunner struct {
    providers map[string]providers.Provider
}

type BenchmarkTest struct {
    Name        string
    Type        string
    Prompt      string
    ExpectedTokens int
}

var StandardBenchmarks = []BenchmarkTest{
    {
        Name:   "Speed Test (Simple)",
        Type:   "speed",
        Prompt: "Count from 1 to 100.",
        ExpectedTokens: 200,
    },
    {
        Name:   "Reasoning Test",
        Type:   "reasoning",
        Prompt: "Solve: If a train leaves Station A at 60mph and another leaves Station B at 80mph, 300 miles apart, heading towards each other, when do they meet?",
        ExpectedTokens: 300,
    },
    {
        Name:   "Code Generation",
        Type:   "quality",
        Prompt: "Write a function to find the longest palindrome in a string.",
        ExpectedTokens: 500,
    },
}

func (br *BenchmarkRunner) RunBenchmark(providerID, modelName string, test BenchmarkTest) (*BenchmarkResult, error) {
    provider := br.providers[providerID]

    req := providers.ChatRequest{
        Model: modelName,
        Messages: []providers.Message{
            {Role: "user", Content: test.Prompt},
        },
    }

    startTime := time.Now()
    resp, err := provider.ChatCompletion(context.Background(), req)
    duration := time.Since(startTime)

    if err != nil {
        return nil, err
    }

    return &BenchmarkResult{
        ProviderID:      providerID,
        ModelName:       modelName,
        BenchmarkType:   test.Type,
        TokensPerSecond: resp.Metrics.TokensPerSecond,
        LatencyMs:       duration.Milliseconds(),
        OutputTokens:    resp.Usage.OutputTokens,
        CostUSD:         calculateCost(resp.Usage),
    }, nil
}
```

### 2. Model Manager UI
**Dosya:** `/home/user/ollama/app/ui/app/src/components/ModelManager.tsx` (YENİ)

```typescript
export function ModelManager() {
  const { data: providers } = useProviders();
  const { data: benchmarks } = useBenchmarks();
  const runBenchmark = useRunBenchmark();

  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const handleBenchmark = async () => {
    for (const modelId of selectedModels) {
      await runBenchmark.mutateAsync({ modelId });
    }
  };

  return (
    <div className="space-y-6">
      {/* Model List */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Available Models</h2>
        <table className="w-full">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2">
                <input type="checkbox" />
              </th>
              <th className="text-left">Model</th>
              <th className="text-left">Provider</th>
              <th className="text-right">Context</th>
              <th className="text-right">Input Price</th>
              <th className="text-right">Output Price</th>
              <th className="text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {providers?.flatMap(p =>
              p.models?.map(m => (
                <tr key={`${p.id}-${m.id}`} className="border-b">
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(`${p.id}-${m.id}`)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedModels([...selectedModels, `${p.id}-${m.id}`]);
                        } else {
                          setSelectedModels(selectedModels.filter(x => x !== `${p.id}-${m.id}`));
                        }
                      }}
                    />
                  </td>
                  <td className="py-2">{m.display_name}</td>
                  <td>{p.name}</td>
                  <td className="text-right">{m.context_window.toLocaleString()}</td>
                  <td className="text-right">${m.input_price_per_1m?.toFixed(2)}</td>
                  <td className="text-right">${m.output_price_per_1m?.toFixed(2)}</td>
                  <td className="text-right">
                    <button className="text-indigo-600">Compare</button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Benchmark Controls */}
      <div className="flex gap-2">
        <button
          onClick={handleBenchmark}
          disabled={selectedModels.length === 0}
          className="px-4 py-2 bg-indigo-600 text-white rounded-md disabled:opacity-50"
        >
          Run Benchmark
        </button>
      </div>

      {/* Benchmark Results */}
      {benchmarks && (
        <div>
          <h3 className="text-xl font-bold mb-4">Benchmark Results</h3>
          <BenchmarkChart data={benchmarks} />
        </div>
      )}
    </div>
  );
}
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ Model listesi görünüyor
2. ✅ Benchmark testleri çalışıyor
3. ✅ Karşılaştırma tablosu doğru

**SONRAKİ:** Phase 9 - Workspace Integration
