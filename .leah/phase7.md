# PHASE 7: PERFORMANCE MONITOR VE API COST TRACKING

## 📋 HEDEFLER
1. ✅ Real-time token tracking
2. ✅ Cost calculation (tüm providers)
3. ✅ Performance metrics (tokens/s, latency)
4. ✅ Usage analytics & charts
5. ✅ Budget alerts
6. ✅ Export reports (CSV/JSON)

## 🏗️ MİMARİ

### Metrics Schema
```sql
CREATE TABLE performance_metrics (
  id INTEGER PRIMARY KEY,
  chat_id TEXT,
  message_id INTEGER,
  provider_id TEXT,
  model_name TEXT,

  -- Token Metrics
  input_tokens INTEGER,
  output_tokens INTEGER,
  total_tokens INTEGER,

  -- Performance Metrics
  duration_ms INTEGER,
  tokens_per_second REAL,
  time_to_first_token_ms INTEGER,

  -- Cost Metrics
  input_cost_usd REAL,
  output_cost_usd REAL,
  total_cost_usd REAL,

  -- Quality Metrics
  cache_hit BOOLEAN,
  error_occurred BOOLEAN,
  retry_count INTEGER,

  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_chat ON performance_metrics(chat_id);
CREATE INDEX idx_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_metrics_provider ON performance_metrics(provider_id);
```

## 📁 DOSYALAR

### 1. Metrics Collector
**Dosya:** `/home/user/ollama/metrics/collector.go` (YENİ)

```go
type MetricsCollector struct {
    store MetricsStore
}

type Metric struct {
    ChatID              string
    MessageID           int64
    ProviderID          string
    ModelName           string
    InputTokens         int
    OutputTokens        int
    TotalTokens         int
    DurationMs          int64
    TokensPerSecond     float64
    TimeToFirstTokenMs  int64
    InputCostUSD        float64
    OutputCostUSD       float64
    TotalCostUSD        float64
    CacheHit            bool
    ErrorOccurred       bool
    RetryCount          int
}

func (mc *MetricsCollector) Track(metric *Metric) error {
    return mc.store.Save(metric)
}

func (mc *MetricsCollector) GetChatStats(chatID string) (*ChatStats, error) {
    metrics, err := mc.store.GetByChatID(chatID)
    if err != nil {
        return nil, err
    }

    stats := &ChatStats{
        TotalMessages:   len(metrics),
        TotalTokens:     0,
        TotalCost:       0,
        AvgTokensPerSec: 0,
        AvgLatency:      0,
    }

    var totalTPS, totalLatency float64

    for _, m := range metrics {
        stats.TotalTokens += m.TotalTokens
        stats.TotalCost += m.TotalCostUSD
        totalTPS += m.TokensPerSecond
        totalLatency += float64(m.DurationMs)
    }

    if len(metrics) > 0 {
        stats.AvgTokensPerSec = totalTPS / float64(len(metrics))
        stats.AvgLatency = totalLatency / float64(len(metrics))
    }

    return stats, nil
}
```

### 2. Performance Dashboard
**Dosya:** `/home/user/ollama/app/ui/app/src/components/PerformanceDashboard.tsx` (YENİ)

```typescript
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export function PerformanceDashboard() {
  const { data: stats } = usePerformanceStats('24h');

  const tokenChart = {
    labels: stats?.timeline.map(t => t.time) || [],
    datasets: [
      {
        label: 'Tokens/Second',
        data: stats?.timeline.map(t => t.tokens_per_sec) || [],
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
      },
    ],
  };

  const costChart = {
    labels: stats?.by_model.map(m => m.model_name) || [],
    datasets: [
      {
        label: 'Cost (USD)',
        data: stats?.by_model.map(m => m.total_cost) || [],
        backgroundColor: 'rgba(34, 197, 94, 0.7)',
      },
    ],
  };

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total Tokens"
          value={stats?.total_tokens.toLocaleString()}
          icon={<ChartBarIcon />}
        />
        <MetricCard
          title="Total Cost"
          value={`$${stats?.total_cost.toFixed(4)}`}
          icon={<BanknotesIcon />}
        />
        <MetricCard
          title="Avg Speed"
          value={`${stats?.avg_tokens_per_sec.toFixed(1)} tok/s`}
          icon={<BoltIcon />}
        />
        <MetricCard
          title="Avg Latency"
          value={`${stats?.avg_latency.toFixed(0)}ms`}
          icon={<ClockIcon />}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Performance Over Time</h3>
          <Line data={tokenChart} options={{ responsive: true }} />
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Cost by Model</h3>
          <Bar data={costChart} options={{ responsive: true }} />
        </div>
      </div>

      {/* Detailed Table */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Detailed Metrics</h3>
        <table className="w-full">
          <thead>
            <tr className="border-b dark:border-gray-700">
              <th className="text-left py-2">Model</th>
              <th className="text-right">Tokens</th>
              <th className="text-right">Avg Speed</th>
              <th className="text-right">Avg Latency</th>
              <th className="text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {stats?.by_model.map(model => (
              <tr key={model.model_name} className="border-b dark:border-gray-700">
                <td className="py-2">{model.model_name}</td>
                <td className="text-right">{model.total_tokens.toLocaleString()}</td>
                <td className="text-right">{model.avg_tokens_per_sec.toFixed(1)} tok/s</td>
                <td className="text-right">{model.avg_latency.toFixed(0)}ms</td>
                <td className="text-right">${model.total_cost.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

### 3. Budget Alerts
**Dosya:** `/home/user/ollama/metrics/alerts.go` (YENİ)

```go
type BudgetAlert struct {
    ID          string
    UserID      string
    BudgetUSD   float64
    Period      string  // daily, weekly, monthly
    CurrentSpend float64
    Threshold   float64  // 0.8 = 80%
    Enabled     bool
}

func (mc *MetricsCollector) CheckBudgetAlerts(userID string) ([]*BudgetAlert, error) {
    alerts, err := mc.store.GetUserAlerts(userID)
    if err != nil {
        return nil, err
    }

    triggered := make([]*BudgetAlert, 0)

    for _, alert := range alerts {
        if !alert.Enabled {
            continue
        }

        currentSpend, err := mc.GetPeriodSpend(userID, alert.Period)
        if err != nil {
            continue
        }

        alert.CurrentSpend = currentSpend

        if currentSpend >= alert.BudgetUSD * alert.Threshold {
            triggered = append(triggered, alert)
        }
    }

    return triggered, nil
}
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ Real-time metrics tracking
2. ✅ Charts güzel görünüyor
3. ✅ Cost tracking doğru
4. ✅ Budget alerts çalışıyor
5. ✅ Export CSV/JSON çalışıyor

**SONRAKİ:** Phase 8 - Model Management
