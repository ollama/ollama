import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import {
  ChartBarIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from "@heroicons/react/20/solid";

interface PerformanceMetrics {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  average_response_time: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost: number;
  provider_stats: Record<string, ProviderMetrics>;
  start_time: string;
  end_time: string;
}

interface ProviderMetrics {
  provider_name: string;
  request_count: number;
  success_count: number;
  failure_count: number;
  average_latency: number;
  total_tokens: number;
  total_cost: number;
  error_rate: number;
  tokens_per_second: number;
}

interface Alert {
  level: string;
  type: string;
  message: string;
  timestamp: string;
}

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const formatCurrency = (amount: number): string => {
  return `$${amount.toFixed(4)}`;
};

export default function PerformanceDashboard() {
  const { data: metrics, isLoading: loadingMetrics } = useQuery({
    queryKey: ["metrics", "performance"],
    queryFn: async () => {
      const res = await fetch("/api/metrics/performance");
      if (!res.ok) throw new Error("Failed to fetch performance metrics");
      return res.json() as Promise<PerformanceMetrics>;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { data: systemMetrics } = useQuery({
    queryKey: ["metrics", "system"],
    queryFn: async () => {
      const res = await fetch("/api/metrics/system");
      if (!res.ok) throw new Error("Failed to fetch system metrics");
      return res.json();
    },
    refetchInterval: 5000,
  });

  const { data: alerts } = useQuery({
    queryKey: ["metrics", "alerts"],
    queryFn: async () => {
      const res = await fetch("/api/metrics/alerts");
      if (!res.ok) throw new Error("Failed to fetch alerts");
      const data = await res.json();
      return data.alerts as Alert[];
    },
    refetchInterval: 10000,
  });

  const { data: costBreakdown } = useQuery({
    queryKey: ["metrics", "cost"],
    queryFn: async () => {
      const res = await fetch("/api/metrics/cost");
      if (!res.ok) throw new Error("Failed to fetch cost breakdown");
      const data = await res.json();
      return data.breakdown as Record<string, number>;
    },
  });

  if (loadingMetrics) {
    return (
      <div className="space-y-4">
        <div className="text-sm text-gray-400">Loading metrics...</div>
      </div>
    );
  }

  const successRate = metrics
    ? ((metrics.successful_requests / metrics.total_requests) * 100).toFixed(1)
    : "0.0";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-semibold text-white">Performance Dashboard</h3>
        <p className="text-sm text-gray-400">
          Real-time monitoring and analytics
        </p>
      </div>

      {/* Alerts */}
      {alerts && alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, idx) => (
            <div
              key={idx}
              className={`flex items-center gap-3 p-3 rounded-lg border ${
                alert.level === "error"
                  ? "bg-red-900/20 border-red-700"
                  : alert.level === "warning"
                  ? "bg-yellow-900/20 border-yellow-700"
                  : "bg-blue-900/20 border-blue-700"
              }`}
            >
              <ExclamationTriangleIcon
                className={`h-5 w-5 flex-shrink-0 ${
                  alert.level === "error"
                    ? "text-red-500"
                    : alert.level === "warning"
                    ? "text-yellow-500"
                    : "text-blue-500"
                }`}
              />
              <div className="flex-1">
                <p className="text-sm text-white">{alert.message}</p>
                <p className="text-xs text-gray-400">
                  {new Date(alert.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
          <div className="flex items-center gap-2 mb-2">
            <ChartBarIcon className="h-5 w-5 text-blue-400" />
            <span className="text-sm text-gray-400">Total Requests</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics?.total_requests.toLocaleString() || 0}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {successRate}% success rate
          </div>
        </div>

        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
          <div className="flex items-center gap-2 mb-2">
            <ClockIcon className="h-5 w-5 text-green-400" />
            <span className="text-sm text-gray-400">Avg Response Time</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics ? formatDuration(metrics.average_response_time / 1000000) : "0ms"}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {systemMetrics?.requests_per_min.toFixed(1) || 0} req/min
          </div>
        </div>

        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
          <div className="flex items-center gap-2 mb-2">
            <ChartBarIcon className="h-5 w-5 text-purple-400" />
            <span className="text-sm text-gray-400">Total Tokens</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics
              ? (metrics.total_input_tokens + metrics.total_output_tokens).toLocaleString()
              : 0}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {metrics?.total_input_tokens.toLocaleString() || 0} in /{" "}
            {metrics?.total_output_tokens.toLocaleString() || 0} out
          </div>
        </div>

        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
          <div className="flex items-center gap-2 mb-2">
            <CurrencyDollarIcon className="h-5 w-5 text-yellow-400" />
            <span className="text-sm text-gray-400">Total Cost</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics ? formatCurrency(metrics.total_cost) : "$0.0000"}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Since {metrics ? new Date(metrics.start_time).toLocaleDateString() : "N/A"}
          </div>
        </div>
      </div>

      {/* Provider Stats */}
      {metrics && metrics.provider_stats && Object.keys(metrics.provider_stats).length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-white mb-3">Provider Performance</h4>
          <div className="space-y-3">
            {Object.entries(metrics.provider_stats).map(([provider, stats]) => (
              <div
                key={provider}
                className="rounded-lg border border-gray-700 bg-gray-800 p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white capitalize">{provider}</span>
                    <Badge>{stats.request_count} requests</Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    {stats.error_rate < 5 ? (
                      <CheckCircleIcon className="h-5 w-5 text-green-500" />
                    ) : (
                      <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                    )}
                    <span className="text-sm text-gray-400">
                      {stats.error_rate.toFixed(1)}% error
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div>
                    <span className="text-gray-500">Avg Latency:</span>
                    <span className="text-white ml-2">
                      {formatDuration(stats.average_latency / 1000000)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Total Tokens:</span>
                    <span className="text-white ml-2">
                      {stats.total_tokens.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Tokens/sec:</span>
                    <span className="text-white ml-2">
                      {stats.tokens_per_second.toFixed(1)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Cost:</span>
                    <span className="text-white ml-2">
                      {formatCurrency(stats.total_cost)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cost Breakdown */}
      {costBreakdown && Object.keys(costBreakdown).length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-white mb-3">Cost Breakdown</h4>
          <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
            <div className="space-y-2">
              {Object.entries(costBreakdown).map(([provider, cost]) => {
                const totalCost = metrics?.total_cost || 1;
                const percentage = (cost / totalCost) * 100;

                return (
                  <div key={provider} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white capitalize">{provider}</span>
                      <span className="text-gray-400">
                        {formatCurrency(cost)} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* System Metrics */}
      {systemMetrics && (
        <div>
          <h4 className="text-md font-semibold text-white mb-3">System Health</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
              <div className="text-sm text-gray-400 mb-1">Uptime</div>
              <div className="text-lg font-semibold text-white">
                {Math.floor(systemMetrics.uptime_seconds / 3600)}h{" "}
                {Math.floor((systemMetrics.uptime_seconds % 3600) / 60)}m
              </div>
            </div>
            <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
              <div className="text-sm text-gray-400 mb-1">CPU Usage</div>
              <div className="text-lg font-semibold text-white">
                {systemMetrics.cpu_usage.toFixed(1)}%
              </div>
            </div>
            <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
              <div className="text-sm text-gray-400 mb-1">Memory Usage</div>
              <div className="text-lg font-semibold text-white">
                {systemMetrics.memory_usage.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
