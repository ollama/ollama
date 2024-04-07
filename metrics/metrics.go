package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	promNamespace      = "ollama"
	promModelSubsystem = "model"
	promChatSubsystem  = "chat"
	promBlobSubsystem  = "blob"
)

// Define a function to create Prometheus counters with common namespace and additional values as parameters.
func newCounterVec(subsystem, name, help string) *prometheus.CounterVec {
	return promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: promNamespace,
		Subsystem: subsystem,
		Name:      name,
		Help:      help,
	},
		[]string{"action", "status_code", "status"},
	)
}

// RequestMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	RequestTotal = newCounterVec(promModelSubsystem, "requests_total", "The total number of requests on all endpoints.")
)

// ModelPullMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelPullTotal = newCounterVec(promModelSubsystem, "pull_requests_total", "The total number of model pulls that have been attempted.")
)

// ModelCreateMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelCreateTotal = newCounterVec(promModelSubsystem, "create_requests_total", "The total number of model creations that have been attempted.")
)

// ModelListMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelListTotal = newCounterVec(promModelSubsystem, "list_requests_total", "The total number of model list requets that have been attempted.")
)

// ModelCopyMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelCopyTotal = newCounterVec(promModelSubsystem, "copy_requests_total", "The total number of model copy requets that have been attempted.")
)

// ModelDeleteMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelDeleteTotal = newCounterVec(promModelSubsystem, "delete_requests_total", "The total number of model delete requets that have been attempted.")
)

// ModelShowMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ModelShowTotal = newCounterVec(promModelSubsystem, "show_requests_total", "The total number of model show requets that have been attempted.")
)

// ChatMetrics tracks the number of model pulls that have been attempted and the number of successes and failures.
var (
	ChatTotal = newCounterVec(promChatSubsystem, "chat_requests_total", "The total number of requets that have been attempted on chat endpoint.")
)
