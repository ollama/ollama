package server

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// ollama_http_requests_total counts HTTP requests by method and status code.
	ollamaHTTPRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ollama_http_requests_total",
			Help: "Total number of HTTP requests handled, partitioned by method and status code.",
		},
		[]string{"method", "code"},
	)

	// ollama_http_request_duration_seconds observes request latency by method.
	ollamaHTTPRequestDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "ollama_http_request_duration_seconds",
			Help:    "HTTP request latency in seconds, partitioned by method.",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		},
		[]string{"method"},
	)

	// ollama_http_errors_total counts 5xx server errors by method.
	ollamaHTTPErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ollama_http_errors_total",
			Help: "Total number of HTTP 5xx server errors, partitioned by method.",
		},
		[]string{"method"},
	)
)

func init() {
	prometheus.MustRegister(
		ollamaHTTPRequestsTotal,
		ollamaHTTPRequestDurationSeconds,
		ollamaHTTPErrorsTotal,
	)
}

// MetricsMiddleware records request counts, latency, and 5xx error counts.
func MetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		c.Next()

		method := c.Request.Method
		code := c.Writer.Status()
		seconds := time.Since(start).Seconds()

		ollamaHTTPRequestsTotal.WithLabelValues(method, strconv.Itoa(code)).Inc()
		ollamaHTTPRequestDurationSeconds.WithLabelValues(method).Observe(seconds)

		if code >= http.StatusInternalServerError {
			ollamaHTTPErrorsTotal.WithLabelValues(method).Inc()
		}
	}
}

// MetricsHandler exposes the Prometheus /metrics endpoint.
func MetricsHandler() gin.HandlerFunc {
	return gin.WrapH(promhttp.Handler())
}
