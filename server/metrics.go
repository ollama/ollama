package server

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// MetricsHandler serves scheduler state in the Prometheus text exposition format
// (https://prometheus.io/docs/instrumenting/exposition_formats/). It is registered only
// when OLLAMA_METRICS is enabled, mirroring llama.cpp's opt-in --metrics endpoint. The
// exposition is written by hand so no metrics library is added as a dependency.
func (s *Server) MetricsHandler(c *gin.Context) {
	m := s.sched.collectMetrics()

	var b strings.Builder
	gauge := func(name, help string, value int) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s %d\n", name, help, name, name, value)
	}
	gauge("ollama_requests_queued", "Number of requests waiting for a model runner.", m.requestsQueued)
	gauge("ollama_queue_capacity", "Maximum number of requests that can be queued (OLLAMA_MAX_QUEUE).", m.queueCapacity)
	gauge("ollama_models_loaded", "Number of models currently loaded in memory.", m.modelsLoaded)

	c.Header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	c.String(http.StatusOK, b.String())
}
