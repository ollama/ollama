package server

import (
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/version"
)

type metricLabel struct {
	name  string
	value string
}

func formatLabels(labels ...metricLabel) string {
	if len(labels) == 0 {
		return ""
	}

	labelNames := make([]string, 0, len(labels))
	labelValues := make(map[string]string, len(labels))
	for _, label := range labels {
		labelNames = append(labelNames, label.name)
		labelValues[label.name] = strings.ReplaceAll(
			strings.ReplaceAll(
				strings.ReplaceAll(
					label.value,
					"\\",
					"\\\\",
				),
				"\n",
				"\\n",
			),
			"\"",
			"\\\"",
		)
	}
	sort.Strings(labelNames)

	var renderedLabels strings.Builder
	renderedLabels.WriteByte('{')
	for i, name := range labelNames {
		if i > 0 {
			renderedLabels.WriteByte(',')
		}
		renderedLabels.WriteString(name)
		renderedLabels.WriteByte('=')
		renderedLabels.WriteByte('"')
		renderedLabels.WriteString(labelValues[name])
		renderedLabels.WriteByte('"')
	}
	renderedLabels.WriteByte('}')
	return renderedLabels.String()
}

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

	addCounter := func(name, help string, values map[httpRequestMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		samples := make([]string, 0, len(values))
		for key, count := range values {
			labelText := formatLabels(
				metricLabel{name: "action", value: key.action},
				metricLabel{name: "status_code", value: strconv.Itoa(key.statusCode)},
				metricLabel{name: "status", value: key.status},
			)
			samples = append(samples, name+labelText+" "+strconv.FormatFloat(float64(count), 'f', 6, 64))
		}
		sort.Strings(samples)

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintln(&b, sample)
		}
	}

	addDurationCounter := func(name, help string, values map[modelMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		samples := make([]string, 0, len(values))
		for key, durationNanoseconds := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, name+formatLabels(labels...)+" "+strconv.FormatFloat(float64(durationNanoseconds)/float64(time.Second), 'f', 6, 64))
		}
		sort.Strings(samples)

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintln(&b, sample)
		}
	}

	addCountCounter := func(name, help string, values map[modelMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		samples := make([]string, 0, len(values))
		for key, count := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, name+formatLabels(labels...)+" "+strconv.FormatUint(count, 10))
		}
		sort.Strings(samples)

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintln(&b, sample)
		}
	}

	addGauge := func(name, help string, values map[modelMetricKey]uint64, labelKey string, labelValue string) {
		if len(values) == 0 && labelKey == "" && labelValue == "" {
			return
		}

		samples := make([]string, 0, len(values))
		for key, value := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, name+formatLabels(labels...)+" "+strconv.FormatUint(value, 10))
		}
		if labelKey != "" && labelValue != "" {
			samples = append(samples, name+formatLabels(metricLabel{name: labelKey, value: labelValue})+" "+strconv.FormatInt(m.startUnix, 10))
		}
		sort.Strings(samples)

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintln(&b, sample)
		}
	}

	gauge("ollama_requests_queued", "Number of requests waiting for a model runner.", m.requestsQueued)
	gauge("ollama_queue_capacity", "Maximum number of requests that can be queued (OLLAMA_MAX_QUEUE).", m.queueCapacity)
	gauge("ollama_models_loaded", "Number of models currently loaded in memory.", m.modelsLoaded)
	addGauge("ollama_build_info", "Ollama build information.", map[modelMetricKey]uint64{}, "version", version.Version)
	addCounter("http_requests_total", "The total number of requests on the endpoints.", m.requestsTotal)
	addDurationCounter("ollama_total_duration_seconds", "The request total duration in seconds.", m.totalDurationNanoseconds)
	addDurationCounter("ollama_load_duration_seconds", "The request load duration in seconds.", m.loadDurationNanoseconds)
	addCountCounter("ollama_prompt_eval_total", "The number of prompt token evaluated.", m.promptEvalCount)
	addDurationCounter("ollama_prompt_eval_duration_seconds", "The prompt evaluation duration in seconds.", m.promptEvalDurationNanoseconds)
	addCountCounter("ollama_eval_total", "The number of token evaluated.", m.evalCount)
	addDurationCounter("ollama_eval_duration_seconds", "The token evaluation duration in seconds.", m.evalDurationNanoseconds)
	addGauge("ollama_peak_memory_bytes", "The peak memory used during computation in bytes.", m.modelPeakMemoryBytes, "", "")

	c.Header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	c.String(http.StatusOK, b.String())
}
