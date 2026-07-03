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

// MetricsHandler serves scheduler state in the Prometheus text exposition format
// (https://prometheus.io/docs/instrumenting/exposition_formats/). It is registered only
// when OLLAMA_METRICS is enabled, mirroring llama.cpp's opt-in --metrics endpoint. The
// exposition is written by hand so no metrics library is added as a dependency.
func (s *Server) MetricsHandler(c *gin.Context) {
	m := s.sched.collectMetrics()

	var b strings.Builder
	type metricLabel struct {
		name  string
		value string
	}

	formatLabels := func(labels ...metricLabel) string {
		if len(labels) == 0 {
			return ""
		}

		sort.Slice(labels, func(i, j int) bool {
			return labels[i].name < labels[j].name
		})

		var renderedLabels strings.Builder
		renderedLabels.WriteByte('{')
		for i, label := range labels {
			if i > 0 {
				renderedLabels.WriteByte(',')
			}
			renderedLabels.WriteString(label.name)
			renderedLabels.WriteByte('=')
			renderedLabels.WriteByte('"')
			renderedLabels.WriteString(strings.ReplaceAll(
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
			))
			renderedLabels.WriteByte('"')
		}
		renderedLabels.WriteByte('}')
		return renderedLabels.String()
	}

	gauge := func(name, help string, value int) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s %d\n", name, help, name, name, value)
	}

	addCounter := func(name, help string, values map[httpRequestMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		type sample struct {
			labelText string
			value     string
		}
		samples := make([]sample, 0, len(values))
		for key, count := range values {
			labelText := formatLabels(
				metricLabel{name: "action", value: key.action},
				metricLabel{name: "status_code", value: strconv.Itoa(key.statusCode)},
				metricLabel{name: "status", value: key.status},
			)
			samples = append(samples, sample{
				labelText: labelText,
				value:     strconv.FormatFloat(float64(count), 'f', 6, 64),
			})
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].labelText < samples[j].labelText
		})

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintf(&b, "%s%s %s\n", name, sample.labelText, sample.value)
		}
	}

	addDurationCounter := func(name, help string, values map[modelMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		type sample struct {
			labelText string
			value     float64
		}
		samples := make([]sample, 0, len(values))
		for key, durationNanoseconds := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, sample{
				labelText: formatLabels(labels...),
				value:     float64(durationNanoseconds) / float64(time.Second),
			})
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].labelText < samples[j].labelText
		})

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintf(&b, "%s%s %s\n", name, sample.labelText, strconv.FormatFloat(sample.value, 'f', 6, 64))
		}
	}

	addCountCounter := func(name, help string, values map[modelMetricKey]uint64) {
		if len(values) == 0 {
			return
		}

		type sample struct {
			labelText string
			value     uint64
		}
		samples := make([]sample, 0, len(values))
		for key, count := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, sample{
				labelText: formatLabels(labels...),
				value:     count,
			})
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].labelText < samples[j].labelText
		})

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", name, help, name)
		for _, sample := range samples {
			fmt.Fprintf(&b, "%s%s %d\n", name, sample.labelText, sample.value)
		}
	}

	addGauge := func(name, help string, values map[modelMetricKey]uint64, labelKey string, labelValue string) {
		if len(values) == 0 && labelKey == "" && labelValue == "" {
			return
		}

		type sample struct {
			labelText string
			value     string
		}
		samples := make([]sample, 0, len(values))
		for key, value := range values {
			labels := []metricLabel{
				{name: "model", value: key.model},
			}
			if key.reasonSet {
				labels = append(labels, metricLabel{name: "reason", value: key.reason})
			}
			samples = append(samples, sample{
				labelText: formatLabels(labels...),
				value:     strconv.FormatUint(value, 10),
			})
		}
		if labelKey != "" && labelValue != "" {
			samples = append(samples, sample{
				labelText: formatLabels(metricLabel{name: labelKey, value: labelValue}),
				value:     strconv.FormatInt(m.startUnix, 10),
			})
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].labelText < samples[j].labelText
		})

		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n", name, help, name)
		for _, sample := range samples {
			if sample.labelText == "" {
				fmt.Fprintf(&b, "%s %s\n", name, sample.value)
			} else {
				fmt.Fprintf(&b, "%s%s %s\n", name, sample.labelText, sample.value)
			}
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
