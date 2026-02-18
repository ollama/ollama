package server

import (
	"context"
	"log/slog"
	"os"

	"github.com/ollama/ollama/version"
	"go.opentelemetry.io/contrib/exporters/autoexport"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
)

const (
	serviceName = "ollama"
)

// isTracingEnabled checks if OpenTelemetry tracing is explicitly configured
// via environment variables. Returns true if any OTEL configuration is present.
func isTracingEnabled() bool {
	// Check for common OTEL environment variables
	otelVars := []string{
		"OTEL_TRACES_EXPORTER",
		"OTEL_EXPORTER_OTLP_ENDPOINT",
		"OTEL_EXPORTER_OTLP_PROTOCOL",
		"OTEL_EXPORTER_OTLP_HEADERS",
		"OTEL_SERVICE_NAME",
	}

	for _, env := range otelVars {
		if os.Getenv(env) != "" {
			return true
		}
	}

	return false
}

// InitTracer initializes the OpenTelemetry tracer provider with autoexport.
// The exporter is automatically configured based on environment variables:
//   - OTEL_TRACES_EXPORTER: otlp, console, none (default: otlp)
//   - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
//   - OTEL_EXPORTER_OTLP_HEADERS: OTLP headers
//   - OTEL_EXPORTER_OTLP_PROTOCOL: grpc or http/protobuf (default: grpc)
//
// It returns a shutdown function that should be called before application exit.
func InitTracer(ctx context.Context) (func(context.Context) error, error) {
	// Create exporter using autoexport - automatically selects based on env vars
	exporter, err := autoexport.NewSpanExporter(ctx)
	if err != nil {
		return nil, err
	}

	// Create resource with service information
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(version.Version),
		),
	)
	if err != nil {
		return nil, err
	}

	// Create trace provider
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)

	// Register as global tracer provider
	otel.SetTracerProvider(tp)

	// Set global propagator to W3C Trace Context (supports trace ID propagation)
	otel.SetTextMapPropagator(
		propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{},
			propagation.Baggage{},
		),
	)

	slog.Info("OpenTelemetry tracer initialized")

	// Return shutdown function
	return tp.Shutdown, nil
}
