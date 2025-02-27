package server

import (
        "context"
        "fmt"
        "log/slog"

        "github.com/gin-gonic/gin"

        "go.opentelemetry.io/contrib/exporters/autoexport"
        "go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
        "go.opentelemetry.io/otel"
        sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func TracingMiddleware() gin.HandlerFunc {
        return otelgin.Middleware("ollama")
}

func initTracerProvider(ctx context.Context) (func(context.Context) error, error) {
        exporter, err := autoexport.NewSpanExporter(ctx)
        slog.Info("Initialized opentelemetry exporter", "type", fmt.Sprintf("%T", exporter))
        if err != nil {
                return nil, err
        }

        tp := sdktrace.NewTracerProvider(
                sdktrace.WithBatcher(exporter),
        )
        otel.SetTracerProvider(tp)
        return tp.Shutdown, nil
}
