package server

import (
        "context"
        "net/http"
        "net/http/httptest"
        "testing"

        sdktrace "go.opentelemetry.io/otel/sdk/trace"
        "go.opentelemetry.io/otel/sdk/trace/tracetest"
        "go.opentelemetry.io/otel"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
)

func Test_Tracing(t *testing.T) {
        type testCase struct {
                Name     string
                Method   string
                Path     string
                NumSpansExpected int
                Expected func(t *testing.T, sr *tracetest.SpanRecorder)
        }

        testCases := []testCase{
                {
                        Name:   "Tags Handler",
                        Method: http.MethodGet,
                        Path:   "/api/tags",
                        NumSpansExpected:       1,
                        Expected: func(t *testing.T, sr *tracetest.SpanRecorder) {
                                span := sr.Ended()[0]
                                if span.Name() != "/api/tags" {
                                        t.Fatalf("Unexpected span name: %v", span.Name())
                                }

                        },
                },
                {
                        Name:   "Generate Handler (400 response)",
                        Method: http.MethodPost,
                        Path:   "/api/generate",
                        NumSpansExpected:       1,
                        Expected: func(t *testing.T, sr *tracetest.SpanRecorder) {
                                span := sr.Ended()[0]
                                if span.Name() != "/api/generate" {
                                        t.Fatalf("Unexpected span name: %v", span.Name())
                                }

                                if span.ChildSpanCount() < 1 {
                                        t.Fatalf("Expected 1+ child spans, observed %d", span.ChildSpanCount())
                                }

                        },
                },
		{
                        Name:   "Chat Handler (400 response)",
                        Method: http.MethodPost,
                        Path:   "/api/chat",
                        NumSpansExpected:       1,
                        Expected: func(t *testing.T, sr *tracetest.SpanRecorder) {
                                span := sr.Ended()[0]
                                if span.Name() != "/api/chat" {
                                        t.Fatalf("Unexpected span name: %v", span.Name())
                                }

                                if span.ChildSpanCount() < 1 {
                                        t.Fatalf("Expected 1+ child spans, observed %d", span.ChildSpanCount())
                                }

                        },
                }, 
		{
                        Name:   "Embed Handler (400 response)",
                        Method: http.MethodPost,
                        Path:   "/api/embed",
                        NumSpansExpected:       1,
                        Expected: func(t *testing.T, sr *tracetest.SpanRecorder) {
                                span := sr.Ended()[0]
                                if span.Name() != "/api/embed" {
                                        t.Fatalf("Unexpected span name: %v", span.Name())
                                }

                                if span.ChildSpanCount() < 1 {
                                        t.Fatalf("Expected 1+ child spans, observed %d", span.ChildSpanCount())
                                }

                        },
                },
        }
        t.Setenv("OTEL_TRACES_EXPORTER", "console")
        shutdownTracerProvider, err := initTracerProvider(context.TODO())
        if err != nil {
                t.Fatalf("failed to initialize tracer provider: %v", err)
        }
        defer shutdownTracerProvider(context.TODO())

        sr := tracetest.NewSpanRecorder()
        tp := otel.GetTracerProvider()
        tp.(*sdktrace.TracerProvider).RegisterSpanProcessor(sr)

	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	c, err := blob.Open(modelsDir)
	if err != nil {
		t.Fatalf("failed to open models dir: %v", err)
	}

	rc := &ollama.Registry{
		// Synced from routes_test.go. 
		HTTPClient: panicOnRoundTrip,
	}

        s := &Server{}
        router, err := s.GenerateRoutes(c, rc)
	if err != nil {
		t.Fatalf("failed to generate routes: %v", err)
	}

        httpSrv := httptest.NewServer(router)
        t.Cleanup(httpSrv.Close)

        for _, tc := range testCases {
                t.Run(tc.Name, func(t *testing.T) {
                        u := httpSrv.URL + tc.Path
                        req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
                        if err != nil {
                                t.Fatalf("failed to create request: %v", err)
                        }

                        resp, err := httpSrv.Client().Do(req)
                        if err != nil {
                                t.Fatalf("failed to do request: %v", err)
                        }
                        defer resp.Body.Close()

                        if len(sr.Ended()) != tc.NumSpansExpected {
                                t.Fatalf("Expected %d spans but recorded %d", tc.NumSpansExpected, len(sr.Ended()))
                        }

                        if tc.Expected != nil {
                                tc.Expected(t, sr)
                        }

                })

                sr.Reset()
        }
}
