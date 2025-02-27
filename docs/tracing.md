# OpenTelemetry Tracing

Ollama uses [OpenTelemetry](https://opentelemetry.io) to capture and export trace spans.

By default Ollama will use `otlp` to export traces to `https://localhost:4318/v1/traces`. To disable trace exports set `OTEL_TRACES_EXPORTER=none`.

Tracing can be configured through a subset of [OpenTelemetry environment variables](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/), including

Env var                            | Default                            | Description
-----------------------------------|------------------------------------|------------
OTEL_SERVICE_NAME                  |                                    | Service name shown in traces. Will be the same across Ollama spans.
OTEL_TRACES_EXPORTER               | `otlp`                             | Controls trace export destination and method, `otlp` to send to an opentelemetry collector, `console` to print to `stdout`, `none` to disable exports.
OTEL_EXPORTER_OTLP_TRACES_INSECURE | `false`                            | Allow `otlp` exports to insecure (plain http) endpoints if set to `true`.
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL | `http/protobuf`                    | Serialization protocol for transporting trace data, either `grpc` or `http/protobuf`.
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT | `https://localhost:4318/v1/traces` | Where to send traces when using `otlp` exporter, defaults to `https://localhost:4317/` with `grpc` protocol and secure transport.
OTEL_TRACES_SAMPLER                | `parentbased_always_on`            | Controls when to capture and export traces. Use `parentbased_traceidratio` to capture only a fraction of all traces.
OTEL_TRACES_SAMPLER_ARG            | `1.0`                              | A fraction, in the range [0..1], of traces to save and export when using `traceidratio` sampler.

Ollama does not provide a trace span collector. See the [OpenTelemetry ecosystem registry](https://opentelemetry.io/ecosystem/registry) for `otlp` compatible collectors, including [opentelemetry-collector](https://opentelemetry.io/docs/collector).
