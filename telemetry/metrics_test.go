package telemetry

import (
	"errors"
	"testing"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/sdk/resource"
)

func TestNewMetrics(t *testing.T) {
	tests := []struct {
		name           string
		meter          metric.Meter
		expectedMetric string
	}{
		{
			name:           "Valid Meter",
			meter:          noop.NewMeterProvider().Meter("test"),
			expectedMetric: "http_requests_total",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics := NewMetrics(tt.meter)

			// Ensure the metric is registered correctly
			assert.NotNil(t, metrics)
			assert.NotNil(t, metrics.Requests)
		})
	}
}

func TestNewPrometheusMeterProvider(t *testing.T) {
	tests := []struct {
		name           string
		wantErr        bool
		mockPrometheus func() (*prometheus.Exporter, error)
		expectedError  error
	}{
		{
			name:    "Successful creation of meter provider",
			wantErr: false,
			mockPrometheus: func() (*prometheus.Exporter, error) {
				return &prometheus.Exporter{
					Reader: sdkmetric.NewManualReader(),
				}, nil
			},
		},
		{
			name:          "Error on resource creation",
			wantErr:       true,
			expectedError: errors.New("error creating prometheus resource"),
			mockPrometheus: func() (*prometheus.Exporter, error) {
				return nil, errors.New("error creating prometheus resource")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := resource.NewSchemaless() // Use an empty resource for testing.
			exp, _ := tt.mockPrometheus()
			mp, err := NewPrometheusMeterProvider(res, exp)

			if tt.wantErr {
				assert.NotNil(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, mp)
			}
		})
	}
}
