package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
)

func TestNewCounterVec(t *testing.T) {
	testCases := []struct {
		subsystem  string
		name       string
		help       string
		labelNames []string
	}{
		{
			subsystem:  "subsys1",
			name:       "counter1",
			help:       "help1",
			labelNames: []string{"action", "status_code", "status"},
		},
		{
			subsystem:  "subsys2",
			name:       "counter2",
			help:       "help2",
			labelNames: []string{"method", "status"},
		},
		// Add more test cases as needed
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Call the function under test
			counterVec := newCounterVec(tc.subsystem, tc.name, tc.help)

			// Verify the returned CounterVec
			assert.NotNil(t, counterVec)
			assert.IsType(t, &prometheus.CounterVec{}, counterVec)
		})
	}
}
