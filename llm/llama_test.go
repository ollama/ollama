package llm

import (
	"bytes"
	"testing"
)

func TestCheckStopConditions(t *testing.T) {
	tests := map[string]struct {
		b                      string
		stop                   []string
		wantB                  string
		wantStop               bool
		wantEndsWithStopPrefix bool
	}{
		"not present": {
			b:                      "abc",
			stop:                   []string{"x"},
			wantStop:               false,
			wantEndsWithStopPrefix: false,
		},
		"exact": {
			b:                      "abc",
			stop:                   []string{"abc"},
			wantStop:               true,
			wantEndsWithStopPrefix: false,
		},
		"substring": {
			b:                      "abc",
			stop:                   []string{"b"},
			wantB:                  "a",
			wantStop:               true,
			wantEndsWithStopPrefix: false,
		},
		"prefix 1": {
			b:                      "abc",
			stop:                   []string{"abcd"},
			wantStop:               false,
			wantEndsWithStopPrefix: true,
		},
		"prefix 2": {
			b:                      "abc",
			stop:                   []string{"bcd"},
			wantStop:               false,
			wantEndsWithStopPrefix: true,
		},
		"prefix 3": {
			b:                      "abc",
			stop:                   []string{"cd"},
			wantStop:               false,
			wantEndsWithStopPrefix: true,
		},
		"no prefix": {
			b:                      "abc",
			stop:                   []string{"bx"},
			wantStop:               false,
			wantEndsWithStopPrefix: false,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var b bytes.Buffer
			b.WriteString(test.b)
			stop, endsWithStopPrefix := handleStopSequences(&b, test.stop)
			if test.wantB != "" {
				gotB := b.String()
				if gotB != test.wantB {
					t.Errorf("got b %q, want %q", gotB, test.wantB)
				}
			}
			if stop != test.wantStop {
				t.Errorf("got stop %v, want %v", stop, test.wantStop)
			}
			if endsWithStopPrefix != test.wantEndsWithStopPrefix {
				t.Errorf("got endsWithStopPrefix %v, want %v", endsWithStopPrefix, test.wantEndsWithStopPrefix)
			}
		})
	}
}
