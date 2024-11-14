package api

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKeepAliveParsingFromJSON(t *testing.T) {
	tests := []struct {
		name string
		req  string
		exp  *Duration
	}{
		{
			name: "Positive Integer",
			req:  `{ "keep_alive": 42 }`,
			exp:  &Duration{42 * time.Second},
		},
		{
			name: "Positive Float",
			req:  `{ "keep_alive": 42.5 }`,
			exp:  &Duration{42 * time.Second},
		},
		{
			name: "Positive Integer String",
			req:  `{ "keep_alive": "42m" }`,
			exp:  &Duration{42 * time.Minute},
		},
		{
			name: "Negative Integer",
			req:  `{ "keep_alive": -1 }`,
			exp:  &Duration{math.MaxInt64},
		},
		{
			name: "Negative Float",
			req:  `{ "keep_alive": -3.14 }`,
			exp:  &Duration{math.MaxInt64},
		},
		{
			name: "Negative Integer String",
			req:  `{ "keep_alive": "-1m" }`,
			exp:  &Duration{math.MaxInt64},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var dec ChatRequest
			err := json.Unmarshal([]byte(test.req), &dec)
			require.NoError(t, err)

			assert.Equal(t, test.exp, dec.KeepAlive)
		})
	}
}

func TestDurationMarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		input    time.Duration
		expected time.Duration
	}{
		{
			"negative duration",
			time.Duration(-1),
			time.Duration(math.MaxInt64),
		},
		{
			"positive duration",
			42 * time.Second,
			42 * time.Second,
		},
		{
			"another positive duration",
			42 * time.Minute,
			42 * time.Minute,
		},
		{
			"zero duration",
			time.Duration(0),
			time.Duration(0),
		},
		{
			"max duration",
			time.Duration(math.MaxInt64),
			time.Duration(math.MaxInt64),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			b, err := json.Marshal(Duration{test.input})
			require.NoError(t, err)

			var d Duration
			err = json.Unmarshal(b, &d)
			require.NoError(t, err)

			assert.Equal(t, test.expected, d.Duration, "input %v, marshalled %v, got %v", test.input, string(b), d.Duration)
		})
	}
}

func TestUseMmapParsingFromJSON(t *testing.T) {
	tr := true
	fa := false
	tests := []struct {
		name string
		req  string
		exp  *bool
	}{
		{
			name: "Undefined",
			req:  `{ }`,
			exp:  nil,
		},
		{
			name: "True",
			req:  `{ "use_mmap": true }`,
			exp:  &tr,
		},
		{
			name: "False",
			req:  `{ "use_mmap": false }`,
			exp:  &fa,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var oMap map[string]interface{}
			err := json.Unmarshal([]byte(test.req), &oMap)
			require.NoError(t, err)
			opts := DefaultOptions()
			err = opts.FromMap(oMap)
			require.NoError(t, err)
			assert.Equal(t, test.exp, opts.UseMMap)
		})
	}
}

func TestUseMmapFormatParams(t *testing.T) {
	tr := true
	fa := false
	tests := []struct {
		name string
		req  map[string][]string
		exp  *bool
		err  error
	}{
		{
			name: "True",
			req: map[string][]string{
				"use_mmap": {"true"},
			},
			exp: &tr,
			err: nil,
		},
		{
			name: "False",
			req: map[string][]string{
				"use_mmap": {"false"},
			},
			exp: &fa,
			err: nil,
		},
		{
			name: "Numeric True",
			req: map[string][]string{
				"use_mmap": {"1"},
			},
			exp: &tr,
			err: nil,
		},
		{
			name: "Numeric False",
			req: map[string][]string{
				"use_mmap": {"0"},
			},
			exp: &fa,
			err: nil,
		},
		{
			name: "invalid string",
			req: map[string][]string{
				"use_mmap": {"foo"},
			},
			exp: nil,
			err: errors.New("invalid bool value [foo]"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resp, err := FormatParams(test.req)
			require.Equal(t, test.err, err)
			respVal, ok := resp["use_mmap"]
			if test.exp != nil {
				assert.True(t, ok, "resp: %v", resp)
				assert.Equal(t, *test.exp, *respVal.(*bool))
			}
		})
	}
}

func TestMessage_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`{"role": "USER", "content": "Hello!"}`, "user"},
		{`{"role": "System", "content": "Initialization complete."}`, "system"},
		{`{"role": "assistant", "content": "How can I help you?"}`, "assistant"},
		{`{"role": "TOOl", "content": "Access granted."}`, "tool"},
	}

	for _, test := range tests {
		var msg Message
		if err := json.Unmarshal([]byte(test.input), &msg); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		if msg.Role != test.expected {
			t.Errorf("role not lowercased: got %v, expected %v", msg.Role, test.expected)
		}
	}
}
