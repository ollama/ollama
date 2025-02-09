package api

import (
    "encoding/json"
    "errors"
    "math"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "bytes"
    "io"
    "os"
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

// Test generated using Keploy
func TestStatusErrorFormatting(t *testing.T) {
    tests := []struct {
        name     string
        input    StatusError
        expected string
    }{
        {
            name: "Both Status and ErrorMessage present",
            input: StatusError{
                Status:       "404 Not Found",
                ErrorMessage: "Resource not found",
            },
            expected: "404 Not Found: Resource not found",
        },
        {
            name: "Only Status present",
            input: StatusError{
                Status: "500 Internal Server Error",
            },
            expected: "500 Internal Server Error",
        },
        {
            name: "Only ErrorMessage present",
            input: StatusError{
                ErrorMessage: "Unexpected error occurred",
            },
            expected: "Unexpected error occurred",
        },
        {
            name: "Neither Status nor ErrorMessage present",
            input: StatusError{},
            expected: "something went wrong, please see the ollama server logs for details",
        },
    }

    for _, test := range tests {
        t.Run(test.name, func(t *testing.T) {
            result := test.input.Error()
            assert.Equal(t, test.expected, result)
        })
    }
}


// Test generated using Keploy
func TestToolCallFunctionArgumentsString(t *testing.T) {
    args := ToolCallFunctionArguments{
        "key1": "value1",
        "key2": 42,
        "key3": true,
    }

    result := args.String()
    expected := `{"key1":"value1","key2":42,"key3":true}`

    assert.JSONEq(t, expected, result)
}


// Test generated using Keploy
func TestMetricsSummary(t *testing.T) {
    metrics := Metrics{
        TotalDuration:      5 * time.Second,
        LoadDuration:       2 * time.Second,
        PromptEvalCount:    10,
        PromptEvalDuration: 1 * time.Second,
        EvalCount:          20,
        EvalDuration:       3 * time.Second,
    }

    // Capture stderr output
    oldStderr := os.Stderr
    r, w, _ := os.Pipe()
    os.Stderr = w

    metrics.Summary()

    w.Close()
    os.Stderr = oldStderr

    var buf bytes.Buffer
    io.Copy(&buf, r)

    output := buf.String()
    assert.Contains(t, output, "total duration:       5s")
    assert.Contains(t, output, "load duration:        2s")
    assert.Contains(t, output, "prompt eval count:    10 token(s)")
    assert.Contains(t, output, "prompt eval duration: 1s")
    assert.Contains(t, output, "eval count:           20 token(s)")
    assert.Contains(t, output, "eval duration:        3s")
}


// Test generated using Keploy
func TestToolsString(t *testing.T) {
    tools := Tools{
        {Type: "tool1", Function: ToolFunction{Name: "func1"}},
        {Type: "tool2", Function: ToolFunction{Name: "func2"}},
    }

    result := tools.String()
    expected := `[{"type":"tool1","function":{"name":"func1","description":"","parameters":{"type":"","required":null,"properties":null}}},{"type":"tool2","function":{"name":"func2","description":"","parameters":{"type":"","required":null,"properties":null}}}]`

    assert.JSONEq(t, expected, result)
}


// Test generated using Keploy
func TestToolFunctionStringSerialization(t *testing.T) {
    toolFunc := ToolFunction{
        Name:        "exampleFunction",
        Description: "This is an example function",
        Parameters: struct {
            Type       string   `json:"type"`
            Required   []string `json:"required"`
            Properties map[string]struct {
                Type        string   `json:"type"`
                Description string   `json:"description"`
                Enum        []string `json:"enum,omitempty"`
            } `json:"properties"`
        }{
            Type:     "object",
            Required: []string{"param1"},
            Properties: map[string]struct {
                Type        string   `json:"type"`
                Description string   `json:"description"`
                Enum        []string `json:"enum,omitempty"`
            }{
                "param1": {
                    Type:        "string",
                    Description: "A required parameter",
                },
            },
        },
    }

    result := toolFunc.String()
    expected := `{"name":"exampleFunction","description":"This is an example function","parameters":{"type":"object","required":["param1"],"properties":{"param1":{"type":"string","description":"A required parameter"}}}}`

    assert.JSONEq(t, expected, result)
}

