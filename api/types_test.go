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
			name: "Unset",
			req:  `{ }`,
			exp:  nil,
		},
		{
			name: "Positive Integer",
			req:  `{ "keep_alive": 42 }`,
			exp:  &Duration{42 * time.Second},
		},
		{
			name: "Positive Float",
			req:  `{ "keep_alive": 42.5 }`,
			exp:  &Duration{42500 * time.Millisecond},
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
			var oMap map[string]any
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

func TestToolFunction_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr string
	}{
		{
			name: "valid enum with same types",
			input: `{
				"name": "test",
				"description": "test function",
				"parameters": {
					"type": "object",
					"required": ["test"],
					"properties": {
						"test": {
							"type": "string",
							"description": "test prop",
							"enum": ["a", "b", "c"]
						}
					}
				}
			}`,
			wantErr: "",
		},
		{
			name: "empty enum array",
			input: `{
				"name": "test",
				"description": "test function",
				"parameters": {
					"type": "object",
					"required": ["test"],
					"properties": {
						"test": {
							"type": "string",
							"description": "test prop",
							"enum": []
						}
					}
				}
			}`,
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tf ToolFunction
			err := json.Unmarshal([]byte(tt.input), &tf)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestToolFunctionParameters_MarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    ToolFunctionParameters
		expected string
	}{
		{
			name: "simple object with string property",
			input: ToolFunctionParameters{
				Type:     "object",
				Required: []string{"name"},
				Properties: map[string]ToolProperty{
					"name": {Type: PropertyType{"string"}},
				},
			},
			expected: `{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}`,
		},
		{
			name: "no required",
			input: ToolFunctionParameters{
				Type: "object",
				Properties: map[string]ToolProperty{
					"name": {Type: PropertyType{"string"}},
				},
			},
			expected: `{"type":"object","properties":{"name":{"type":"string"}}}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data, err := json.Marshal(test.input)
			require.NoError(t, err)
			assert.Equal(t, test.expected, string(data))
		})
	}
}

func TestToolCallFunction_IndexAlwaysMarshals(t *testing.T) {
	fn := ToolCallFunction{
		Name:      "echo",
		Arguments: ToolCallFunctionArguments{"message": "hi"},
	}

	data, err := json.Marshal(fn)
	require.NoError(t, err)

	raw := map[string]any{}
	require.NoError(t, json.Unmarshal(data, &raw))
	require.Contains(t, raw, "index")
	assert.Equal(t, float64(0), raw["index"])

	fn.Index = 3
	data, err = json.Marshal(fn)
	require.NoError(t, err)

	raw = map[string]any{}
	require.NoError(t, json.Unmarshal(data, &raw))
	require.Contains(t, raw, "index")
	assert.Equal(t, float64(3), raw["index"])
}

func TestPropertyType_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected PropertyType
	}{
		{
			name:     "string type",
			input:    `"string"`,
			expected: PropertyType{"string"},
		},
		{
			name:     "array of types",
			input:    `["string", "number"]`,
			expected: PropertyType{"string", "number"},
		},
		{
			name:     "array with single type",
			input:    `["string"]`,
			expected: PropertyType{"string"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var pt PropertyType
			if err := json.Unmarshal([]byte(test.input), &pt); err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if len(pt) != len(test.expected) {
				t.Errorf("Length mismatch: got %v, expected %v", len(pt), len(test.expected))
			}

			for i, v := range pt {
				if v != test.expected[i] {
					t.Errorf("Value mismatch at index %d: got %v, expected %v", i, v, test.expected[i])
				}
			}
		})
	}
}

func TestPropertyType_MarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    PropertyType
		expected string
	}{
		{
			name:     "single type",
			input:    PropertyType{"string"},
			expected: `"string"`,
		},
		{
			name:     "multiple types",
			input:    PropertyType{"string", "number"},
			expected: `["string","number"]`,
		},
		{
			name:     "empty type",
			input:    PropertyType{},
			expected: `[]`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data, err := json.Marshal(test.input)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if string(data) != test.expected {
				t.Errorf("Marshaled data mismatch: got %v, expected %v", string(data), test.expected)
			}
		})
	}
}

func TestThinking_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedThinking *ThinkValue
		expectedError    bool
	}{
		{
			name:             "true",
			input:            `{ "think": true }`,
			expectedThinking: &ThinkValue{Value: true},
		},
		{
			name:             "false",
			input:            `{ "think": false }`,
			expectedThinking: &ThinkValue{Value: false},
		},
		{
			name:             "unset",
			input:            `{ }`,
			expectedThinking: nil,
		},
		{
			name:             "string_high",
			input:            `{ "think": "high" }`,
			expectedThinking: &ThinkValue{Value: "high"},
		},
		{
			name:             "string_medium",
			input:            `{ "think": "medium" }`,
			expectedThinking: &ThinkValue{Value: "medium"},
		},
		{
			name:             "string_low",
			input:            `{ "think": "low" }`,
			expectedThinking: &ThinkValue{Value: "low"},
		},
		{
			name:             "invalid_string",
			input:            `{ "think": "invalid" }`,
			expectedThinking: nil,
			expectedError:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var req GenerateRequest
			err := json.Unmarshal([]byte(test.input), &req)
			if test.expectedError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				if test.expectedThinking == nil {
					assert.Nil(t, req.Think)
				} else {
					require.NotNil(t, req.Think)
					assert.Equal(t, test.expectedThinking.Value, req.Think.Value)
				}
			}
		})
	}
}

func TestToolPropertyNestedProperties(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected ToolProperty
	}{
		{
			name: "nested object properties",
			input: `{
				"type": "object",
				"description": "Location details",
				"properties": {
					"address": {
						"type": "string",
						"description": "Street address"
					},
					"city": {
						"type": "string",
						"description": "City name"
					}
				}
			}`,
			expected: ToolProperty{
				Type:        PropertyType{"object"},
				Description: "Location details",
				Properties: map[string]ToolProperty{
					"address": {
						Type:        PropertyType{"string"},
						Description: "Street address",
					},
					"city": {
						Type:        PropertyType{"string"},
						Description: "City name",
					},
				},
			},
		},
		{
			name: "deeply nested properties",
			input: `{
				"type": "object",
				"description": "Event",
				"properties": {
					"location": {
						"type": "object",
						"description": "Location",
						"properties": {
							"coordinates": {
								"type": "object",
								"description": "GPS coordinates",
								"properties": {
									"lat": {"type": "number", "description": "Latitude"},
									"lng": {"type": "number", "description": "Longitude"}
								}
							}
						}
					}
				}
			}`,
			expected: ToolProperty{
				Type:        PropertyType{"object"},
				Description: "Event",
				Properties: map[string]ToolProperty{
					"location": {
						Type:        PropertyType{"object"},
						Description: "Location",
						Properties: map[string]ToolProperty{
							"coordinates": {
								Type:        PropertyType{"object"},
								Description: "GPS coordinates",
								Properties: map[string]ToolProperty{
									"lat": {Type: PropertyType{"number"}, Description: "Latitude"},
									"lng": {Type: PropertyType{"number"}, Description: "Longitude"},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var prop ToolProperty
			err := json.Unmarshal([]byte(tt.input), &prop)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, prop)

			// Round-trip test: marshal and unmarshal again
			data, err := json.Marshal(prop)
			require.NoError(t, err)

			var prop2 ToolProperty
			err = json.Unmarshal(data, &prop2)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, prop2)
		})
	}
}

func TestToolFunctionParameters_String(t *testing.T) {
	tests := []struct {
		name     string
		params   ToolFunctionParameters
		expected string
	}{
		{
			name: "simple object with string property",
			params: ToolFunctionParameters{
				Type:     "object",
				Required: []string{"name"},
				Properties: map[string]ToolProperty{
					"name": {
						Type:        PropertyType{"string"},
						Description: "The name of the person",
					},
				},
			},
			expected: `{"type":"object","required":["name"],"properties":{"name":{"type":"string","description":"The name of the person"}}}`,
		},
		{
			name: "marshal failure returns empty string",
			params: ToolFunctionParameters{
				Type: "object",
				Defs: func() any {
					// Create a cycle that will cause json.Marshal to fail
					type selfRef struct {
						Self *selfRef
					}
					s := &selfRef{}
					s.Self = s
					return s
				}(),
				Properties: map[string]ToolProperty{},
			},
			expected: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.params.String()
			assert.Equal(t, test.expected, result)
		})
	}
}
