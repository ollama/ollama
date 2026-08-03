package api

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests, order not preserved)
func testPropsMap(m map[string]ToolProperty) *ToolPropertiesMap {
	props := NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

func testIntPtr(v int) *int {
	return &v
}

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests, order not preserved)
func testArgs(m map[string]any) ToolCallFunctionArguments {
	args := NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

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

func TestMainGPUParsingFromJSON(t *testing.T) {
	tests := []struct {
		name    string
		req     string
		wantGPU *int
	}{
		{
			name: "Undefined",
			req:  `{}`,
		},
		{
			name:    "Zero",
			req:     `{ "main_gpu": 0 }`,
			wantGPU: testIntPtr(0),
		},
		{
			name:    "Nonzero",
			req:     `{ "main_gpu": 1 }`,
			wantGPU: testIntPtr(1),
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

			if test.wantGPU == nil {
				assert.Nil(t, opts.MainGPU)
			} else if assert.NotNil(t, opts.MainGPU) {
				assert.Equal(t, *test.wantGPU, *opts.MainGPU)
			}
		})
	}
}

func TestGenerationDefaultMappingsAreOptions(t *testing.T) {
	jsonOpts := make(map[string]struct{})
	for _, field := range reflect.VisibleFields(reflect.TypeOf(Options{})) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = struct{}{}
		}
	}

	for _, option := range model.GenerationDefaultOptions() {
		if _, ok := jsonOpts[option]; !ok {
			t.Fatalf("%s should be defined on api.Options", option)
		}
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

func TestMainGPUFormatParams(t *testing.T) {
	resp, err := FormatParams(map[string][]string{"main_gpu": {"0"}})
	require.NoError(t, err)
	assert.Equal(t, int64(0), resp["main_gpu"])
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
				Properties: testPropsMap(map[string]ToolProperty{
					"name": {Type: PropertyType{"string"}},
				}),
			},
			expected: `{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}`,
		},
		{
			name: "no required",
			input: ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]ToolProperty{
					"name": {Type: PropertyType{"string"}},
				}),
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
		Arguments: testArgs(map[string]any{"message": "hi"}),
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
			name:             "string_max",
			input:            `{ "think": "max" }`,
			expectedThinking: &ThinkValue{Value: "max"},
		},
		{
			name:             "invalid_string",
			input:            `{ "think": "invalid" }`,
			expectedThinking: nil,
			expectedError:    true,
		},
		{
			name:             "budget",
			input:            `{ "think": 8192 }`,
			expectedThinking: &ThinkValue{Value: 8192},
		},
		{
			name:             "zero_budget",
			input:            `{ "think": 0 }`,
			expectedThinking: nil,
			expectedError:    true,
		},
		{
			name:             "negative_budget",
			input:            `{ "think": -1 }`,
			expectedThinking: nil,
			expectedError:    true,
		},
		{
			name:             "fractional_budget",
			input:            `{ "think": 1.5 }`,
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
				Properties: testPropsMap(map[string]ToolProperty{
					"address": {
						Type:        PropertyType{"string"},
						Description: "Street address",
					},
					"city": {
						Type:        PropertyType{"string"},
						Description: "City name",
					},
				}),
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
				Properties: testPropsMap(map[string]ToolProperty{
					"location": {
						Type:        PropertyType{"object"},
						Description: "Location",
						Properties: testPropsMap(map[string]ToolProperty{
							"coordinates": {
								Type:        PropertyType{"object"},
								Description: "GPS coordinates",
								Properties: testPropsMap(map[string]ToolProperty{
									"lat": {Type: PropertyType{"number"}, Description: "Latitude"},
									"lng": {Type: PropertyType{"number"}, Description: "Longitude"},
								}),
							},
						}),
					},
				}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var prop ToolProperty
			err := json.Unmarshal([]byte(tt.input), &prop)
			require.NoError(t, err)

			// Compare JSON representations since pointer comparison doesn't work
			expectedJSON, err := json.Marshal(tt.expected)
			require.NoError(t, err)
			actualJSON, err := json.Marshal(prop)
			require.NoError(t, err)
			assert.JSONEq(t, string(expectedJSON), string(actualJSON))

			// Round-trip test: marshal and unmarshal again
			data, err := json.Marshal(prop)
			require.NoError(t, err)

			var prop2 ToolProperty
			err = json.Unmarshal(data, &prop2)
			require.NoError(t, err)

			prop2JSON, err := json.Marshal(prop2)
			require.NoError(t, err)
			assert.JSONEq(t, string(expectedJSON), string(prop2JSON))
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
				Properties: testPropsMap(map[string]ToolProperty{
					"name": {
						Type:        PropertyType{"string"},
						Description: "The name of the person",
					},
				}),
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
				Properties: testPropsMap(map[string]ToolProperty{}),
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

func TestToolCallFunctionArguments_OrderPreservation(t *testing.T) {
	t.Run("marshal preserves insertion order", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		args.Set("zebra", "z")
		args.Set("apple", "a")
		args.Set("mango", "m")

		data, err := json.Marshal(args)
		require.NoError(t, err)

		// Should preserve insertion order, not alphabetical
		assert.Equal(t, `{"zebra":"z","apple":"a","mango":"m"}`, string(data))
	})

	t.Run("unmarshal preserves JSON order", func(t *testing.T) {
		jsonData := `{"zebra":"z","apple":"a","mango":"m"}`

		var args ToolCallFunctionArguments
		err := json.Unmarshal([]byte(jsonData), &args)
		require.NoError(t, err)

		// Verify iteration order matches JSON order
		var keys []string
		for k := range args.All() {
			keys = append(keys, k)
		}
		assert.Equal(t, []string{"zebra", "apple", "mango"}, keys)
	})

	t.Run("round trip preserves order", func(t *testing.T) {
		original := `{"z":1,"a":2,"m":3,"b":4}`

		var args ToolCallFunctionArguments
		err := json.Unmarshal([]byte(original), &args)
		require.NoError(t, err)

		data, err := json.Marshal(args)
		require.NoError(t, err)

		assert.Equal(t, original, string(data))
	})

	t.Run("String method returns ordered JSON", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		args.Set("c", 3)
		args.Set("a", 1)
		args.Set("b", 2)

		assert.Equal(t, `{"c":3,"a":1,"b":2}`, args.String())
	})

	t.Run("Get retrieves correct values", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		args.Set("key1", "value1")
		args.Set("key2", 42)

		v, ok := args.Get("key1")
		assert.True(t, ok)
		assert.Equal(t, "value1", v)

		v, ok = args.Get("key2")
		assert.True(t, ok)
		assert.Equal(t, 42, v)

		_, ok = args.Get("nonexistent")
		assert.False(t, ok)
	})

	t.Run("Len returns correct count", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		assert.Equal(t, 0, args.Len())

		args.Set("a", 1)
		assert.Equal(t, 1, args.Len())

		args.Set("b", 2)
		assert.Equal(t, 2, args.Len())
	})

	t.Run("empty args marshal to empty object", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		data, err := json.Marshal(args)
		require.NoError(t, err)
		assert.Equal(t, `{}`, string(data))
	})

	t.Run("zero value args marshal to empty object", func(t *testing.T) {
		var args ToolCallFunctionArguments
		assert.Equal(t, "{}", args.String())
	})
}

func TestToolPropertiesMap_OrderPreservation(t *testing.T) {
	t.Run("marshal preserves insertion order", func(t *testing.T) {
		props := NewToolPropertiesMap()
		props.Set("zebra", ToolProperty{Type: PropertyType{"string"}})
		props.Set("apple", ToolProperty{Type: PropertyType{"number"}})
		props.Set("mango", ToolProperty{Type: PropertyType{"boolean"}})

		data, err := json.Marshal(props)
		require.NoError(t, err)

		// Should preserve insertion order, not alphabetical
		expected := `{"zebra":{"type":"string"},"apple":{"type":"number"},"mango":{"type":"boolean"}}`
		assert.Equal(t, expected, string(data))
	})

	t.Run("unmarshal preserves JSON order", func(t *testing.T) {
		jsonData := `{"zebra":{"type":"string"},"apple":{"type":"number"},"mango":{"type":"boolean"}}`

		var props ToolPropertiesMap
		err := json.Unmarshal([]byte(jsonData), &props)
		require.NoError(t, err)

		// Verify iteration order matches JSON order
		var keys []string
		for k := range props.All() {
			keys = append(keys, k)
		}
		assert.Equal(t, []string{"zebra", "apple", "mango"}, keys)
	})

	t.Run("round trip preserves order", func(t *testing.T) {
		original := `{"z":{"type":"string"},"a":{"type":"number"},"m":{"type":"boolean"}}`

		var props ToolPropertiesMap
		err := json.Unmarshal([]byte(original), &props)
		require.NoError(t, err)

		data, err := json.Marshal(props)
		require.NoError(t, err)

		assert.Equal(t, original, string(data))
	})

	t.Run("Get retrieves correct values", func(t *testing.T) {
		props := NewToolPropertiesMap()
		props.Set("name", ToolProperty{Type: PropertyType{"string"}, Description: "The name"})
		props.Set("age", ToolProperty{Type: PropertyType{"integer"}, Description: "The age"})

		v, ok := props.Get("name")
		assert.True(t, ok)
		assert.Equal(t, "The name", v.Description)

		v, ok = props.Get("age")
		assert.True(t, ok)
		assert.Equal(t, "The age", v.Description)

		_, ok = props.Get("nonexistent")
		assert.False(t, ok)
	})

	t.Run("Len returns correct count", func(t *testing.T) {
		props := NewToolPropertiesMap()
		assert.Equal(t, 0, props.Len())

		props.Set("a", ToolProperty{})
		assert.Equal(t, 1, props.Len())

		props.Set("b", ToolProperty{})
		assert.Equal(t, 2, props.Len())
	})

	t.Run("nil props marshal to null", func(t *testing.T) {
		var props *ToolPropertiesMap
		data, err := json.Marshal(props)
		require.NoError(t, err)
		assert.Equal(t, `null`, string(data))
	})

	t.Run("ToMap returns regular map", func(t *testing.T) {
		props := NewToolPropertiesMap()
		props.Set("a", ToolProperty{Type: PropertyType{"string"}})
		props.Set("b", ToolProperty{Type: PropertyType{"number"}})

		m := props.ToMap()
		assert.Equal(t, 2, len(m))
		assert.Equal(t, PropertyType{"string"}, m["a"].Type)
		assert.Equal(t, PropertyType{"number"}, m["b"].Type)
	})
}

func TestToolCallFunctionArguments_ComplexValues(t *testing.T) {
	t.Run("nested objects preserve order", func(t *testing.T) {
		jsonData := `{"outer":{"z":1,"a":2},"simple":"value"}`

		var args ToolCallFunctionArguments
		err := json.Unmarshal([]byte(jsonData), &args)
		require.NoError(t, err)

		// Outer keys should be in order
		var keys []string
		for k := range args.All() {
			keys = append(keys, k)
		}
		assert.Equal(t, []string{"outer", "simple"}, keys)
	})

	t.Run("arrays as values", func(t *testing.T) {
		args := NewToolCallFunctionArguments()
		args.Set("items", []string{"a", "b", "c"})
		args.Set("numbers", []int{1, 2, 3})

		data, err := json.Marshal(args)
		require.NoError(t, err)

		assert.Equal(t, `{"items":["a","b","c"],"numbers":[1,2,3]}`, string(data))
	})
}

func TestToolPropertiesMap_NestedProperties(t *testing.T) {
	t.Run("nested properties preserve order", func(t *testing.T) {
		props := NewToolPropertiesMap()

		nestedProps := NewToolPropertiesMap()
		nestedProps.Set("z_field", ToolProperty{Type: PropertyType{"string"}})
		nestedProps.Set("a_field", ToolProperty{Type: PropertyType{"number"}})

		props.Set("outer", ToolProperty{
			Type:       PropertyType{"object"},
			Properties: nestedProps,
		})

		data, err := json.Marshal(props)
		require.NoError(t, err)

		// Both outer and inner should preserve order
		expected := `{"outer":{"type":"object","properties":{"z_field":{"type":"string"},"a_field":{"type":"number"}}}}`
		assert.Equal(t, expected, string(data))
	})
}

func TestThinkValueBudgetTokens(t *testing.T) {
	tests := []struct {
		name     string
		think    *ThinkValue
		numCtx   int
		expected int
	}{
		{name: "unset", think: nil, numCtx: 32768, expected: 0},
		{name: "true is unrestricted", think: &ThinkValue{Value: true}, numCtx: 32768, expected: 0},
		{name: "false is unrestricted", think: &ThinkValue{Value: false}, numCtx: 32768, expected: 0},
		{name: "explicit budget", think: &ThinkValue{Value: 8192}, numCtx: 32768, expected: 8192},
		{name: "explicit budget ignores context", think: &ThinkValue{Value: 8192}, numCtx: 0, expected: 8192},
		{name: "max is four fifths", think: &ThinkValue{Value: "max"}, numCtx: 32768, expected: 26214},
		{name: "high is one half", think: &ThinkValue{Value: "high"}, numCtx: 32768, expected: 16384},
		{name: "medium is one quarter", think: &ThinkValue{Value: "medium"}, numCtx: 32768, expected: 8192},
		{name: "low is one eighth", think: &ThinkValue{Value: "low"}, numCtx: 32768, expected: 4096},
		{name: "minimal is one sixteenth", think: &ThinkValue{Value: "minimal"}, numCtx: 32768, expected: 2048},
		{name: "effort without context", think: &ThinkValue{Value: "high"}, numCtx: 0, expected: 0},
		{name: "effort with tiny context", think: &ThinkValue{Value: "low"}, numCtx: 2, expected: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.think.BudgetTokens(tt.numCtx))
		})
	}
}

func TestThinkValueIsValid(t *testing.T) {
	valid := []*ThinkValue{nil, {Value: nil}, {Value: true}, {Value: false}, {Value: "max"}, {Value: "low"}, {Value: "minimal"}, {Value: 1}}
	for _, think := range valid {
		assert.True(t, think.IsValid(), "expected %v to be valid", think)
	}

	invalid := []*ThinkValue{{Value: "invalid"}, {Value: 0}, {Value: -1}, {Value: 1.5}}
	for _, think := range invalid {
		assert.False(t, think.IsValid(), "expected %v to be invalid", think)
	}
}

func TestThinkBudgetOption(t *testing.T) {
	// think_budget is set as a model default with `PARAMETER think_budget`,
	// which reaches Options through the generic parameter map. It takes either
	// a token count or an effort level.
	tests := []struct {
		name     string
		param    string // as written in a Modelfile
		fromMap  []any  // equivalent values arriving through FromMap
		expected any    // ThinkValue.Value once decoded
		budget   int    // resolved against a 32768 token context
	}{
		{
			name:     "token count",
			param:    "8192",
			fromMap:  []any{int64(8192), float64(8192)},
			expected: 8192,
			budget:   8192,
		},
		{
			name:     "effort level",
			param:    "high",
			fromMap:  []any{"high"},
			expected: "high",
			budget:   16384,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := FormatParams(map[string][]string{"think_budget": {tt.param}})
			require.NoError(t, err)

			// what FormatParams produces must itself be decodable
			values := append([]any{params["think_budget"]}, tt.fromMap...)
			for _, value := range values {
				opts := DefaultOptions()
				require.NoError(t, opts.FromMap(map[string]any{"think_budget": value}), "value %#v", value)
				require.NotNil(t, opts.ThinkBudget)
				assert.Equal(t, tt.expected, opts.ThinkBudget.Value)
				assert.Equal(t, tt.budget, opts.ThinkBudget.BudgetTokens(32768))
			}
		})
	}

	assert.Nil(t, DefaultOptions().ThinkBudget)
	assert.Equal(t, 0, DefaultOptions().ThinkBudget.BudgetTokens(32768))

	opts := DefaultOptions()
	assert.Error(t, opts.FromMap(map[string]any{"think_budget": "enormous"}))
}

func TestThinkBudgetMessageOption(t *testing.T) {
	// think_budget_message ships with a model the same way the budget does, and
	// is free text: whatever wording works for that model.
	const message = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."

	params, err := FormatParams(map[string][]string{"think_budget_message": {message}})
	require.NoError(t, err)

	for _, value := range []any{params["think_budget_message"], message} {
		opts := DefaultOptions()
		require.NoError(t, opts.FromMap(map[string]any{"think_budget_message": value}), "value %#v", value)
		assert.Equal(t, message, opts.ThinkBudgetMessage)
	}

	// unset means the bare closing tag is forced, which is what every runner
	// does today
	assert.Empty(t, DefaultOptions().ThinkBudgetMessage)
}

func TestThinkValueLevel(t *testing.T) {
	// Level is what a model that consumes effort levels directly receives.
	tests := []struct {
		name  string
		think *ThinkValue
		level string
	}{
		{name: "unset", think: nil, level: ""},
		{name: "low", think: &ThinkValue{Value: "low"}, level: "low"},
		{name: "medium", think: &ThinkValue{Value: "medium"}, level: "medium"},
		{name: "high", think: &ThinkValue{Value: "high"}, level: "high"},
		{name: "max is reported as high", think: &ThinkValue{Value: "max"}, level: "high"},
		{name: "minimal is reported as low", think: &ThinkValue{Value: "minimal"}, level: "low"},
		{name: "true", think: &ThinkValue{Value: true}, level: "medium"},
		{name: "false", think: &ThinkValue{Value: false}, level: ""},
		{name: "a budget carries no level", think: &ThinkValue{Value: 8192}, level: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.level, tt.think.Level())
		})
	}

	// Reporting a level as its nearest neighbour must not change the budget it
	// asked for, or the requested level itself
	for _, tt := range []struct {
		requested string
		reported  string
		budget    int
	}{
		{requested: "max", reported: "high", budget: 26214},
		{requested: "minimal", reported: "low", budget: 2048},
	} {
		think := &ThinkValue{Value: tt.requested}
		assert.Equal(t, tt.reported, think.Level())
		assert.Equal(t, tt.budget, think.BudgetTokens(32768))
		assert.Equal(t, tt.requested, think.String(), "the requested level is preserved")
	}
}

func TestThinkLevelsAreIndependentOfBudgets(t *testing.T) {
	// A level a model understands does not have to carry a budget. Tying the
	// two together would reject any level that exists only to be handed to the
	// model.
	for _, level := range thinkLevels {
		think := &ThinkValue{Value: level}
		assert.True(t, think.IsValid(), level)
		assert.True(t, think.Bool(), level)
		assert.Equal(t, level, think.String(), level)
	}

	for level := range thinkBudgetFraction {
		assert.Contains(t, thinkLevels, level, "budget fraction for an unknown level")
	}

	thinkLevels = append(thinkLevels, "exhaustive")
	t.Cleanup(func() { thinkLevels = thinkLevels[:len(thinkLevels)-1] })

	budgetless := &ThinkValue{Value: "exhaustive"}
	assert.True(t, budgetless.IsValid())
	assert.True(t, budgetless.Bool())
	assert.Equal(t, "exhaustive", budgetless.Level(), "the level still reaches the model")
	assert.Equal(t, 0, budgetless.BudgetTokens(32768), "and simply carries no budget")
}

func TestThinkBudgetWindow(t *testing.T) {
	tests := []struct {
		name       string
		numCtx     int
		numPredict int
		want       int
	}{
		{name: "no cap on the response", numCtx: 128000, numPredict: -1, want: 128000},
		{name: "unset num_predict", numCtx: 128000, numPredict: 0, want: 128000},
		// the case this exists for: a level resolved against the context would
		// have matched num_predict exactly and bounded nothing
		{name: "response capped below the context", numCtx: 128000, numPredict: 32000, want: 32000},
		{name: "response cap above the context", numCtx: 8192, numPredict: 32000, want: 8192},
		{name: "no context length known", numCtx: 0, numPredict: 4096, want: 4096},
		{name: "neither known", numCtx: 0, numPredict: 0, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ThinkBudgetWindow(tt.numCtx, tt.numPredict); got != tt.want {
				t.Errorf("ThinkBudgetWindow(%d, %d) = %d, want %d", tt.numCtx, tt.numPredict, got, tt.want)
			}
		})
	}
}

func TestThinkLevelSharesTheResponseNotTheContext(t *testing.T) {
	// Cline sends num_ctx 128000 with num_predict 32000, where "medium" as a
	// quarter of the context is exactly the response cap: the model can spend
	// every token it is allowed to emit inside the thinking block and stop with
	// no answer at all.
	const numCtx, numPredict = 128000, 32000

	think := &ThinkValue{Value: "medium"}
	if got := think.BudgetTokens(numCtx); got != numPredict {
		t.Fatalf("premise changed: a quarter of %d is %d, not the response cap %d", numCtx, got, numPredict)
	}

	window := ThinkBudgetWindow(numCtx, numPredict)
	budget := think.BudgetTokens(window)
	if budget >= numPredict {
		t.Errorf("budget %d does not leave room to answer within num_predict %d", budget, numPredict)
	}
	if want := numPredict / 4; budget != want {
		t.Errorf("budget = %d, want %d (a quarter of the response)", budget, want)
	}
}
