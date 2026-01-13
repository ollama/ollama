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

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests, order not preserved)
func testPropsMap(m map[string]ToolProperty) *ToolPropertiesMap {
	props := NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
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
