package parsers

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestParseTypedToolValueNumberRange(t *testing.T) {
	for _, tt := range []struct {
		raw  string
		kind string
		want any
	}{
		{"1e20", "number", float64(1e20)},
		{"-1e20", "number", float64(-1e20)},
		{"1e308", "number", float64(1e308)},
		{"9223372036854775808", "number", float64(9223372036854775808)},
		{"-9223372036854777856", "number", float64(-9223372036854777856)},
		{"9223372036854774784", "number", int64(9223372036854774784)},
		{"-9223372036854775808", "number", int64(-9223372036854775808)},
		{"1e3", "number", int(1000)},
		{"1.5", "number", float64(1.5)},
		{"9223372036854775807", "integer", int64(9223372036854775807)},
		{"9223372036854775808", "integer", "9223372036854775808"},
	} {
		t.Run(tt.kind+"/"+tt.raw, func(t *testing.T) {
			got := parseTypedToolValue(tt.raw, api.PropertyType{tt.kind})
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v (%T), want %v (%T)", got, got, tt.want, tt.want)
			}
		})
	}
}

func TestToolCallLargeNumber(t *testing.T) {
	for _, tt := range []struct {
		name   string
		parser Parser
		input  string
	}{
		{"qwen3-coder", &Qwen3CoderParser{}, "<tool_call><function=calculate><parameter=value>1e20</parameter></function></tool_call>"},
		{"qwen3.5", &Qwen35Parser{}, "<tool_call><function=calculate><parameter=value>1e20</parameter></function></tool_call>"},
		{"glm-4.6", &GLM46Parser{}, "<think></think><tool_call>calculate<arg_key>value</arg_key><arg_value>1e20</arg_value></tool_call>"},
		{"laguna", &LagunaParser{}, "<tool_call>calculate\n<arg_key>value</arg_key>\n<arg_value>1e20</arg_value>\n</tool_call>"},
		{"glimmer", &GlimmerParser{}, ` to=calculate<|message|><atem:function_calls><atem:invoke name="calculate"><atem:parameter name="value">1e20</atem:parameter></atem:invoke></atem:function_calls><|eot|>`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tt.parser.Init([]api.Tool{tool("calculate", map[string]api.ToolProperty{
				"value": {Type: api.PropertyType{"number"}},
			})}, nil, &api.ThinkValue{Value: false})
			var calls []api.ToolCall
			for i, chunk := range []string{tt.input[:len(tt.input)/2], tt.input[len(tt.input)/2:]} {
				_, _, got, err := tt.parser.Add(chunk, i == 1)
				if err != nil {
					t.Fatal(err)
				}
				calls = append(calls, got...)
			}
			if len(calls) != 1 {
				t.Fatalf("got %d tool calls, want 1", len(calls))
			}
			data, err := json.Marshal(calls[0].Function.Arguments)
			if err != nil {
				t.Fatal(err)
			}
			var args map[string]float64
			if err := json.Unmarshal(data, &args); err != nil {
				t.Fatal(err)
			}
			if args["value"] != 1e20 {
				t.Fatalf("serialized tool arguments = %s, want value 1e20", data)
			}
		})
	}
}
