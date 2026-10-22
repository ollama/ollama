package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestGemma4ParserTrailingNoise(t *testing.T) {
	tests := []struct {
		name, payload, arguments string
	}{
		{
			name:      "nested arguments",
			payload:   `call:ask_user{message:<|"|>Which cloud?<|"|>,requestedSchema:{properties:{cloud_provider:{description:<|"|>Provider {aws|gcp}<|"|>,type:<|"|>string<|"|>}}}}$$`,
			arguments: `{"message":"Which cloud?","requestedSchema":{"properties":{"cloud_provider":{"description":"Provider {aws|gcp}","type":"string"}}}}`,
		},
		{
			name:      "literal braces and noise inside strings",
			payload:   `call:ask_user{message:<|"|>literal $$ } { [ ] "quoted"<|"|>,requestedSchema:{description:"a } and \"quoted\" [ value"}}$$`,
			arguments: `{"message":"literal $$ } { [ ] \"quoted\"","requestedSchema":{"description":"a } and \"quoted\" [ value"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wantCalls := []api.ToolCall{{Function: api.ToolCallFunction{Name: "ask_user", Arguments: args(tt.arguments)}}}
			for _, closing := range []string{"", "<tool_call|>After"} {
				input := "Before<|tool_call>" + tt.payload + closing
				wantContent := "Before"
				if closing != "" {
					wantContent += "After"
				}
				check := func(chunks []string) {
					t.Helper()
					parser := ParserForName("gemma4")
					parser.Init(nil, nil, nil)
					var content strings.Builder
					var calls []api.ToolCall
					for i, chunk := range chunks {
						text, thinking, got, err := parser.Add(chunk, i == len(chunks)-1)
						if err != nil || thinking != "" {
							t.Fatalf("Add(%q): thinking=%q, err=%v", chunk, thinking, err)
						}
						content.WriteString(text)
						calls = append(calls, got...)
					}
					if content.String() != wantContent {
						t.Fatalf("content=%q, want %q", content.String(), wantContent)
					}
					if diff := cmp.Diff(wantCalls, calls, argsComparer); diff != "" {
						t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
					}
				}

				check([]string{input})
				for split := 0; split <= len(input); split++ {
					check([]string{input[:split], input[split:]})
				}
				chunks := make([]string, 0, len(input)+1)
				for i := range len(input) {
					chunks = append(chunks, input[i:i+1])
				}
				check(append(chunks, ""))
			}
		})
	}
}

func TestGemma4ToolCallTrailingNoiseRejectsMalformed(t *testing.T) {
	for _, payload := range []string{
		`call:ask_user{message:<|"|>hi<|"|>,requestedSchema:{properties:{name:{type:<|"|>string<|"|>}}}$$`,
		`call:ask_user{requestedSchema:{value:???}}$$`,
		`call:ask_user{message:"bad \x escape"}$$`,
		`call:ask_user{message:<|"|>hi<|"|>} {other:1}`,
		`call:ask_user{message:<|"|>hi<|"|>} {other:`,
		`call:ask_user{message:<|"|>hi<|"|>} []`,
		`call:ask_user{message:<|"|>hi<|"|>} 42`,
		`call:ask_user{message:<|"|>hi<|"|>} 1e10000`,
		`call:ask_user{message:<|"|>hi<|"|>} true`,
		`call:ask_user{message:<|"|>hi<|"|>},other:1`,
	} {
		t.Run(payload, func(t *testing.T) {
			if call, err := parseGemma4ToolCall(payload, nil); err == nil {
				t.Fatalf("expected malformed call to be rejected, got %v", call)
			}
		})
	}
}

func TestGemma4TrailingNoiseDoesNotStartToolCall(t *testing.T) {
	input := `Example: call:ask_user{message:<|"|>hi<|"|>}$$`
	parser := ParserForName("gemma4")
	parser.Init(nil, nil, nil)
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil || content != input || thinking != "" || len(calls) != 0 {
		t.Fatalf("got content=%q, thinking=%q, calls=%v, err=%v", content, thinking, calls, err)
	}
}
