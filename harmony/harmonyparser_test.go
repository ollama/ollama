package harmony

import (
	"fmt"
	"reflect"
	"testing"
)

func TestHeaderParsing(t *testing.T) {
	tests := []struct {
		in, wantRole, wantChannel, wantRecipient string
	}{
		{
			in:            "assistant<|channel|>analysis",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "",
		},
		{
			in:            "assistant<|channel|>analysis to=functions.get_weather",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "functions.get_weather",
		},
		{
			in:            "assistant to=functions.get_weather<|channel|>analysis",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "functions.get_weather",
		},
		// special case where the role is replaced by the recipient (matches reference code)
		{
			in:            "to=functions.get_weather<|channel|>analysis",
			wantRole:      "tool",
			wantChannel:   "analysis",
			wantRecipient: "functions.get_weather",
		},
		// extra token after the recipient is ignored
		{
			in:            "assistant to=functions.get_weather abc<|channel|>analysis",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "functions.get_weather",
		},
		// with constrain tag, recipient after channel tag
		{
			in:            "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
			wantRole:      "assistant",
			wantChannel:   "commentary",
			wantRecipient: "functions.get_weather",
		},
		// with constrain tag, recipient before channel tag
		{
			in:            "assistant to=functions.get_weather<|channel|>commentary <|constrain|>json",
			wantRole:      "assistant",
			wantChannel:   "commentary",
			wantRecipient: "functions.get_weather",
		},
		// constrain tag without space
		{
			in:            "assistant<|channel|>commentary to=functions.get_weather<|constrain|>json",
			wantRole:      "assistant",
			wantChannel:   "commentary",
			wantRecipient: "functions.get_weather",
		},
		// constrain tag without space, different order
		{
			in:            "assistant to=functions.get_weather<|channel|>commentary<|constrain|>json",
			wantRole:      "assistant",
			wantChannel:   "commentary",
			wantRecipient: "functions.get_weather",
		},
	}
	for i, tt := range tests {
		parser := HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		}
		header := parser.parseHeader(tt.in)

		if header.Role != tt.wantRole {
			t.Errorf("case %d: got role \"%s\", want \"%s\"", i, header.Role, tt.wantRole)
		}
		if header.Channel != tt.wantChannel {
			t.Errorf("case %d: got channel \"%s\", want \"%s\"", i, header.Channel, tt.wantChannel)
		}
		if header.Recipient != tt.wantRecipient {
			t.Errorf("case %d: got recipient \"%s\", want \"%s\"", i, header.Recipient, tt.wantRecipient)
		}
	}
}

func TestHarmonyParserHeaderEvent(t *testing.T) {
	tests := []struct {
		in, wantRole, wantChannel, wantRecipient string
		implicitStart                            bool
	}{
		{
			in:            "<|start|>user<|message|>What is 2 + 2?<|end|>",
			wantRole:      "user",
			wantChannel:   "",
			wantRecipient: "",
		},
		{
			in:            "<|start|>assistant<|channel|>analysis<|message|>What is 2 + 2?<|end|>",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "",
		},
		{
			in:            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"San Francisco\"}<|call|><|start|>functions.get_weather to=assistant<|message|>{\"sunny\": true, \"temperature\": 20}<|end|>",
			wantRole:      "assistant",
			wantChannel:   "commentary",
			wantRecipient: "functions.get_weather",
		},
		{
			in:            "<|channel|>analysis<|message|>User asks weather in SF. We need location. Use get_current_weather with location \"San Francisco, CA\".<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"location\":\"San Francisco, CA\"}<|call|>",
			wantRole:      "assistant",
			wantChannel:   "analysis",
			wantRecipient: "",
			implicitStart: true,
		},
	}
	for i, tt := range tests {
		parser := HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		}
		if tt.implicitStart {
			parser.AddImplicitStart()
		}
		gotEvents := parser.AddContent(tt.in)
		if len(gotEvents) == 0 {
			t.Errorf("case %d: got no events, want at least one", i)
		}

		var firstHeaderEvent *HarmonyEventHeaderComplete
		// print events
		for _, event := range gotEvents {
			fmt.Printf("event: %+v\n", event)
		}
		for _, event := range gotEvents {
			if event, ok := event.(HarmonyEventHeaderComplete); ok {
				firstHeaderEvent = &event
				break
			}
		}

		if firstHeaderEvent == nil {
			t.Errorf("case %d: got no header complete event, want one", i)
			continue
		}
		gotHeader := firstHeaderEvent.Header
		if gotHeader.Role != tt.wantRole || gotHeader.Channel != tt.wantChannel || gotHeader.Recipient != tt.wantRecipient {
			t.Errorf("case %d: got header %+v, want role=%s channel=%s recipient=%s", i, gotHeader, tt.wantRole, tt.wantChannel, tt.wantRecipient)
		}
	}
}

func TestHarmonyParserNonStreaming(t *testing.T) {
	tests := []struct {
		in            string
		implicitStart bool
		wantEvents    []HarmonyEvent
	}{
		{
			in: "<|start|>user<|message|>What is 2 + 2?<|end|>",
			wantEvents: []HarmonyEvent{
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
				HarmonyEventContentEmitted{Content: "What is 2 + 2?"},
				HarmonyEventMessageEnd{},
			},
		},
		{
			in: "<|start|>assistant<|channel|>analysis<|message|>The answer is 4<|end|>",
			wantEvents: []HarmonyEvent{
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "analysis", Recipient: ""}},
				HarmonyEventContentEmitted{Content: "The answer is 4"},
				HarmonyEventMessageEnd{},
			},
		},
		{
			in: "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>Computing...<|end|>",
			wantEvents: []HarmonyEvent{
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "commentary", Recipient: "functions.calc"}},
				HarmonyEventContentEmitted{Content: "Computing..."},
				HarmonyEventMessageEnd{},
			},
		},
		{
			in: "<|start|>user<|message|><|end|>",
			wantEvents: []HarmonyEvent{
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
				HarmonyEventMessageEnd{},
			},
		},
		{
			in: "<|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>Hi!<|end|>",
			wantEvents: []HarmonyEvent{
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
				HarmonyEventContentEmitted{Content: "Hello"},
				HarmonyEventMessageEnd{},
				HarmonyEventMessageStart{},
				HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "", Recipient: ""}},
				HarmonyEventContentEmitted{Content: "Hi!"},
				HarmonyEventMessageEnd{},
			},
		},
		{
			in:            "<|channel|>analysis<|message|>Thinking about the request<|end|>",
			implicitStart: true,
			wantEvents:    []HarmonyEvent{HarmonyEventMessageStart{}, HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "analysis", Recipient: ""}}, HarmonyEventContentEmitted{Content: "Thinking about the request"}, HarmonyEventMessageEnd{}},
		},
	}
	for i, tt := range tests {
		parser := HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		}
		if tt.implicitStart {
			parser.AddImplicitStart()
		}
		gotEvents := parser.AddContent(tt.in)
		if !reflect.DeepEqual(gotEvents, tt.wantEvents) {
			t.Errorf("case %d: got events %#v, want %#v", i, gotEvents, tt.wantEvents)
		}
	}
}

func TestHarmonyParserStreaming(t *testing.T) {
	type step struct {
		input      string
		wantEvents []HarmonyEvent
	}

	cases := []struct {
		desc          string
		implicitStart bool
		steps         []step
	}{
		{
			desc: "simple message streamed character by character",
			steps: []step{
				{
					input:      "<",
					wantEvents: nil,
				},
				{
					input:      "|",
					wantEvents: nil,
				},
				{
					input:      "start|>u",
					wantEvents: []HarmonyEvent{HarmonyEventMessageStart{}},
				},
				{
					input:      "ser<|mess",
					wantEvents: nil,
				},
				{
					input: "age|>Hi",
					wantEvents: []HarmonyEvent{
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
						HarmonyEventContentEmitted{Content: "Hi"},
					},
				},
				{
					input:      " there",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: " there"}},
				},
				{
					input:      "<|e",
					wantEvents: nil,
				},
				{
					input:      "nd|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc: "message with channel streamed",
			steps: []step{
				{
					input:      "<|start|>assistant",
					wantEvents: []HarmonyEvent{HarmonyEventMessageStart{}},
				},
				{
					input:      "<|chan",
					wantEvents: nil,
				},
				{
					input:      "nel|>analysis",
					wantEvents: nil,
				},
				{
					input:      "<|message|>",
					wantEvents: []HarmonyEvent{HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "analysis", Recipient: ""}}},
				},
				{
					input:      "Thinking",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "Thinking"}},
				},
				{
					input:      "...",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "..."}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc: "message with channel and recipient",
			steps: []step{
				{
					input: "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>",
					wantEvents: []HarmonyEvent{
						HarmonyEventMessageStart{},
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "commentary", Recipient: "functions.calc"}},
					},
				},
				{
					input:      "{\"x\": 5}",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "{\"x\": 5}"}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc: "message with channel and recipient (receipient before channel)",
			steps: []step{
				{
					input: "<|start|>assistant to=functions.calc<|channel|>commentary<|message|>",
					wantEvents: []HarmonyEvent{
						HarmonyEventMessageStart{},
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "commentary", Recipient: "functions.calc"}},
					},
				},
				{
					input:      "{\"x\": 5}",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "{\"x\": 5}"}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc:          "implicit start with channel",
			implicitStart: true,
			steps: []step{
				{
					input:      "<|channel|>thinking",
					wantEvents: []HarmonyEvent{HarmonyEventMessageStart{}},
				},
				{
					input:      "<|message|>",
					wantEvents: []HarmonyEvent{HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "thinking", Recipient: ""}}},
				},
				{
					input:      "Processing request",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "Processing request"}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc: "multiple messages streamed",
			steps: []step{
				{
					input: "<|start|>user<|message|>Hello<|end|>",
					wantEvents: []HarmonyEvent{
						HarmonyEventMessageStart{},
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
						HarmonyEventContentEmitted{Content: "Hello"},
						HarmonyEventMessageEnd{},
					},
				},
				{
					input:      "<|start|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageStart{}},
				},
				{
					input:      "assistant<|message|>",
					wantEvents: []HarmonyEvent{HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "assistant", Channel: "", Recipient: ""}}},
				},
				{
					input:      "Hi!",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "Hi!"}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
		{
			desc: "empty message",
			steps: []step{
				{
					input: "<|start|>system<|message|><|end|>",
					wantEvents: []HarmonyEvent{
						HarmonyEventMessageStart{},
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "system", Channel: "", Recipient: ""}},
						HarmonyEventMessageEnd{},
					},
				},
			},
		},
		{
			desc: "partial tag that looks like end but isn't",
			steps: []step{
				{
					input: "<|start|>user<|message|>test<|e",
					wantEvents: []HarmonyEvent{
						HarmonyEventMessageStart{},
						HarmonyEventHeaderComplete{Header: HarmonyHeader{Role: "user", Channel: "", Recipient: ""}},
						HarmonyEventContentEmitted{Content: "test"},
					},
				},
				{
					input:      "xample|>more",
					wantEvents: []HarmonyEvent{HarmonyEventContentEmitted{Content: "<|example|>more"}},
				},
				{
					input:      "<|end|>",
					wantEvents: []HarmonyEvent{HarmonyEventMessageEnd{}},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := HarmonyParser{
				MessageStartTag: "<|start|>",
				MessageEndTag:   "<|end|>",
				HeaderEndTag:    "<|message|>",
			}
			if tc.implicitStart {
				parser.AddImplicitStart()
			}

			for i, step := range tc.steps {
				gotEvents := parser.AddContent(step.input)
				if !reflect.DeepEqual(gotEvents, step.wantEvents) {
					t.Errorf("step %d: input %q: got events %#v, want %#v", i, step.input, gotEvents, step.wantEvents)
				}
			}
		})
	}
}

// TestFunctionConvertToValidChars tests only FunctionNameMap.convert(), which doesn't
// handle any saving (and therefore no dupe handling)
func TestFunctionConvertToValidChars(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "replace spaces with underscores", in: "get weather", want: "get_weather"},
		{name: "replace hyphens with underscores", in: "get-weather", want: "get_weather"},
		{name: "replace periods with underscores", in: "get.weather", want: "get_weather"},
		{name: "disallow non-word characters", in: "get weather!", want: "get_weather"},
		{name: "strip out invalid non-alphanumeric unicode characters", in: "a🫠bc", want: "abc"},
		{name: "names that only contain invalid characters", in: "🫠", want: "unnamed"},
		{name: "leading number", in: "123", want: "_123"},
		{name: "$ allowed", in: "$", want: "$"},
		// show that we allow weird unicode letter characters, though we might want
		// to convert them to their closest ASCII equivalents in the future
		{name: "allow weird unicode letter characters", in: "𝓸𝓵𝓵𝓪𝓶𝓪", want: "𝓸𝓵𝓵𝓪𝓶𝓪"},
		// names that look like words but are invalid (i.e., not ID_Start/ID_Continue)
		{name: "disallow non-word characters that look like words", in: "ⓞⓛⓛⓐⓜⓐ123", want: "_123"},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewFunctionNameMap()
			got := parser.convertToValidChars(tt.in)
			if got != tt.want {
				t.Errorf("case %d: got %q, want %q", i, got, tt.want)
			}
		})
	}
}

func TestFunctionConvertAndAdd(t *testing.T) {
	// make a fresh map for each test, but within a test use the same map so we can test for dupe handling
	tests := []struct {
		name string
		in   []string
		want []string
	}{
		{name: "basic dupe handling", in: []string{"get weather", "get weather"}, want: []string{"get_weather", "get_weather_2"}},
		{name: "dupes from different user-specified names", in: []string{"get weather", "get_weather", "get-weather"}, want: []string{"get_weather", "get_weather_2", "get_weather_3"}},
		{name: "non dupes after dupes", in: []string{"get weather", "get_weather", "get-weather", "something-different"}, want: []string{"get_weather", "get_weather_2", "get_weather_3", "something_different"}},
		{name: "multiple sets of dupes", in: []string{"a", "a", "b", "a", "a", "b", "a"}, want: []string{"a", "a_2", "b", "a_3", "a_4", "b_2", "a_5"}},
		{name: "built-in functions should not be renamed", in: []string{"browser.open", "python", "not.a.built-in.function", "browser.not_a_real_built_in"}, want: []string{"browser.open", "python", "not_a_built_in_function", "browser_not_a_real_built_in"}},
	}

	for i, tt := range tests {
		parser := NewFunctionNameMap()
		t.Run(tt.name, func(t *testing.T) {
			for j, in := range tt.in {
				got := parser.ConvertAndAdd(in)
				want := tt.want[j]
				if got != want {
					t.Errorf("case %d: got %q, want %q", i, got, want)
				}
				// check that the maps are correct
				if parser.userToHarmony[in] != want {
					t.Errorf("case %d: userToHarmony[%q] = %q, want %q", i, in, parser.userToHarmony[in], want)
				}
				if parser.harmonyToUser[want] != in {
					t.Errorf("case %d: harmonyToUser[%q] = %q, want %q", i, want, parser.harmonyToUser[want], in)
				}
			}
		})
	}
}
