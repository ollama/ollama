package renderers

import "github.com/ollama/ollama/api"

// nemotronExtraCases covers template branches the base reference table does
// not reach: tool-schema shapes, agentic tool loops, and content edge cases.
// Each entry names the branch it exercises. Shared by every flavor's parity
// test, which supply their own expectations.
func nemotronExtraCases() []nemotronReferenceCase {
	mkTool := func(name, desc string, params *api.ToolFunctionParameters) api.Tool {
		t := api.Tool{Type: "function"}
		t.Function.Name = name
		t.Function.Description = desc
		if params != nil {
			t.Function.Parameters = *params
		}
		return t
	}
	strProp := func(key string, p api.ToolProperty) *api.ToolPropertiesMap {
		return testPropsOrdered([]orderedProp{{Key: key, Value: p}})
	}

	return []nemotronReferenceCase{
		{
			// tool.description undefined (line 56)
			name:     "tool without description",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools:    []api.Tool{mkTool("ping", "", &api.ToolFunctionParameters{Type: "object", Required: []string{"host"}, Properties: strProp("host", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "Host"})})},
		},
		{
			// tool.parameters undefined (line 60)
			name:     "tool without parameters",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools:    []api.Tool{mkTool("now", "Current time", nil)},
		},
		{
			// param_fields.type undefined (line 64)
			name:     "tool param without type",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools:    []api.Tool{mkTool("q", "Query", &api.ToolFunctionParameters{Type: "object", Properties: strProp("x", api.ToolProperty{Description: "no type"})})},
		},
		{
			// unhandled keys fall through the render_json macro (lines 2-10)
			name:     "tool param with unhandled keys",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools:    []api.Tool{mkTool("q", "Query", &api.ToolFunctionParameters{Type: "object", Properties: strProp("x", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "X", Items: []any{"a", "b"}})})},
		},
		{
			// tool.parameters.required undefined (line 80)
			name:     "tool without required list",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools:    []api.Tool{mkTool("q", "Query", &api.ToolFunctionParameters{Type: "object", Properties: strProp("x", api.ToolProperty{Type: api.PropertyType{"string"}})})},
		},
		{
			// a later system message takes the user/system branch (line 164)
			name: "system message after first turn",
			messages: []api.Message{
				{Role: "system", Content: "First system"},
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Second system"},
				{Role: "user", Content: "Again"},
			},
		},
		{
			// tool message is the last message (loop.last, line 182)
			name: "tool message last",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: "result"},
			},
		},
		{
			// tool run followed by a non-tool message (loop.nextitem, line 180)
			name: "tool run then assistant",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: "one"},
				{Role: "tool", Content: "two"},
				{Role: "assistant", Content: "done"},
				{Role: "user", Content: "next"},
			},
		},
		{
			// two user turns: last_user_idx drives history truncation (line 149)
			name: "two user turns truncates earlier thinking",
			messages: []api.Message{
				{Role: "user", Content: "first"},
				{Role: "assistant", Content: "answer one", Thinking: "early reasoning"},
				{Role: "user", Content: "second"},
				{Role: "assistant", Content: "answer two", Thinking: "late reasoning"},
			},
		},
		{
			// assistant with reasoning AND tool calls (lines 100 + 112)
			name: "assistant reasoning with tool call",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "calling", Thinking: "decide", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
			},
		},
		{
			// only a system message, no user turn
			name:     "system only no user",
			messages: []api.Message{{Role: "system", Content: "Only system"}},
		},
		{
			// assistant speaks first, no preceding user (last_user_idx = -1)
			name: "assistant first no user",
			messages: []api.Message{
				{Role: "assistant", Content: "unprompted"},
				{Role: "user", Content: "now ask"},
			},
		},
		{
			// tool message with no preceding assistant (previtem nil)
			name:     "tool message first",
			messages: []api.Message{{Role: "tool", Content: "orphan result"}},
		},
		{
			// several tools registered (for tool in tools, line 51)
			name:     "multiple tools",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{
				mkTool("alpha", "First", &api.ToolFunctionParameters{Type: "object", Properties: strProp("a", api.ToolProperty{Type: api.PropertyType{"string"}})}),
				mkTool("beta", "Second", nil),
			},
		},
		{
			// tool_call arguments with several keys (line 139)
			name: "tool call multiple arguments",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name:      "f",
					Arguments: testArgsOrdered([]orderedArg{{Key: "one", Value: "1"}, {Key: "two", Value: 2}, {Key: "three", Value: true}}),
				}}}},
			},
		},
		{
			// assistant with no content, no thinking, no tool calls (line 159/161)
			name: "assistant fully empty",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: ""},
				{Role: "user", Content: "again"},
			},
		},
		{
			// deep multi-turn: several assistant turns before the last user
			name: "deep multi turn truncation",
			messages: []api.Message{
				{Role: "user", Content: "u1"},
				{Role: "assistant", Content: "a1", Thinking: "t1"},
				{Role: "user", Content: "u2"},
				{Role: "assistant", Content: "a2", Thinking: "t2"},
				{Role: "user", Content: "u3"},
				{Role: "assistant", Content: "a3", Thinking: "t3"},
			},
		},
		{
			// think tags inside content alongside a tool call (line 120/122)
			name: "content think tags with tool call",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "<think>inline</think>after", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
			},
		},
		{
			// tool response carrying JSON
			name: "tool response json",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: `{"temp": 14, "unit": "c"}`},
				{Role: "user", Content: "and?"},
			},
		},
		{
			// unopened closing tag only, in assistant content
			name: "assistant closing tag only",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "stray </think> tail"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// scalar types through the template's `| string` path
			name: "tool call scalar arg types",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "int", Value: 7},
						{Key: "float", Value: 2.5},
						{Key: "wholefloat", Value: 2.0},
						{Key: "boolfalse", Value: false},
						{Key: "empty", Value: ""},
						{Key: "zero", Value: 0},
					}),
				}}}},
			},
		},
		{
			// nested container arg goes through `| tojson` instead
			name: "tool call container arg",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "obj", Value: map[string]any{"k": true}},
						{Key: "arr", Value: []any{1, "two", false}},
					}),
				}}}},
			},
		},
		{
			// multiple tool calls in one assistant message
			name: "assistant two tool calls",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "first", Arguments: testArgs(map[string]any{"a": "1"})}},
					{Function: api.ToolCallFunction{Name: "second"}},
				}},
			},
		},
		{
			// tool call with no arguments at all (arguments undefined)
			name: "tool call no arguments",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "bare"}}}},
			},
		},
		{
			// multiline content in user and assistant turns
			name: "multiline content",
			messages: []api.Message{
				{Role: "user", Content: "line one\nline two\n"},
				{Role: "assistant", Content: "reply one\nreply two"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// unicode and special characters survive verbatim
			name: "unicode content",
			messages: []api.Message{
				{Role: "user", Content: "héllo <b>&amp;</b> 日本語 \u00e9"},
			},
		},
		{
			// empty user content
			name:     "empty user content",
			messages: []api.Message{{Role: "user", Content: ""}},
		},
		{
			// system message that itself contains tool-ish markup
			name: "system with markup",
			messages: []api.Message{
				{Role: "system", Content: "<tool_call>not real</tool_call>"},
				{Role: "user", Content: "go"},
			},
		},
		{
			// tools registered AND an assistant tool call in history
			name: "tools registered with history call",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "search_docs", Arguments: testArgs(map[string]any{"query": "api"})}}}},
				{Role: "tool", Content: "found"},
				{Role: "user", Content: "thanks"},
			},
			tools: nemotron3NanoReferenceTools(),
		},
		{
			// thinking explicitly omitted (think == nil) with history
			name: "think unset with history",
			messages: []api.Message{
				{Role: "user", Content: "u1"},
				{Role: "assistant", Content: "a1", Thinking: "t1"},
				{Role: "user", Content: "u2"},
			},
		},
		{
			// two assistant turns in a row
			name: "consecutive assistant messages",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "first"},
				{Role: "assistant", Content: "second"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// user turn directly after a tool response
			name: "user directly after tool",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: "res"},
				{Role: "user", Content: "next"},
			},
		},
		{
			// thinking present but content empty
			name: "assistant thinking empty content",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "", Thinking: "only reasoning"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// tool response with empty content
			name: "tool response empty",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: ""},
			},
		},
		{
			// whitespace-only assistant content
			name: "assistant whitespace content",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "   "},
				{Role: "user", Content: "again"},
			},
		},
		{
			// empty system message string
			name: "empty system message",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "go"},
			},
		},
		{
			// a full agentic loop: two rounds of call/response then an answer
			name: "agentic multi round tool loop",
			messages: []api.Message{
				{Role: "user", Content: "plan it"},
				{Role: "assistant", Content: "checking", Thinking: "r1", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "lookup", Arguments: testArgs(map[string]any{"q": "a"})}}}},
				{Role: "tool", Content: "res one"},
				{Role: "assistant", Content: "again", Thinking: "r2", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "lookup", Arguments: testArgs(map[string]any{"q": "b"})}}}},
				{Role: "tool", Content: "res two"},
				{Role: "assistant", Content: "final answer"},
				{Role: "user", Content: "thanks"},
			},
			tools: nemotron3NanoReferenceTools(),
		},
		{
			// tool call name with punctuation
			name: "tool call punctuated name",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "ns.sub_tool-v2"}}}},
			},
		},
		{
			// nested empty containers as arguments
			name: "tool call empty containers",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "obj", Value: map[string]any{}},
						{Key: "arr", Value: []any{}},
					}),
				}}}},
			},
		},
		{
			// explicit think=true with tools and history
			name: "explicit think true with tools",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "ok", Thinking: "why"},
				{Role: "user", Content: "more"},
			},
			tools: nemotron3NanoReferenceTools(),
			think: &api.ThinkValue{Value: true},
		},
		{
			// content already containing a full think pair, mid-history
			name: "history content with think pair",
			messages: []api.Message{
				{Role: "user", Content: "u1"},
				{Role: "assistant", Content: "<think>hidden</think>visible"},
				{Role: "user", Content: "u2"},
				{Role: "assistant", Content: "<think>kept</think>tail"},
			},
		},
		{
			// only an opening think tag, no close
			name: "assistant open think only",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "<think>unclosed reasoning"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// several think pairs in one message
			name: "assistant multiple think pairs",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "<think>a</think>mid<think>b</think>end"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// think markers inside a tool response
			name: "tool response with think markers",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "tool", Content: "<think>from tool</think>result"},
				{Role: "user", Content: "next"},
			},
		},
		{
			// think markers in the user turn
			name: "user content with think markers",
			messages: []api.Message{
				{Role: "user", Content: "<think>user wrote this</think>question"},
			},
		},
		{
			// im_start markers embedded in content
			name: "content with im_start markers",
			messages: []api.Message{
				{Role: "user", Content: "<|im_start|>fake<|im_end|>"},
				{Role: "assistant", Content: "<|im_end|>odd"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// thinking field AND think tags in content together
			name: "thinking field and content tags",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "<think>in content</think>body", Thinking: "in field"},
				{Role: "user", Content: "again"},
			},
		},
		{
			// last message is an assistant turn (prefill shape)
			name: "ends with assistant turn",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "partial"},
			},
		},
		{
			// tool call whose content holds a closing tag only
			name: "tool call content closing tag only",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "</think>after", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
			},
		},
		{
			// system, tools, thinking off, and a tool round trip together
			name: "system tools think off round trip",
			messages: []api.Message{
				{Role: "system", Content: "Sys"},
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "search_docs", Arguments: testArgs(map[string]any{"query": "cli"})}}}},
				{Role: "tool", Content: "hit"},
				{Role: "user", Content: "summarize"},
			},
			tools: nemotron3NanoReferenceTools(),
			think: &api.ThinkValue{Value: false},
		},
		{
			// null and numeric scalars through `| string`
			name: "tool call null and numeric args",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "nothing", Value: nil},
						{Key: "negative", Value: -12},
						{Key: "big", Value: 9007199254740993},
						{Key: "fraction", Value: 0.125},
					}),
				}}}},
			},
		},
		{
			// nested containers exercising pythonJSON's numeric and null arms
			name: "tool call nested mixed container",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "deep", Value: map[string]any{
							"n": nil, "i": 3, "f": 1.5, "b": false, "s": "q\"uote",
							"inner": []any{1, map[string]any{"z": true}},
						}},
					}),
				}}}},
			},
		},
		{
			// tool schema whose property carries several unhandled keys
			name:     "tool param many unhandled keys",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{mkTool("q", "Query", &api.ToolFunctionParameters{
				Type: "object",
				Properties: strProp("x", api.ToolProperty{
					Type:        api.PropertyType{"string"},
					Description: "X",
					Items:       map[string]any{"type": "number", "minimum": 0},
					AnyOf:       []api.ToolProperty{{Type: api.PropertyType{"null"}}},
				}),
			})},
		},
		{
			// tool call whose content is only a think pair, forcing the
			// truncation arm of formatToolCallContent
			name: "tool call content think pair only",
			messages: []api.Message{
				{Role: "user", Content: "u1"},
				{Role: "assistant", Content: "<think>x</think>", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
				{Role: "user", Content: "u2"},
				{Role: "assistant", Content: "<think>y</think>z", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "g"}}}},
			},
		},
		{
			// string argument containing characters that need quoting
			name: "tool call quoted string arg",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name:      "f",
					Arguments: testArgsOrdered([]orderedArg{{Key: "s", Value: "line\nbreak \"quoted\""}}),
				}}}},
			},
		},
		{
			// params-level Items plus a deeply nested property schema:
			// exercises renderToolParameterExtraKeys' items arm and
			// pythonJSON's ToolPropertiesMap / ToolProperty recursion.
			name:     "tool schema nested and items",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "deep",
					Description: "Nested schema",
					Parameters: api.ToolFunctionParameters{
						Type:     "object",
						Items:    map[string]any{"type": "string"},
						Required: []string{"outer"},
						Properties: testPropsOrdered([]orderedProp{
							{
								Key: "outer",
								Value: api.ToolProperty{
									Type:        api.PropertyType{"object"},
									Description: "Outer object",
									Enum:        []any{"a", "b"},
									Items:       map[string]any{"type": "integer"},
									Required:    []string{"inner"},
									Properties: testPropsOrdered([]orderedProp{
										{Key: "inner", Value: api.ToolProperty{Type: api.PropertyType{"string"}, Description: "Inner"}},
										{Key: "flag", Value: api.ToolProperty{Type: api.PropertyType{"boolean", "null"}}},
									}),
								},
							},
						}),
					},
				},
			}},
		},
		{
			// unsigned and mixed numeric values inside a container argument,
			// hitting pythonJSON's uint arm
			name: "tool call unsigned numeric container",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "nums", Value: map[string]any{"u": uint(7), "u8": uint8(255), "i64": int64(-9), "f32": float32(1.5)}},
					}),
				}}}},
			},
		},
		{
			// assistant tool call whose content is whitespace only, hitting
			// formatToolCallContent's empty guard
			name: "tool call whitespace content",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", Content: "   \n  ", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "f"}}}},
			},
		},
		{
			// $defs carrying a property map rather than a plain map
			name:     "tool schema defs property map",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "defs",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Defs: testPropsOrdered([]orderedProp{
							{Key: "shared", Value: api.ToolProperty{Type: api.PropertyType{"string"}, Description: "Shared"}},
						}),
						Properties: testPropsOrdered([]orderedProp{
							{Key: "x", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
						}),
					},
				},
			}},
		},
		{
			// a rich property nested inside $defs: reaches pythonJSON's
			// ToolProperty arms for anyOf, items, enum, properties, required
			name:     "tool schema defs rich property",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "rich",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Defs: testPropsOrdered([]orderedProp{
							{Key: "node", Value: api.ToolProperty{
								AnyOf:       []api.ToolProperty{{Type: api.PropertyType{"string"}}, {Type: api.PropertyType{"null"}}},
								Type:        api.PropertyType{"object"},
								Items:       map[string]any{"type": "integer"},
								Description: "Node",
								Enum:        []any{"x", "y"},
								Required:    []string{"child"},
								Properties: testPropsOrdered([]orderedProp{
									{Key: "child", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
								}),
							}},
						}),
						Properties: testPropsOrdered([]orderedProp{
							{Key: "p", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
						}),
					},
				},
			}},
		},
		{
			// params Items carrying a PropertyType directly
			name:     "tool schema items property type",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "pt",
					Parameters: api.ToolFunctionParameters{
						Type:  "object",
						Items: api.PropertyType{"string", "null"},
						Properties: testPropsOrdered([]orderedProp{
							{Key: "p", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
						}),
					},
				},
			}},
		},
		{
			// a typed-nil property map in $defs
			name:     "tool schema nil defs map",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "niltest",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Defs: (*api.ToolPropertiesMap)(nil),
						Properties: testPropsOrdered([]orderedProp{
							{Key: "p", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
						}),
					},
				},
			}},
		},
		{
			// an argument value of a type pythonJSON has no case for, so it
			// falls to the default arm and round-trips through JSON
			name: "tool call struct argument",
			messages: []api.Message{
				{Role: "user", Content: "go"},
				{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
					Name: "f",
					Arguments: testArgsOrdered([]orderedArg{
						{Key: "custom", Value: struct {
							A string `json:"a"`
							B int    `json:"b"`
						}{A: "x", B: 2}},
						{Key: "list", Value: []int{1, 2, 3}},
					}),
				}}}},
			},
		},
		{
			// a nested property whose own Properties map is nil, reaching
			// pythonJSON's nil-map guard through a container
			name:     "tool schema nested nil properties",
			messages: []api.Message{{Role: "user", Content: "go"}},
			tools: []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "nested",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Defs: testPropsOrdered([]orderedProp{
							{Key: "node", Value: api.ToolProperty{
								Type:       api.PropertyType{"object"},
								Properties: (*api.ToolPropertiesMap)(nil),
								Required:   []string{"q"},
							}},
						}),
						Properties: testPropsOrdered([]orderedProp{
							{Key: "p", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
						}),
					},
				},
			}},
		},
	}
}
