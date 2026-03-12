package server

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
)

type qwen35RoundTripCase struct {
	name                         string
	think                        bool
	rawGeneratedSuffix           string
	expectParsedContent          string
	expectParsedThinking         string
	expectCanonicalizedBody      string
	allowCanonicalizationDrift   bool
	expectFullGeneratedReuse     bool
	expectFullAssistantRoundTrip bool
	expectToolCallBodyRoundTrip  bool
	expectFreshUserBoundaryShift bool
	expectNoCompletedToolCall    bool
	clientStyle                  qwen35ClientStyle
	toolResultStyle              qwen35ToolResultStyle
	continuationStyle            qwen35ContinuationStyle
}

type qwen35ClientStyle string

const (
	qwen35ClientStyleDirectStructured qwen35ClientStyle = "direct-structured"
	qwen35ClientStyleJSONRoundTrip    qwen35ClientStyle = "json-roundtrip"
)

type qwen35ToolResultStyle string

const (
	qwen35ToolResultRoleTool         qwen35ToolResultStyle = "role-tool"
	qwen35ToolResultUserToolResponse qwen35ToolResultStyle = "user-tool-response"
)

type qwen35ContinuationStyle string

const (
	qwen35ContinuationToolOnly  qwen35ContinuationStyle = "tool-only"
	qwen35ContinuationFreshUser qwen35ContinuationStyle = "fresh-user"
)

type qwen35RoundTripObservation struct {
	prevPrompt                string
	nextPrompt                string
	prevFull                  string
	prevAssistantTranscript   string
	nextAssistantTranscript   string
	totalPrefixReuseBytes     int
	generatedReuseBytes       int
	generatedTotalBytes       int
	assistantTranscriptCommon int
	prefillStub               string
	prefillStubReuseBytes     int
	toolCallBody              string
	toolCallBodyReuseBytes    int
	baseHistoryBytes          int
	baseHistoryReuseBytes     int
	content                   string
	thinking                  string
	calls                     []api.ToolCall
	continuationStyle         qwen35ContinuationStyle
}

func TestQwen35KVCacheRoundTripConsistency(t *testing.T) {
	t.Helper()

	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "exec",
			Description: "Execute a command or payload.",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"payload"},
				Properties: qwen35Props(
					qwen35Prop("cmd", api.ToolProperty{Type: api.PropertyType{"string"}}),
					qwen35Prop("payload", api.ToolProperty{Type: api.PropertyType{"object"}}),
					qwen35Prop("n", api.ToolProperty{Type: api.PropertyType{"number"}}),
					qwen35Prop("flag", api.ToolProperty{Type: api.PropertyType{"boolean"}}),
				),
			},
		},
	}

	baseMessages := []api.Message{
		{Role: "system", Content: "You are a coding agent."},
		{Role: "user", Content: "Use the tool and continue."},
	}

	cases := []qwen35RoundTripCase{
		{
			name:                         "tool-only continuation: canonical scalar tool call with think=false",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical scalar tool call with think=false and prose before tool call",
			think:                        false,
			rawGeneratedSuffix:           "I'll run that now.\n\n" + qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "I'll run that now.",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical scalar tool call with think=true",
			think:                        true,
			rawGeneratedSuffix:           "\n</think>\n\n" + qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical scalar tool call with think=true and explicit reasoning before tool call",
			think:                        true,
			rawGeneratedSuffix:           "Need to inspect the workspace first.\n</think>\n\n" + qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "Need to inspect the workspace first.",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical structured JSON argument",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical structured JSON argument via json roundtrip client",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleJSONRoundTrip,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: canonical scalar tool call with user-style tool_response wrapper",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultUserToolResponse,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                       "tool-only continuation: compact structured JSON spacing",
			think:                      false,
			rawGeneratedSuffix:         qwen35ToolCallXML("exec", "payload", `{"a":1,"b":2}`),
			expectParsedContent:        "",
			expectParsedThinking:       "",
			expectCanonicalizedBody:    qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			allowCanonicalizationDrift: true,
			clientStyle:                qwen35ClientStyleDirectStructured,
			toolResultStyle:            qwen35ToolResultRoleTool,
			continuationStyle:          qwen35ContinuationToolOnly,
		},
		{
			name:                       "tool-only continuation: extra structured JSON spacing",
			think:                      false,
			rawGeneratedSuffix:         qwen35ToolCallXML("exec", "payload", `{  "a" : 1,   "b" : 2  }`),
			expectParsedContent:        "",
			expectParsedThinking:       "",
			expectCanonicalizedBody:    qwen35ToolCallXML("exec", "payload", `{"a": 1, "b": 2}`),
			allowCanonicalizationDrift: true,
			clientStyle:                qwen35ClientStyleDirectStructured,
			toolResultStyle:            qwen35ToolResultRoleTool,
			continuationStyle:          qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: noncanonical key order",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"z": 1, "a": 2}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"z": 1, "a": 2}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: nested key order drift",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"outer": {"z": 1, "a": 2}, "b": 3}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"outer": {"z": 1, "a": 2}, "b": 3}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: number lexical form 1.0",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "n", "1.0"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "n", "1.0"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: number lexical form 1e3",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "n", "1e3"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "n", "1e3"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: literal html chars remain literal",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"snippet": "if (x < 5 && y > 3) {}", "tag": "<ok>&</ok>"}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"snippet": "if (x < 5 && y > 3) {}", "tag": "<ok>&</ok>"}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: real client resend via JSON roundtrip and user tool_response",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "payload", `{"z": 1, "a": 2}`),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "payload", `{"z": 1, "a": 2}`),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleJSONRoundTrip,
			toolResultStyle:              qwen35ToolResultUserToolResponse,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "tool-only continuation: real client resend via JSON roundtrip canonical scalar",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectFullGeneratedReuse:     true,
			expectFullAssistantRoundTrip: true,
			expectToolCallBodyRoundTrip:  true,
			clientStyle:                  qwen35ClientStyleJSONRoundTrip,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationToolOnly,
		},
		{
			name:                         "fresh-user continuation: canonical scalar tool call stays stable after the expected boundary shift",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectToolCallBodyRoundTrip:  true,
			expectFreshUserBoundaryShift: true,
			clientStyle:                  qwen35ClientStyleDirectStructured,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationFreshUser,
		},
		{
			name:                         "fresh-user continuation: real client resend via JSON roundtrip keeps a canonical scalar tool-call body stable",
			think:                        false,
			rawGeneratedSuffix:           qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectParsedContent:          "",
			expectParsedThinking:         "",
			expectCanonicalizedBody:      qwen35ToolCallXML("exec", "cmd", "ls -la"),
			expectToolCallBodyRoundTrip:  true,
			expectFreshUserBoundaryShift: true,
			clientStyle:                  qwen35ClientStyleJSONRoundTrip,
			toolResultStyle:              qwen35ToolResultRoleTool,
			continuationStyle:            qwen35ContinuationFreshUser,
		},
		{
			name:                      "tool-only continuation: cancelled tool call before close tag",
			think:                     false,
			rawGeneratedSuffix:        "<tool_call>\n<function=exec>\n<parameter=cmd>\nls",
			expectParsedContent:       "",
			expectParsedThinking:      "",
			expectNoCompletedToolCall: true,
			clientStyle:               qwen35ClientStyleJSONRoundTrip,
			toolResultStyle:           qwen35ToolResultRoleTool,
			continuationStyle:         qwen35ContinuationToolOnly,
		},
		{
			name:                      "tool-only continuation: cancelled think=true reasoning before think close",
			think:                     true,
			rawGeneratedSuffix:        "Need to inspect the workspace first.",
			expectParsedContent:       "",
			expectParsedThinking:      "Need to inspect the workspace first.",
			expectNoCompletedToolCall: true,
			clientStyle:               qwen35ClientStyleDirectStructured,
			toolResultStyle:           qwen35ToolResultRoleTool,
			continuationStyle:         qwen35ContinuationToolOnly,
		},
		{
			name:                      "tool-only continuation: cancelled think=true after think close but before tool call completes",
			think:                     true,
			rawGeneratedSuffix:        "Need to inspect the workspace first.\n</think>\n\n<tool_call>\n<function=exec>\n<parameter=cmd>\nls",
			expectParsedContent:       "",
			expectParsedThinking:      "Need to inspect the workspace first.",
			expectNoCompletedToolCall: true,
			clientStyle:               qwen35ClientStyleDirectStructured,
			toolResultStyle:           qwen35ToolResultRoleTool,
			continuationStyle:         qwen35ContinuationToolOnly,
		},
		{
			name:                      "tool-only continuation: cancelled think=false with prose before tool call completes",
			think:                     false,
			rawGeneratedSuffix:        "I'll run that now.\n\n<tool_call>\n<function=exec>\n<parameter=cmd>\nls",
			expectParsedContent:       "I'll run that now.",
			expectParsedThinking:      "",
			expectNoCompletedToolCall: true,
			clientStyle:               qwen35ClientStyleDirectStructured,
			toolResultStyle:           qwen35ToolResultRoleTool,
			continuationStyle:         qwen35ContinuationToolOnly,
		},
	}

	var issues []string
	for _, tc := range cases {
		tc := tc
		obs, err := observeQwen35RoundTrip(baseMessages, []api.Tool{tool}, tc)
		if err != nil {
			issues = append(issues, fmt.Sprintf("%s\nfailed to build round-trip observation: %v", tc.name, err))
			continue
		}

		if tc.expectNoCompletedToolCall {
			if obs.content != tc.expectParsedContent {
				issues = append(issues, fmt.Sprintf("%s\nunexpected parsed assistant content for cancellation case.\nExpected content:\n%q\nObserved content:\n%q\nRaw generated suffix:\n%s",
					tc.name, tc.expectParsedContent, obs.content, obs.rawGeneratedSuffix()))
			}
			if obs.thinking != tc.expectParsedThinking {
				issues = append(issues, fmt.Sprintf("%s\nunexpected parsed assistant thinking for cancellation case.\nExpected thinking:\n%q\nObserved thinking:\n%q\nRaw generated suffix:\n%s",
					tc.name, tc.expectParsedThinking, obs.thinking, obs.rawGeneratedSuffix()))
			}
			if len(obs.calls) != 0 {
				issues = append(issues, fmt.Sprintf("%s\nexpected cancellation before `</tool_call>` to surface no completed tool calls, but parser produced %d call(s): %#v",
					tc.name, len(obs.calls), obs.calls))
			}
			if strings.Contains(obs.nextAssistantTranscript, "<tool_call>") {
				issues = append(issues, fmt.Sprintf("%s\nexpected cancellation before `</tool_call>` to drop the incomplete tool-call XML from historical assistant replay, but the next assistant transcript still contains `<tool_call>`.\nRaw generated suffix:\n%s\nNext assistant transcript:\n%s",
					tc.name, obs.rawGeneratedSuffix(), obs.nextAssistantTranscript))
			}
			continue
		}

		if obs.content != tc.expectParsedContent {
			issues = append(issues, fmt.Sprintf("%s\nunexpected parsed assistant content.\nExpected content:\n%q\nObserved content:\n%q\nRaw generated suffix:\n%s",
				tc.name, tc.expectParsedContent, obs.content, obs.rawGeneratedSuffix()))
		}
		if obs.thinking != tc.expectParsedThinking {
			issues = append(issues, fmt.Sprintf("%s\nunexpected parsed assistant thinking.\nExpected thinking:\n%q\nObserved thinking:\n%q\nRaw generated suffix:\n%s",
				tc.name, tc.expectParsedThinking, obs.thinking, obs.rawGeneratedSuffix()))
		}

		if tc.expectCanonicalizedBody != "" && obs.toolCallBody != tc.expectCanonicalizedBody {
			issues = append(issues, fmt.Sprintf("%s\nunexpected canonicalized historical tool-call body.\nExpected canonical body:\n%s\nObserved canonical body:\n%s\nRaw generated suffix:\n%s",
				tc.name, tc.expectCanonicalizedBody, obs.toolCallBody, obs.rawGeneratedSuffix()))
		}
		if tc.allowCanonicalizationDrift {
			rawToolCallBody := qwen35RawToolCallBody(obs.rawGeneratedSuffix())
			if tc.rawGeneratedSuffix == tc.expectCanonicalizedBody {
				issues = append(issues, fmt.Sprintf("%s\ntest setup bug: allowCanonicalizationDrift is set, but the raw generated suffix already matches the expected canonical body.\nRaw generated suffix:\n%s",
					tc.name, obs.rawGeneratedSuffix()))
			}
			if rawToolCallBody == "" {
				issues = append(issues, fmt.Sprintf("%s\ntest setup bug: allowCanonicalizationDrift is set, but no tool-call body could be extracted from the raw generated suffix.\nRaw generated suffix:\n%s",
					tc.name, obs.rawGeneratedSuffix()))
			} else if obs.toolCallBody == rawToolCallBody {
				issues = append(issues, fmt.Sprintf("%s\nexpected non-canonical tool-call JSON to be canonicalized to the official Qwen 3.5 form, but the historical assistant replay kept the original non-canonical bytes.\nRaw generated suffix:\n%s\nObserved historical tool-call body:\n%s",
					tc.name, obs.rawGeneratedSuffix(), obs.toolCallBody))
			}
		}

		if obs.baseHistoryReuseBytes != obs.baseHistoryBytes {
			issues = append(issues, fmt.Sprintf(
				"%s\nstable history before the trailing assistant prefill/tool-call boundary did not fully reuse.\n"+
					"Expected base history reuse: %d/%d bytes\n"+
					"Observed base history reuse: %d/%d bytes\n"+
					"This means the cache miss started earlier than the newest assistant turn, which is worse than a 'latest-turn-only' miss.\n"+
					"Previous prompt around the history boundary:\n%s\n"+
					"Next prompt around the history boundary:\n%s",
				tc.name,
				obs.baseHistoryBytes, obs.baseHistoryBytes,
				obs.baseHistoryReuseBytes, obs.baseHistoryBytes,
				qwen35Excerpt(obs.prevFull, obs.baseHistoryReuseBytes),
				qwen35Excerpt(obs.nextPrompt, obs.baseHistoryReuseBytes),
			))
		}

		if tc.continuationStyle == qwen35ContinuationToolOnly && obs.prefillStub != "" && obs.prefillStubReuseBytes != len(obs.prefillStub) {
			issues = append(issues, fmt.Sprintf(
				"%s\nassistant prefill stub did not round-trip.\n"+
					"Prefill stub reused: %d/%d bytes\n"+
					"Prefill stub:\n%s\n"+
					"This pinpoints a cache miss in the newest assistant turn before the tool-call body even begins.\n"+
					"Previous assistant transcript:\n%s\n"+
					"Next historical assistant transcript:\n%s",
				tc.name,
				obs.prefillStubReuseBytes, len(obs.prefillStub),
				obs.prefillStub,
				obs.prevAssistantTranscript,
				obs.nextAssistantTranscript,
			))
		}

		if tc.expectToolCallBodyRoundTrip && obs.toolCallBodyReuseBytes != len(obs.toolCallBody) {
			issues = append(issues, fmt.Sprintf(
				"%s\ntool-call body did not round-trip byte-for-byte.\n"+
					"Tool-call body reused: %d/%d bytes\n"+
					"Tool-call body:\n%s\n"+
					"This is the cache-relevant portion of the newest assistant turn after any prefill/thinking stub.\n"+
					"Previous assistant transcript:\n%s\n"+
					"Next historical assistant transcript:\n%s",
				tc.name,
				obs.toolCallBodyReuseBytes, len(obs.toolCallBody),
				obs.toolCallBody,
				obs.prevAssistantTranscript,
				obs.nextAssistantTranscript,
			))
		}

		if tc.expectFullGeneratedReuse && obs.generatedReuseBytes != obs.generatedTotalBytes {
			issues = append(issues, qwen35GeneratedReuseFailure(tc.name, obs))
		}

		if tc.expectFullAssistantRoundTrip && (obs.assistantTranscriptCommon != len(obs.prevAssistantTranscript) || obs.prevAssistantTranscript != obs.nextAssistantTranscript) {
			issues = append(issues, qwen35AssistantTranscriptFailure(tc.name, obs))
		}

		if tc.expectFreshUserBoundaryShift {
			if obs.prefillStub == "" {
				issues = append(issues, fmt.Sprintf("%s\nexpected a prompt-side assistant prefill stub before the generated tool call, but none was found.\nPrevious assistant transcript:\n%s",
					tc.name, obs.prevAssistantTranscript))
			} else if obs.prefillStubReuseBytes >= len(obs.prefillStub) {
				issues = append(issues, fmt.Sprintf("%s\nexpected a fresh-user continuation to break at the assistant prefill boundary, but the prefill stub still round-tripped completely.\nThis usually means the scenario stopped being a true 'fresh user after tool result' case.\nPrefill stub:\n%s\nPrevious assistant transcript:\n%s\nNext historical assistant transcript:\n%s",
					tc.name, obs.prefillStub, obs.prevAssistantTranscript, obs.nextAssistantTranscript))
			}
		}

		if strings.Contains(obs.nextAssistantTranscript, `\u003c`) || strings.Contains(obs.nextAssistantTranscript, `\u003e`) || strings.Contains(obs.nextAssistantTranscript, `\u0026`) {
			issues = append(issues, fmt.Sprintf("%s\nhistorical assistant transcript still contains HTML-escaped characters after the serializer fix.\nNext assistant transcript:\n%s",
				tc.name, obs.nextAssistantTranscript))
		}
	}

	if len(issues) > 0 {
		t.Fatalf(
			"Qwen 3.5 round-trip inconsistencies detected.\n\n"+
				"Why this matters:\n"+
				"- `runner/llamarunner/cache.go` reuses KV state by matching a common input prefix.\n"+
				"- Any divergence in the rerendered next-turn prompt forces prompt reprocessing from that boundary onward.\n"+
				"- This harness separates the two real continuation shapes:\n"+
				"  * tool-result-only continuation, which is the primary agentic loop and should round-trip cleanly for canonical outputs\n"+
				"  * fresh-user continuation, where a boundary shift is expected but the historical tool-call body should still stay stable\n"+
				"- In rapid tool loops, failures here mean extra prompt ingestion, lower throughput, and avoidable cache misses exactly where coding agents spend their time.\n\n"+
				"Collected issues:\n\n%s",
			strings.Join(issues, "\n\n---\n\n"),
		)
	}
}

func observeQwen35RoundTrip(baseMessages []api.Message, tools []api.Tool, tc qwen35RoundTripCase) (qwen35RoundTripObservation, error) {
	think := &api.ThinkValue{Value: tc.think}
	prevPrompt, err := renderers.RenderWithRenderer("qwen3.5", baseMessages, tools, think)
	if err != nil {
		return qwen35RoundTripObservation{}, err
	}

	parser := parsers.ParserForName("qwen3.5")
	if parser == nil {
		return qwen35RoundTripObservation{}, fmt.Errorf("qwen3.5 parser not registered")
	}
	parser.Init(tools, nil, think)

	content, thinking, calls, err := parser.Add(tc.rawGeneratedSuffix, true)
	if err != nil {
		return qwen35RoundTripObservation{}, err
	}

	nextMessages := append([]api.Message{}, baseMessages...)
	if len(calls) > 0 || content != "" || thinking != "" {
		assistantMessage := api.Message{
			Role:      "assistant",
			Content:   content,
			Thinking:  thinking,
			ToolCalls: calls,
		}
		switch tc.clientStyle {
		case qwen35ClientStyleJSONRoundTrip:
			var roundTripped api.Message
			b, err := json.Marshal(assistantMessage)
			if err != nil {
				return qwen35RoundTripObservation{}, fmt.Errorf("marshal assistant message for client roundtrip: %w", err)
			}
			if err := json.Unmarshal(b, &roundTripped); err != nil {
				return qwen35RoundTripObservation{}, fmt.Errorf("unmarshal assistant message for client roundtrip: %w", err)
			}
			assistantMessage = roundTripped
		case qwen35ClientStyleDirectStructured:
		default:
			return qwen35RoundTripObservation{}, fmt.Errorf("unknown client style %q", tc.clientStyle)
		}
		nextMessages = append(nextMessages, assistantMessage)
	}
	if len(calls) > 0 {
		switch tc.toolResultStyle {
		case qwen35ToolResultRoleTool:
			nextMessages = append(nextMessages, api.Message{Role: "tool", Content: "ok"})
		case qwen35ToolResultUserToolResponse:
			nextMessages = append(nextMessages, api.Message{Role: "user", Content: "<tool_response>\nok\n</tool_response>"})
		default:
			return qwen35RoundTripObservation{}, fmt.Errorf("unknown tool result style %q", tc.toolResultStyle)
		}
	}
	switch tc.continuationStyle {
	case qwen35ContinuationToolOnly:
	case qwen35ContinuationFreshUser:
		nextMessages = append(nextMessages, api.Message{Role: "user", Content: "continue"})
	default:
		return qwen35RoundTripObservation{}, fmt.Errorf("unknown continuation style %q", tc.continuationStyle)
	}

	nextPrompt, err := renderers.RenderWithRenderer("qwen3.5", nextMessages, tools, think)
	if err != nil {
		return qwen35RoundTripObservation{}, err
	}

	prevFull := prevPrompt + tc.rawGeneratedSuffix
	prefix := qwen35CommonPrefixLen(prevFull, nextPrompt)
	reusedGenerated := prefix - len(prevPrompt)
	if reusedGenerated < 0 {
		reusedGenerated = 0
	}

	prevAssistant := qwen35PreviousAssistantTranscript(prevPrompt, tc.rawGeneratedSuffix)
	nextAssistant := qwen35FirstHistoricalAssistantTranscript(nextPrompt)
	prefillStub, toolCallBody := qwen35SplitAssistantTranscript(prevAssistant)
	baseHistoryBytes := len(prevPrompt) - len(qwen35AssistantOpenTag(prevPrompt))
	if baseHistoryBytes < 0 {
		baseHistoryBytes = 0
	}
	baseHistoryReuseBytes := min(prefix, baseHistoryBytes)

	prefillStubReuseBytes := 0
	toolCallBodyReuseBytes := 0
	if nextAssistant != "" {
		nextAssistantBody := nextAssistant[len("<|im_start|>assistant\n"):]
		if len(prefillStub) > 0 {
			prefillStubReuseBytes = qwen35CommonPrefixLen(prefillStub, nextAssistantBody)
			if prefillStubReuseBytes > len(prefillStub) {
				prefillStubReuseBytes = len(prefillStub)
			}
		}
		nextToolCallBody := nextAssistantBody
		if idx := strings.Index(nextAssistantBody, "<tool_call>"); idx != -1 {
			nextToolCallBody = nextAssistantBody[idx:]
		}
		toolCallBodyReuseBytes = qwen35CommonPrefixLen(toolCallBody, nextToolCallBody)
		if toolCallBodyReuseBytes > len(toolCallBody) {
			toolCallBodyReuseBytes = len(toolCallBody)
		}
	}

	return qwen35RoundTripObservation{
		prevPrompt:                prevPrompt,
		nextPrompt:                nextPrompt,
		prevFull:                  prevFull,
		prevAssistantTranscript:   prevAssistant,
		nextAssistantTranscript:   nextAssistant,
		totalPrefixReuseBytes:     prefix,
		generatedReuseBytes:       reusedGenerated,
		generatedTotalBytes:       len(tc.rawGeneratedSuffix),
		assistantTranscriptCommon: qwen35CommonPrefixLen(prevAssistant, nextAssistant),
		prefillStub:               prefillStub,
		prefillStubReuseBytes:     prefillStubReuseBytes,
		toolCallBody:              toolCallBody,
		toolCallBodyReuseBytes:    toolCallBodyReuseBytes,
		baseHistoryBytes:          baseHistoryBytes,
		baseHistoryReuseBytes:     baseHistoryReuseBytes,
		content:                   content,
		thinking:                  thinking,
		calls:                     calls,
		continuationStyle:         tc.continuationStyle,
	}, nil
}

func qwen35GeneratedReuseFailure(name string, obs qwen35RoundTripObservation) string {
	divergence := qwen35CommonPrefixLen(obs.prevFull, obs.nextPrompt)

	return fmt.Sprintf(
		"%s\nexpected the next prompt to reuse the full model-emitted assistant suffix (%d/%d bytes, excluding `<|im_end|>`), but only reused %d/%d bytes.\n"+
			"Region summary:\n"+
			"- stable prior history reused: %d/%d bytes\n"+
			"- assistant prefill stub reused: %d/%d bytes\n"+
			"- assistant tool-call body reused: %d/%d bytes\n"+
			"This is a direct KV-cache miss in the tool-result-only continuation path: the next request cannot fully reuse the assistant tool-call tokens it just generated.\n"+
			"Previous prompt tail around divergence:\n%s\n"+
			"Next prompt tail around divergence:\n%s\n"+
			"Previous assistant transcript as cached after generation:\n%s\n"+
			"Next historical assistant transcript:\n%s",
		name,
		obs.generatedReuseBytes, obs.generatedTotalBytes,
		obs.generatedReuseBytes, obs.generatedTotalBytes,
		obs.baseHistoryReuseBytes, obs.baseHistoryBytes,
		obs.prefillStubReuseBytes, len(obs.prefillStub),
		obs.toolCallBodyReuseBytes, len(obs.toolCallBody),
		qwen35Excerpt(obs.prevFull, divergence),
		qwen35Excerpt(obs.nextPrompt, divergence),
		obs.prevAssistantTranscript,
		obs.nextAssistantTranscript,
	)
}

func qwen35AssistantTranscriptFailure(name string, obs qwen35RoundTripObservation) string {
	divergence := qwen35CommonPrefixLen(obs.prevAssistantTranscript, obs.nextAssistantTranscript)
	return fmt.Sprintf(
		"%s\nassistant transcript did not round-trip byte-for-byte.\n"+
			"The parser produced structured history that rerenders differently from the assistant text that was just cached.\n"+
			"This guarantees prompt prefix drift on the next tool loop.\n"+
			"Previous assistant transcript:\n%s\n"+
			"Next historical assistant transcript:\n%s\n"+
			"Assistant divergence context (previous):\n%s\n"+
			"Assistant divergence context (next):\n%s",
		name,
		obs.prevAssistantTranscript,
		obs.nextAssistantTranscript,
		qwen35Excerpt(obs.prevAssistantTranscript, divergence),
		qwen35Excerpt(obs.nextAssistantTranscript, divergence),
	)
}

func qwen35CommonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func qwen35Excerpt(s string, pos int) string {
	start := pos - 60
	if start < 0 {
		start = 0
	}
	end := pos + 120
	if end > len(s) {
		end = len(s)
	}
	return s[start:end]
}

func qwen35AssistantOpenTag(prompt string) string {
	idx := strings.LastIndex(prompt, "<|im_start|>assistant\n")
	if idx == -1 {
		return ""
	}
	return prompt[idx:]
}

func qwen35PreviousAssistantTranscript(prevPrompt, rawSuffix string) string {
	idx := strings.LastIndex(prevPrompt, "<|im_start|>assistant\n")
	if idx == -1 {
		return rawSuffix
	}
	return prevPrompt[idx:] + rawSuffix
}

func qwen35FirstHistoricalAssistantTranscript(prompt string) string {
	const open = "<|im_start|>assistant\n"
	start := strings.Index(prompt, open)
	if start == -1 {
		return ""
	}
	rest := prompt[start:]
	end := strings.Index(rest, "<|im_end|>")
	if end == -1 {
		return rest
	}
	return rest[:end]
}

func qwen35SplitAssistantTranscript(transcript string) (prefillStub string, toolCallBody string) {
	const open = "<|im_start|>assistant\n"
	if !strings.HasPrefix(transcript, open) {
		return "", transcript
	}
	body := transcript[len(open):]
	if idx := strings.Index(body, "<tool_call>"); idx != -1 {
		return body[:idx], body[idx:]
	}
	return body, ""
}

func qwen35RawToolCallBody(raw string) string {
	if idx := strings.Index(raw, "<tool_call>"); idx != -1 {
		return raw[idx:]
	}
	return ""
}

func qwen35ToolCallXML(name, param, value string) string {
	return fmt.Sprintf("<tool_call>\n<function=%s>\n<parameter=%s>\n%s\n</parameter>\n</function>\n</tool_call>", name, param, value)
}

type qwen35OrderedProp struct {
	key  string
	prop api.ToolProperty
}

func qwen35Prop(key string, prop api.ToolProperty) qwen35OrderedProp {
	return qwen35OrderedProp{key: key, prop: prop}
}

func qwen35Props(entries ...qwen35OrderedProp) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for _, entry := range entries {
		props.Set(entry.key, entry.prop)
	}
	return props
}

func (o qwen35RoundTripObservation) rawGeneratedSuffix() string {
	if idx := len(o.prevFull) - o.generatedTotalBytes; idx >= 0 && idx <= len(o.prevFull) {
		return o.prevFull[idx:]
	}
	return ""
}
