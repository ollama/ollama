package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/require"
)

const preciseToolArguments = `{"id":9007199254740993,"decimal":1.234567890123456789,"nested":{"values":[-9007199254740993,42,0,0.5]},"label":"unchanged","enabled":true,"empty":null}`

func TestToolCallFunctionArgumentsNumericPrecision(t *testing.T) {
	// Arrange.
	var args ToolCallFunctionArguments
	// Act.
	err := json.Unmarshal([]byte(preciseToolArguments), &args)
	// Assert.
	require.NoError(t, err)
	encoded, err := json.Marshal(args)
	require.NoError(t, err)
	require.Equal(t, preciseToolArguments, string(encoded))
	id, ok := args.Get("id")
	require.True(t, ok)
	require.Equal(t, json.Number("9007199254740993"), id)
}

func TestChatToolCallNumericPrecision(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			// Arrange: use literal response bytes to avoid rounding in the fixture.
			bodies := make(chan []byte, 2)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				bodies <- body
				w.Header().Set("Content-Type", "application/x-ndjson")
				fmt.Fprintln(w, `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":`+preciseToolArguments+`}}]},"done":true}`)
			}))
			defer server.Close()
			endpoint, err := url.Parse(server.URL)
			require.NoError(t, err)
			client := NewClient(endpoint, server.Client())
			request := &ChatRequest{Model: "fixture", Stream: &stream, Messages: []Message{{Role: "user", Content: "lookup"}}}
			var response ChatResponse
			// Act: receive a tool call, then send it back in conversation history.
			err = client.Chat(context.Background(), request, func(r ChatResponse) error { response = r; return nil })
			require.NoError(t, err)
			request.Messages = append(request.Messages, response.Message)
			err = client.Chat(context.Background(), request, func(ChatResponse) error { return nil })
			// Assert.
			require.NoError(t, err)
			require.Len(t, response.Message.ToolCalls, 1)
			encoded, err := json.Marshal(response.Message.ToolCalls[0].Function.Arguments)
			require.NoError(t, err)
			require.Equal(t, preciseToolArguments, string(encoded))
			<-bodies
			require.Contains(t, string(<-bodies), `"arguments":`+preciseToolArguments)
		})
	}
}

func TestToolCallFunctionArgumentsNumericTypes(t *testing.T) {
	// Arrange.
	var args ToolCallFunctionArguments
	// Act.
	err := json.Unmarshal([]byte(`{"small":42,"fraction":0.5,"nested":{"numbers":[0,9007199254740993]},"text":"42"}`), &args)
	// Assert: all JSON numbers use the same representation, even small numbers.
	require.NoError(t, err)
	values := args.ToMap()
	require.Equal(t, json.Number("42"), values["small"])
	require.Equal(t, json.Number("0.5"), values["fraction"])
	require.Equal(t, []any{json.Number("0"), json.Number("9007199254740993")}, values["nested"].(map[string]any)["numbers"])
	require.Equal(t, "42", values["text"])
}

func TestToolCallFunctionArgumentsInvalidNumbers(t *testing.T) {
	for _, input := range []string{`{"id":01}`, `{"id":NaN}`, `{"id":1e}`, `{"id":1} {"id":2}`} {
		t.Run(input, func(t *testing.T) {
			// Arrange.
			var args ToolCallFunctionArguments
			// Act.
			err := json.Unmarshal([]byte(input), &args)
			// Assert.
			require.Error(t, err)
		})
	}
}

func TestToolCallFunctionArgumentsCallerNumbers(t *testing.T) {
	// Arrange.
	args := NewToolCallFunctionArguments()
	args.Set("id", int64(9007199254740993))
	args.Set("decimal", json.Number("1.234567890123456789"))
	// Act.
	encoded, err := json.Marshal(args)
	// Assert: explicit caller types retain their existing behavior.
	require.NoError(t, err)
	require.Equal(t, `{"id":9007199254740993,"decimal":1.234567890123456789}`, string(encoded))
	id, ok := args.Get("id")
	require.True(t, ok)
	require.Equal(t, int64(9007199254740993), id)
}
