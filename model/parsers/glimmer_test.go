package parsers

import (
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func glimmerTestTool(name string, properties map[string]api.ToolProperty) api.Tool {
	return api.Tool{Type: "function", Function: api.ToolFunction{
		Name: name,
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: testPropsMap(properties),
		},
	}}
}

func glimmerTestATEM(name, parameters string) string {
	return `<atem:function_calls>
<atem:invoke name="` + name + `">
` + parameters + `</atem:invoke>
</atem:function_calls>`
}

func TestGlimmerParserFinalAnswer(t *testing.T) {
	p := &GlimmerParser{}
	p.Init(nil, nil, nil)
	content, thinking, calls, err := p.Add(` to=user<|message|>Hello`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Hello" || thinking != "" || len(calls) != 0 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func TestGlimmerParserSelfRecipientIsThinking(t *testing.T) {
	p := &GlimmerParser{}
	p.Init(nil, nil, nil)
	content, thinking, calls, err := p.Add(` to=self<|message|>Check the facts.`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "Check the facts." || len(calls) != 0 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func TestGlimmerParserSuppressesThinking(t *testing.T) {
	p := &GlimmerParser{}
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := p.Add(
		` to=self<|message|>Check the facts.<|eom|><|start|>assistant to=user<|message|>Answer<|eot|>`,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Answer" || thinking != "" || len(calls) != 0 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func TestGlimmerParserStreamingATEMAtEveryBoundary(t *testing.T) {
	tool := glimmerTestTool("get_weather", map[string]api.ToolProperty{
		"city": {Type: api.PropertyType{"string"}},
	})
	input := ` to=get_weather<|message|>` + glimmerTestATEM(
		"get_weather",
		`<atem:parameter name="city">SF</atem:parameter>
`,
	) + `<|eot|>`

	for split := 0; split <= len(input); split++ {
		p := &GlimmerParser{}
		p.Init([]api.Tool{tool}, nil, nil)

		var content, thinking string
		var calls []api.ToolCall
		for i, chunk := range []string{input[:split], input[split:]} {
			gotContent, gotThinking, gotCalls, err := p.Add(chunk, i == 1)
			if err != nil {
				t.Fatalf("split %d: %v", split, err)
			}
			content += gotContent
			thinking += gotThinking
			calls = append(calls, gotCalls...)
		}

		if content != "" || thinking != "" || len(calls) != 1 {
			t.Fatalf("split %d: content=%q thinking=%q calls=%v", split, content, thinking, calls)
		}
		if calls[0].Function.Name != "get_weather" || calls[0].Function.Index != 0 {
			t.Fatalf("split %d: unexpected call: %#v", split, calls[0])
		}
		if city, ok := calls[0].Function.Arguments.Get("city"); !ok || city != "SF" {
			t.Fatalf("split %d: city = %#v, %v; want SF", split, city, ok)
		}
	}
}

func TestGlimmerParserATEMValueTypesAndWhitespace(t *testing.T) {
	tool := glimmerTestTool("tools.run", map[string]api.ToolProperty{
		"text":    {Type: api.PropertyType{"string"}},
		"count":   {Type: api.PropertyType{"integer"}},
		"enabled": {Type: api.PropertyType{"boolean"}},
		"items":   {Type: api.PropertyType{"array"}},
		"config":  {Type: api.PropertyType{"object"}},
		"choice": {AnyOf: []api.ToolProperty{
			{Type: api.PropertyType{"null"}},
			{Type: api.PropertyType{"string"}},
		}},
	})
	input := ` to=tools.run<|message|>` + glimmerTestATEM(
		"tools.run",
		`<atem:parameter name="text"> keep  spaces </atem:parameter>
<atem:parameter name="count">3</atem:parameter>
<atem:parameter name="enabled">true</atem:parameter>
<atem:parameter name="items">["one", "two"]</atem:parameter>
<atem:parameter name="config">{"mode":"fast"}</atem:parameter>
<atem:parameter name="choice">null</atem:parameter>
`,
	)

	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	content, thinking, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 1 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}

	want := map[string]any{
		"text":    " keep  spaces ",
		"count":   3,
		"enabled": true,
		"items":   []any{"one", "two"},
		"config":  map[string]any{"mode": "fast"},
		"choice":  nil,
	}
	for name, expected := range want {
		got, ok := calls[0].Function.Arguments.Get(name)
		if !ok || !reflect.DeepEqual(got, expected) {
			t.Errorf("%s = %#v, %v; want %#v", name, got, ok, expected)
		}
	}
}

func TestGlimmerParserATEMValueMayContainControlTokenText(t *testing.T) {
	tool := glimmerTestTool("echo", map[string]api.ToolProperty{
		"text": {Type: api.PropertyType{"string"}},
	})
	value := `literal <|eot|> and <|start|>assistant to=user<|message|> text`
	input := ` to=echo<|message|>` + glimmerTestATEM(
		"echo",
		`<atem:parameter name="text">`+value+`</atem:parameter>
`,
	) + `<|eot|>`

	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	content, thinking, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 1 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
	if got, ok := calls[0].Function.Arguments.Get("text"); !ok || got != value {
		t.Fatalf("text = %#v, %v; want %q", got, ok, value)
	}
}

func TestGlimmerParserMultipleMessagesAndIndices(t *testing.T) {
	first := glimmerTestTool("first", map[string]api.ToolProperty{
		"x": {Type: api.PropertyType{"integer"}},
	})
	second := glimmerTestTool("second", nil)
	input := " to=self<|message|>Think<|eom|>" +
		"<|start|>assistant to=first<|message|>" +
		glimmerTestATEM("first", `<atem:parameter name="x">1</atem:parameter>
`) + "<|eom|>" +
		"<|start|>assistant to=second<|message|>" +
		glimmerTestATEM("second", "") + "<|eot|>"

	for split := 0; split <= len(input); split++ {
		p := &GlimmerParser{}
		p.Init([]api.Tool{first, second}, nil, nil)

		var content, thinking string
		var calls []api.ToolCall
		for i, chunk := range []string{input[:split], input[split:]} {
			gotContent, gotThinking, gotCalls, err := p.Add(chunk, i == 1)
			if err != nil {
				t.Fatalf("split %d: %v", split, err)
			}
			content += gotContent
			thinking += gotThinking
			calls = append(calls, gotCalls...)
		}
		if content != "" || thinking != "Think" || len(calls) != 2 {
			t.Fatalf("split %d: content=%q thinking=%q calls=%v", split, content, thinking, calls)
		}
		if calls[0].Function.Name != "first" || calls[0].Function.Index != 0 {
			t.Fatalf("split %d: first call = %#v", split, calls[0])
		}
		if calls[1].Function.Name != "second" || calls[1].Function.Index != 1 {
			t.Fatalf("split %d: second call = %#v", split, calls[1])
		}
	}
}

// TestGlimmerParserStrayMessageTagInInvokeName covers the observed stress
// failure where the model fumbles a <|message|> boundary token into the
// invoke name (`name="read<|message|>">`), echoing the header form
// `to=read<|message|>`. The call must parse as the intended tool.
func TestGlimmerParserStrayMessageTagInInvokeName(t *testing.T) {
	tool := glimmerTestTool("read", map[string]api.ToolProperty{
		"path": {Type: api.PropertyType{"string"}},
	})
	input := ` to=read<|message|>` + glimmerTestATEM(
		"read<|message|>",
		`<atem:parameter name="path">go.mod</atem:parameter>
`,
	) + `<|eot|>`

	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	content, thinking, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" || len(calls) != 1 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
	if calls[0].Function.Name != "read" {
		t.Fatalf("call name = %q, want %q", calls[0].Function.Name, "read")
	}
	if got, ok := calls[0].Function.Arguments.Get("path"); !ok || got != "go.mod" {
		t.Fatalf("path = %#v, %v; want %q", got, ok, "go.mod")
	}
}

// TestGlimmerParserMessageTagReplacesInvokeTerminator covers the fleet-observed
// stress failure where the model emits a <|message|> boundary token in place
// of the `">` terminator of the invoke name (`name="read<|message|><atem:parameter ...`).
// The name must recover as the text before the tag and parameter parsing must
// resume at the parameter element.
func TestGlimmerParserMessageTagReplacesInvokeTerminator(t *testing.T) {
	tool := glimmerTestTool("read", map[string]api.ToolProperty{
		"path": {Type: api.PropertyType{"string"}},
	})

	for name, invoke := range map[string]string{
		"terminator replaced": `read<|message|><atem:parameter name="path">go.mod</atem:parameter>
`,
		"tag mid-name": `re<|message|>ad"><atem:parameter name="path">go.mod</atem:parameter>
`,
		"repeated tags": `read<|message|><|message|>"><atem:parameter name="path">go.mod</atem:parameter>
`,
	} {
		t.Run(name, func(t *testing.T) {
			input := ` to=read<|message|><atem:function_calls>
<atem:invoke name="` + invoke + `</atem:invoke>
</atem:function_calls><|eot|>`

			p := &GlimmerParser{}
			p.Init([]api.Tool{tool}, nil, nil)
			content, thinking, calls, err := p.Add(input, true)
			if err != nil {
				t.Fatal(err)
			}
			if content != "" || thinking != "" || len(calls) != 1 {
				t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
			}
			if calls[0].Function.Name != "read" {
				t.Fatalf("call name = %q, want %q", calls[0].Function.Name, "read")
			}
			if got, ok := calls[0].Function.Arguments.Get("path"); !ok || got != "go.mod" {
				t.Fatalf("path = %#v, %v; want %q", got, ok, "go.mod")
			}
		})
	}
}

// TestGlimmerParserMessageTagInParamValueUntouched pins the non-recovery side:
// a literal <|message|> inside a parameter value sits after the name
// terminator and must be preserved verbatim, not treated as a fumble.
func TestGlimmerParserMessageTagInParamValueUntouched(t *testing.T) {
	tool := glimmerTestTool("echo", map[string]api.ToolProperty{
		"text": {Type: api.PropertyType{"string"}},
	})
	value := `keep <|message|> literal`
	input := ` to=echo<|message|>` + glimmerTestATEM(
		"echo",
		`<atem:parameter name="text">`+value+`</atem:parameter>
`,
	) + `<|eot|>`

	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls = %v, want 1", calls)
	}
	if got, ok := calls[0].Function.Arguments.Get("text"); !ok || got != value {
		t.Fatalf("text = %#v, %v; want %q", got, ok, value)
	}
}

// TestGlimmerParserNamespacedSelfReference covers the observed stress failure
// where the model addresses an undotted tool through its own derived
// namespace (`read.read` for a tool declared as `read`) — the chat template
// advertises recipients "<ns>.*" with ns = first dot-component of each
// declared name, so this form is in-protocol and must resolve.
func TestGlimmerParserNamespacedSelfReference(t *testing.T) {
	tool := glimmerTestTool("read", map[string]api.ToolProperty{
		"path": {Type: api.PropertyType{"string"}},
	})
	params := `<atem:parameter name="path">go.mod</atem:parameter>
`
	for _, tc := range []struct{ recipient, invoke string }{
		{"read", "read.read"},
		{"read.read", "read.read"},
		{"read.read", "read"},
	} {
		p := &GlimmerParser{}
		p.Init([]api.Tool{tool}, nil, nil)
		input := ` to=` + tc.recipient + `<|message|>` + glimmerTestATEM(tc.invoke, params) + `<|eot|>`
		content, thinking, calls, err := p.Add(input, true)
		if err != nil {
			t.Fatalf("recipient=%q invoke=%q: %v", tc.recipient, tc.invoke, err)
		}
		if content != "" || thinking != "" || len(calls) != 1 {
			t.Fatalf("recipient=%q invoke=%q: content=%q thinking=%q calls=%v", tc.recipient, tc.invoke, content, thinking, calls)
		}
		if calls[0].Function.Name != "read" {
			t.Fatalf("recipient=%q invoke=%q: call name = %q, want %q", tc.recipient, tc.invoke, calls[0].Function.Name, "read")
		}
	}
}

// TestGlimmerParserNamespaceResolutionStaysStrict pins the limits of namespace
// resolution: undeclared namespaces do not resolve, and dotted declared names
// match exactly.
func TestGlimmerParserNamespaceResolutionStaysStrict(t *testing.T) {
	read := glimmerTestTool("read", nil)
	repoRead := glimmerTestTool("repo.read", nil)
	tools := map[string]api.Tool{"read": read, "repo.read": repoRead}

	for _, tc := range []struct {
		name, want string
		ok         bool
	}{
		{"read", "read", true},
		{"read.read", "read", true},
		{"repo.read", "repo.read", true},
		{"functions.read", "functions.read", false},
		{"write.write", "write.write", false},
		// double-prefix through the declared name's own namespace still
		// resolves: ns "repo" + declared "repo.read"
		{"repo.repo.read", "repo.read", true},
	} {
		got, ok := glimmerResolveToolName(tools, tc.name)
		if got != tc.want || ok != tc.ok {
			t.Errorf("glimmerResolveToolName(%q) = %q, %v; want %q, %v", tc.name, got, ok, tc.want, tc.ok)
		}
	}
}

func TestGlimmerParserRejectsMismatchedATEMInvoke(t *testing.T) {
	tool := glimmerTestTool("get_weather", nil)
	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	_, _, _, err := p.Add(
		` to=get_weather<|message|>`+glimmerTestATEM("read_file", ""),
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("error = %v, want recipient mismatch", err)
	}
}

func TestGlimmerParserRejectsMalformedATEM(t *testing.T) {
	tool := glimmerTestTool("get_weather", nil)
	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	_, _, _, err := p.Add(` to=get_weather<|message|>{"city":"SF"}`, true)
	if err == nil || !strings.Contains(err.Error(), "function_calls wrapper") {
		t.Fatalf("error = %v, want missing wrapper", err)
	}
}

// TestGlimmerParserFumbledRecipientToolCallFallback covers the observed failure
// mode where the model emits a well-formed ATEM block for a declared tool but
// fumbles the recipient header — omitting it, addressing the namespace or the
// user, or naming the wrapper element. The block must come back as the tool
// call named by the invoke element, never as raw XML content.
func TestGlimmerParserFumbledRecipientToolCallFallback(t *testing.T) {
	atem := glimmerTestATEM("muse.bash", `<atem:parameter name="command">pwd && ls -la</atem:parameter>
`)
	cases := map[string]struct {
		input    string
		thinking string
		content  string
	}{
		"namespace recipient": {input: ` to=muse<|message|>` + atem + `<|eom|>`},
		"missing recipient":   {input: `<|message|>` + atem + `<|eom|>`},
		"user recipient":      {input: ` to=user<|message|>` + atem + `<|eot|>`},
		"wrapper as recipient": {
			input: ` to=atem:function_calls<|message|>` + atem + `<|eom|>`,
		},
		"after thinking": {
			input:    ` to=self<|message|>let me look<|eom|><|start|>assistant<|message|>` + atem + `<|eom|>`,
			thinking: "let me look",
		},
		"implicit boundary": {
			input:   `<|message|>` + atem + `<|start|>assistant to=user<|message|>Done<|eot|>`,
			content: "Done",
		},
	}
	tools := []api.Tool{glimmerTestTool("muse.bash", map[string]api.ToolProperty{
		"command": {Type: api.PropertyType{"string"}},
	})}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := &GlimmerParser{}
			p.Init(tools, nil, nil)
			content, thinking, calls, err := p.Add(tc.input, true)
			if err != nil {
				t.Fatal(err)
			}
			if content != tc.content || thinking != tc.thinking || len(calls) != 1 {
				t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
			}
			if calls[0].Function.Name != "muse.bash" {
				t.Fatalf("call name = %q, want muse.bash", calls[0].Function.Name)
			}
			if command, ok := calls[0].Function.Arguments.Get("command"); !ok || command != "pwd && ls -la" {
				t.Fatalf("command = %#v, %v; want pwd && ls -la", command, ok)
			}
		})
	}
}

// TestGlimmerParserFumbledRecipientStreaming re-runs the fallback at every chunk
// boundary: the held block must never leak partial XML into the content
// stream regardless of where the model output is split.
func TestGlimmerParserFumbledRecipientStreaming(t *testing.T) {
	tools := []api.Tool{glimmerTestTool("muse.bash", map[string]api.ToolProperty{
		"command": {Type: api.PropertyType{"string"}},
	})}
	input := `<|message|>` + glimmerTestATEM("muse.bash", `<atem:parameter name="command">pwd</atem:parameter>
`) + `<|eom|>`

	for split := 0; split <= len(input); split++ {
		p := &GlimmerParser{}
		p.Init(tools, nil, nil)

		var content, thinking string
		var calls []api.ToolCall
		for i, chunk := range []string{input[:split], input[split:]} {
			gotContent, gotThinking, gotCalls, err := p.Add(chunk, i == 1)
			if err != nil {
				t.Fatalf("split %d: %v", split, err)
			}
			content += gotContent
			thinking += gotThinking
			calls = append(calls, gotCalls...)
		}

		if content != "" || thinking != "" || len(calls) != 1 {
			t.Fatalf("split %d: content=%q thinking=%q calls=%v", split, content, thinking, calls)
		}
		if calls[0].Function.Name != "muse.bash" {
			t.Fatalf("split %d: call name = %q", split, calls[0].Function.Name)
		}
	}
}

// The fallback must not fire for content that merely resembles a tool call:
// blocks for undeclared tools, blocks embedded in surrounding prose, and
// sessions with no tools at all stay ordinary content.
func TestGlimmerParserContentThatResemblesToolCallStaysContent(t *testing.T) {
	atem := glimmerTestATEM("muse.bash", `<atem:parameter name="command">pwd</atem:parameter>
`)
	declared := []api.Tool{glimmerTestTool("muse.bash", map[string]api.ToolProperty{
		"command": {Type: api.PropertyType{"string"}},
	})}
	cases := map[string]struct {
		tools   []api.Tool
		input   string
		content string
	}{
		"undeclared tool": {
			tools:   declared,
			input:   ` to=user<|message|>` + glimmerTestATEM("other.run", "") + `<|eot|>`,
			content: glimmerTestATEM("other.run", ""),
		},
		"leading prose": {
			tools: declared,
			input: ` to=user<|message|>Running:
` + atem + `<|eot|>`,
			content: "Running:\n" + atem,
		},
		"trailing prose": {
			tools: declared,
			input: ` to=user<|message|>` + atem + `
Done.<|eot|>`,
			content: atem + "\nDone.",
		},
		"no tools declared": {
			tools:   nil,
			input:   ` to=user<|message|>` + atem + `<|eot|>`,
			content: atem,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := &GlimmerParser{}
			p.Init(tc.tools, nil, nil)
			content, thinking, calls, err := p.Add(tc.input, true)
			if err != nil {
				t.Fatal(err)
			}
			if content != tc.content || thinking != "" || len(calls) != 0 {
				t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
			}
		})
	}
}

func TestGlimmerParserUndeclaredRecipientRemainsContent(t *testing.T) {
	p := &GlimmerParser{}
	p.Init(nil, nil, nil)
	body := glimmerTestATEM("not_a_tool", "")
	content, thinking, calls, err := p.Add(` to=not_a_tool<|message|>`+body, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != body || thinking != "" || len(calls) != 0 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func TestGlimmerParserImplicitMessageBoundary(t *testing.T) {
	tool := glimmerTestTool("get_weather", map[string]api.ToolProperty{
		"city": {Type: api.PropertyType{"string"}},
	})
	input := " to=self<|message|>Think" +
		"<|start|>assistant to=get_weather<|message|>" +
		glimmerTestATEM("get_weather", `<atem:parameter name="city">SF</atem:parameter>
`) +
		"<|start|>assistant to=user<|message|>Done<|eot|>"

	p := &GlimmerParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	content, thinking, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Done" || thinking != "Think" || len(calls) != 1 {
		t.Fatalf("got content=%q thinking=%q calls=%v", content, thinking, calls)
	}
}

func TestGlimmerParserWithholdsSplitControlToken(t *testing.T) {
	p := &GlimmerParser{}
	p.Init(nil, nil, nil)
	content, _, _, err := p.Add(" to=user<|message|>Hello<|eo", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Hello" {
		t.Fatalf("first content = %q, want Hello", content)
	}
	content, _, _, err = p.Add("t|>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("control token leaked as content: %q", content)
	}
}

func TestGlimmerParserLiteralStartTokenInMessages(t *testing.T) {
	tests := []struct {
		name      string
		recipient string
		content   string
		thinking  string
	}{
		{name: "content", recipient: "user", content: "Use `<|start|>` here."},
		{name: "thinking", recipient: "self", thinking: "Use `<|start|>` here."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &GlimmerParser{}
			p.Init(nil, nil, nil)
			chunks := []string{" to=" + tt.recipient + "<|message|>Use `<", "|", "start", "|>` here."}
			var content, thinking string
			for i, chunk := range chunks {
				gotContent, gotThinking, calls, err := p.Add(chunk, i == len(chunks)-1)
				if err != nil {
					t.Fatal(err)
				}
				if len(calls) != 0 {
					t.Fatalf("chunk %d got calls=%v", i, calls)
				}
				content += gotContent
				thinking += gotThinking
			}
			if content != tt.content || thinking != tt.thinking {
				t.Fatalf("got content=%q thinking=%q", content, thinking)
			}
		})
	}
}

func FuzzGlimmerParser(f *testing.F) {
	tool := glimmerTestTool("echo", map[string]api.ToolProperty{
		"text": {Type: api.PropertyType{"string"}},
	})
	for _, seed := range []string{
		` to=user<|message|>Hello<|eot|>`,
		` to=self<|message|>Think<|eom|><|start|>assistant to=user<|message|>Done<|eot|>`,
		` to=echo<|message|>` + glimmerTestATEM("echo", `<atem:parameter name="text">value</atem:parameter>`),
		` to=echo<|message|><atem:function_calls><atem:invoke name="other">`,
		`literal <|start|> and <|message|> text`,
	} {
		f.Add(seed, uint(0))
		f.Add(seed, uint(len(seed)/2))
	}

	f.Fuzz(func(t *testing.T, input string, split uint) {
		p := &GlimmerParser{}
		p.Init([]api.Tool{tool}, nil, nil)

		at := 0
		if len(input) > 0 {
			at = int(split % uint(len(input)+1))
		}

		var calls []api.ToolCall
		for i, chunk := range []string{input[:at], input[at:]} {
			_, _, got, err := p.Add(chunk, i == 1)
			if err != nil {
				return
			}
			calls = append(calls, got...)
		}

		lastIndex := -1
		for _, call := range calls {
			if call.Function.Name != "echo" {
				t.Fatalf("undeclared tool call emitted: %#v", call)
			}
			if call.Function.Index <= lastIndex {
				t.Fatalf("tool call indices are not increasing: %#v", calls)
			}
			lastIndex = call.Function.Index
		}
	})
}
