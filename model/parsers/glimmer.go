package parsers

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	glimmerStartTag      = "<|start|>"
	glimmerMessageTag    = "<|message|>"
	glimmerEndMessageTag = "<|eom|>"
	glimmerEndTurnTag    = "<|eot|>"

	glimmerAssistantHeaderPrefix = glimmerStartTag + "assistant"

	glimmerATEMCallsOpen   = "<atem:function_calls>"
	glimmerATEMCallsClose  = "</atem:function_calls>"
	glimmerATEMInvokeOpen  = `<atem:invoke name="`
	glimmerATEMInvokeClose = "</atem:invoke>"
	glimmerATEMParamOpen   = `<atem:parameter name="`
	glimmerATEMParamClose  = "</atem:parameter>"
)

type glimmerParserState int

const (
	glimmerParserHeader glimmerParserState = iota
	glimmerParserContent
	glimmerParserThinking
	glimmerParserTool
)

type GlimmerParser struct {
	state        glimmerParserState
	buffer       strings.Builder
	recipient    string
	tools        map[string]api.Tool
	callIndex    int
	emitThinking bool

	// contentStreamed records whether any of the current message's body has
	// already been emitted as content. The fumbled-recipient tool-call
	// fallback only applies while nothing has streamed, so a message either
	// becomes a tool call or streams as content — never half of each.
	contentStreamed bool
}

func (p *GlimmerParser) HasToolSupport() bool     { return true }
func (p *GlimmerParser) HasThinkingSupport() bool { return true }

func (p *GlimmerParser) PreservedTokens() []string {
	return []string{glimmerStartTag, glimmerMessageTag, glimmerEndMessageTag, glimmerEndTurnTag}
}

func (p *GlimmerParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = glimmerParserHeader
	p.buffer.Reset()
	p.recipient = ""
	p.tools = glimmerToolsByName(tools)
	p.callIndex = 0
	p.emitThinking = thinkValue == nil || thinkValue.Bool()
	p.contentStreamed = false
	return tools
}

func (p *GlimmerParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	var contentSB, thinkingSB strings.Builder

	for {
		if p.state == glimmerParserHeader {
			progress, fallback := p.consumeHeader(done)
			contentSB.WriteString(fallback)
			if !progress {
				break
			}
			continue
		}

		progress, body, complete := p.consumeBody(done)
		switch p.state {
		case glimmerParserContent:
			if call, ok := p.contentToolCallFallback(body, complete); ok {
				calls = append(calls, call)
			} else if body != "" {
				contentSB.WriteString(body)
				p.contentStreamed = true
			}
		case glimmerParserThinking:
			if p.emitThinking {
				thinkingSB.WriteString(body)
			}
		case glimmerParserTool:
			if complete {
				call, parseErr := p.parseToolCall(body)
				if parseErr != nil {
					return contentSB.String(), thinkingSB.String(), calls, parseErr
				}
				calls = append(calls, call)
			}
		}
		if complete {
			p.state = glimmerParserHeader
			p.recipient = ""
			p.contentStreamed = false
		}
		if !progress {
			break
		}
	}

	return contentSB.String(), thinkingSB.String(), calls, nil
}

func (p *GlimmerParser) consumeHeader(done bool) (progress bool, fallback string) {
	acc := p.buffer.String()
	if strings.HasPrefix(acc, glimmerAssistantHeaderPrefix) {
		acc = acc[len(glimmerAssistantHeaderPrefix):]
		p.buffer.Reset()
		p.buffer.WriteString(acc)
	} else if strings.HasPrefix(glimmerAssistantHeaderPrefix, acc) && strings.HasPrefix(acc, "<") && !done {
		return false, ""
	}

	idx := strings.Index(acc, glimmerMessageTag)
	if idx < 0 {
		if !done {
			return false, ""
		}
		p.buffer.Reset()
		return acc != "", acc
	}

	header := strings.TrimSpace(acc[:idx])
	p.buffer.Reset()
	p.buffer.WriteString(acc[idx+len(glimmerMessageTag):])
	p.recipient = strings.TrimSpace(strings.TrimPrefix(header, "to="))
	p.setStateForRecipient()
	return true, ""
}

func (p *GlimmerParser) setStateForRecipient() {
	_, isTool := glimmerResolveToolName(p.tools, p.recipient)
	switch {
	case p.recipient == "self":
		p.state = glimmerParserThinking
	case isTool:
		p.state = glimmerParserTool
	default:
		p.state = glimmerParserContent
	}
}

func (p *GlimmerParser) consumeBody(done bool) (progress bool, body string, complete bool) {
	acc := p.buffer.String()
	hold := p.state == glimmerParserTool || p.holdContent(acc)
	controlStart := 0
	if hold {
		if close := strings.Index(acc, glimmerATEMCallsClose); close >= 0 {
			controlStart = close + len(glimmerATEMCallsClose)
		} else {
			controlStart = len(acc)
		}
	}

	idx, markerLen := glimmerBodyTerminator(acc[controlStart:])
	if idx >= 0 {
		idx += controlStart
	}
	implicitIdx, waitForImplicit := p.implicitHeaderStart(acc, controlStart, done)
	if implicitIdx >= 0 && !waitForImplicit && (idx < 0 || implicitIdx < idx) {
		body = acc[:implicitIdx]
		p.buffer.Reset()
		p.buffer.WriteString(acc[implicitIdx:])
		return true, body, true
	}
	if idx >= 0 {
		body = acc[:idx]
		p.buffer.Reset()
		p.buffer.WriteString(acc[idx+markerLen:])
		return true, body, true
	}
	if done {
		p.buffer.Reset()
		return acc != "" || p.state == glimmerParserTool, acc, true
	}
	if hold {
		return false, "", false
	}
	if waitForImplicit {
		body = acc[:implicitIdx]
		p.buffer.Reset()
		p.buffer.WriteString(acc[implicitIdx:])
		return body != "", body, false
	}

	keep := 0
	for _, marker := range []string{glimmerEndMessageTag, glimmerEndTurnTag} {
		keep = max(keep, overlap(acc, marker))
	}
	if len(acc) == keep {
		return false, "", false
	}
	body = acc[:len(acc)-keep]
	p.buffer.Reset()
	p.buffer.WriteString(acc[len(acc)-keep:])
	return body != "", body, false
}

func (p *GlimmerParser) implicitHeaderStart(s string, search int, done bool) (idx int, wait bool) {
	for {
		i := strings.Index(s[search:], glimmerStartTag)
		if i < 0 {
			break
		}
		i += search

		switch p.assistantHeaderStatus(s[i:], done) {
		case glimmerHeaderValid:
			return i, false
		case glimmerHeaderPending:
			return i, true
		}

		search = i + len(glimmerStartTag)
	}

	if done {
		return -1, false
	}
	if keep := overlap(s[search:], glimmerAssistantHeaderPrefix); keep > 0 {
		return len(s) - keep, true
	}
	return -1, false
}

type glimmerHeaderStatus int

const (
	glimmerHeaderInvalid glimmerHeaderStatus = iota
	glimmerHeaderPending
	glimmerHeaderValid
)

func (p *GlimmerParser) assistantHeaderStatus(s string, done bool) glimmerHeaderStatus {
	if len(s) < len(glimmerAssistantHeaderPrefix) {
		if !done && strings.HasPrefix(glimmerAssistantHeaderPrefix, s) {
			return glimmerHeaderPending
		}
		return glimmerHeaderInvalid
	}
	if !strings.HasPrefix(s, glimmerAssistantHeaderPrefix) {
		return glimmerHeaderInvalid
	}

	rest := s[len(glimmerAssistantHeaderPrefix):]
	if strings.HasPrefix(rest, glimmerMessageTag) {
		return glimmerHeaderValid
	}
	if !done && strings.HasPrefix(glimmerMessageTag, rest) {
		return glimmerHeaderPending
	}
	if rest == "" {
		if done {
			return glimmerHeaderInvalid
		}
		return glimmerHeaderPending
	}

	const recipientPrefix = " to="
	if len(rest) < len(recipientPrefix) {
		if !done && strings.HasPrefix(recipientPrefix, rest) {
			return glimmerHeaderPending
		}
		return glimmerHeaderInvalid
	}
	if !strings.HasPrefix(rest, recipientPrefix) {
		return glimmerHeaderInvalid
	}

	recipientAndAfter := rest[len(recipientPrefix):]
	messageIdx := strings.Index(recipientAndAfter, glimmerMessageTag)
	if messageIdx >= 0 {
		if p.validAssistantRecipient(recipientAndAfter[:messageIdx]) {
			return glimmerHeaderValid
		}
		return glimmerHeaderInvalid
	}
	if done {
		return glimmerHeaderInvalid
	}
	if recipientAndAfter == "" {
		return glimmerHeaderPending
	}
	if strings.ContainsAny(recipientAndAfter, " \t\r\n") {
		return glimmerHeaderInvalid
	}
	if lt := strings.IndexByte(recipientAndAfter, '<'); lt >= 0 {
		if strings.HasPrefix(glimmerMessageTag, recipientAndAfter[lt:]) {
			return glimmerHeaderPending
		}
		return glimmerHeaderInvalid
	}
	return glimmerHeaderPending
}

func (p *GlimmerParser) validAssistantRecipient(recipient string) bool {
	return recipient == "self" || recipient == "user" || p.tools[recipient].Function.Name != ""
}

func glimmerBodyTerminator(s string) (int, int) {
	idx, markerLen := -1, 0
	for _, marker := range []string{glimmerEndMessageTag, glimmerEndTurnTag} {
		if i := strings.Index(s, marker); i >= 0 && (idx < 0 || i < idx) {
			idx = i
			markerLen = len(marker)
		}
	}
	return idx, markerLen
}

// holdContent reports whether the content accumulated so far may be a fumbled
// tool call: a message that opens with the ATEM wrapper despite a non-tool
// recipient (the model sometimes omits the recipient, addresses the namespace
// or the user, or names the wrapper element itself). Such content is withheld
// from streaming until the message completes, so contentToolCallFallback can
// decide whether it is a tool call or ordinary content. Anything else streams
// immediately.
func (p *GlimmerParser) holdContent(acc string) bool {
	if p.state != glimmerParserContent || p.contentStreamed || len(p.tools) == 0 {
		return false
	}
	trimmed := strings.TrimLeft(acc, " \t\r\n")
	return strings.HasPrefix(glimmerATEMCallsOpen, trimmed) || strings.HasPrefix(trimmed, glimmerATEMCallsOpen)
}

// contentToolCallFallback recovers a tool call from a content-position message
// whose complete body is solely a well-formed ATEM block invoking a declared
// tool. The recipient header is authoritative when it names a tool; this
// fallback exists for the fumbled headers holdContent describes, where
// streaming the raw XML at the client is never the right answer. The function
// name is read from the invoke element, not the header.
func (p *GlimmerParser) contentToolCallFallback(body string, complete bool) (api.ToolCall, bool) {
	if !complete || p.contentStreamed || len(p.tools) == 0 {
		return api.ToolCall{}, false
	}
	trimmed := strings.TrimSpace(body)
	if !strings.HasPrefix(trimmed, glimmerATEMCallsOpen) || !strings.HasSuffix(trimmed, glimmerATEMCallsClose) {
		return api.ToolCall{}, false
	}
	resolved, ok := glimmerResolveToolName(p.tools, glimmerATEMInvokeName(trimmed))
	if !ok {
		return api.ToolCall{}, false
	}
	tool := p.tools[resolved]
	if _, args, err := parseGlimmerATEM(trimmed, tool); err == nil {
		return p.newToolCall(resolved, args), true
	}
	return api.ToolCall{}, false
}

// glimmerATEMInvokeName extracts the invoke element's function name from a body
// already known to carry the ATEM wrapper, or "" when the invoke is malformed.
func glimmerATEMInvokeName(body string) string {
	inner := strings.TrimSpace(body[len(glimmerATEMCallsOpen) : len(body)-len(glimmerATEMCallsClose)])
	if !strings.HasPrefix(inner, glimmerATEMInvokeOpen) {
		return ""
	}
	name, _, ok := strings.Cut(inner[len(glimmerATEMInvokeOpen):], `">`)
	if !ok {
		return ""
	}
	return glimmerTrimStrayMessageTag(name)
}

// glimmerTrimStrayMessageTag drops <|message|> boundary tokens the model
// occasionally fumbles into the invoke name (e.g. `name="read<|message|>">`),
// echoing the header form `to=read<|message|>`. Function names are
// identifiers, so the tag is never legitimate there; parameter values are
// left untouched (control-token text is preserved in values).
func glimmerTrimStrayMessageTag(name string) string {
	if !strings.Contains(name, glimmerMessageTag) {
		return name
	}
	slog.Warn("glimmer parser recovered stray message boundary token in ATEM invoke name", "name", name)
	return strings.ReplaceAll(name, glimmerMessageTag, "")
}

func (p *GlimmerParser) parseToolCall(body string) (api.ToolCall, error) {
	if p.recipient == "" {
		return api.ToolCall{}, fmt.Errorf("empty Glimmer function name")
	}

	recipient, ok := glimmerResolveToolName(p.tools, p.recipient)
	if !ok {
		return api.ToolCall{}, fmt.Errorf("undeclared Glimmer function %q", p.recipient)
	}
	tool := p.tools[recipient]

	name, args, err := parseGlimmerATEM(body, tool)
	if err != nil {
		return api.ToolCall{}, fmt.Errorf("parse Glimmer call to %s: %w", p.recipient, err)
	}
	if resolved, ok := glimmerResolveToolName(p.tools, name); !ok || resolved != recipient {
		return api.ToolCall{}, fmt.Errorf("Glimmer recipient %q does not match ATEM invoke %q", p.recipient, name)
	}
	return p.newToolCall(recipient, args), nil
}

// glimmerResolveToolName resolves a recipient or ATEM invoke name to a declared
// tool name. Exact matches win. The chat template derives a namespace from
// the first dot-component of each declared tool name and advertises
// recipients "<ns>.*" to the model, so an undotted tool `read` is
// legitimately addressable as `read.read`; resolve `ns.fn` to the declared
// `fn` when ns is fn's own derived namespace.
func glimmerResolveToolName(tools map[string]api.Tool, name string) (string, bool) {
	if tools[name].Function.Name != "" {
		return name, true
	}
	if ns, fn, ok := strings.Cut(name, "."); ok {
		if tools[fn].Function.Name != "" {
			fnNS, _, _ := strings.Cut(fn, ".")
			if ns == fnNS {
				return fn, true
			}
		}
	}
	return name, false
}

func glimmerToolsByName(tools []api.Tool) map[string]api.Tool {
	if len(tools) == 0 {
		return nil
	}

	byName := make(map[string]api.Tool, len(tools))
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Function.Name)
		if name == "" {
			continue
		}
		byName[name] = tool
	}
	return byName
}

func parseGlimmerATEM(body string, tool api.Tool) (string, api.ToolCallFunctionArguments, error) {
	body = strings.TrimSpace(body)
	if !strings.HasPrefix(body, glimmerATEMCallsOpen) || !strings.HasSuffix(body, glimmerATEMCallsClose) {
		return "", api.ToolCallFunctionArguments{}, fmt.Errorf("missing ATEM function_calls wrapper")
	}

	invoke := strings.TrimSpace(body[len(glimmerATEMCallsOpen) : len(body)-len(glimmerATEMCallsClose)])
	if !strings.HasPrefix(invoke, glimmerATEMInvokeOpen) || !strings.HasSuffix(invoke, glimmerATEMInvokeClose) {
		return "", api.ToolCallFunctionArguments{}, fmt.Errorf("missing ATEM invoke wrapper")
	}
	invoke = invoke[len(glimmerATEMInvokeOpen):]
	nameEnd := strings.Index(invoke, `">`)
	if nameEnd < 0 {
		return "", api.ToolCallFunctionArguments{}, fmt.Errorf("malformed ATEM invoke name")
	}
	name := glimmerTrimStrayMessageTag(invoke[:nameEnd])
	params := invoke[nameEnd+2 : len(invoke)-len(glimmerATEMInvokeClose)]
	params = strings.TrimPrefix(params, "\n")
	params = strings.TrimSuffix(params, "\n")

	args := api.NewToolCallFunctionArguments()
	for params != "" {
		if !strings.HasPrefix(params, glimmerATEMParamOpen) {
			return "", api.ToolCallFunctionArguments{}, fmt.Errorf("malformed ATEM parameter")
		}
		params = params[len(glimmerATEMParamOpen):]
		paramNameEnd := strings.Index(params, `">`)
		if paramNameEnd < 0 {
			return "", api.ToolCallFunctionArguments{}, fmt.Errorf("malformed ATEM parameter name")
		}
		paramName := params[:paramNameEnd]
		params = params[paramNameEnd+2:]
		valueEnd := strings.Index(params, glimmerATEMParamClose)
		if valueEnd < 0 {
			return "", api.ToolCallFunctionArguments{}, fmt.Errorf("unterminated ATEM parameter %q", paramName)
		}
		value := params[:valueEnd]
		params = params[valueEnd+len(glimmerATEMParamClose):]
		params = strings.TrimPrefix(params, "\n")

		var paramType api.PropertyType
		if tool.Function.Parameters.Properties != nil {
			if property, ok := tool.Function.Parameters.Properties.Get(paramName); ok {
				if len(property.AnyOf) > 0 {
					for _, option := range property.AnyOf {
						paramType = append(paramType, option.Type...)
					}
				} else {
					paramType = property.Type
				}
			}
		}
		args.Set(paramName, parseTypedToolValue(value, paramType))
	}

	return name, args, nil
}

func (p *GlimmerParser) newToolCall(name string, args api.ToolCallFunctionArguments) api.ToolCall {
	call := api.ToolCall{Function: api.ToolCallFunction{
		Name:      name,
		Arguments: args,
		Index:     p.callIndex,
	}}
	p.callIndex++
	return call
}
