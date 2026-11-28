package parsers

import (
	"encoding/json"
	"regexp"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

// OlalaParser parses OVHai Olala v1.0 output:
//
//	{reasoning}<|channel_end|>
//	<|channel_start|>final<|content|>{answer}<|channel_end|>
//	[<|channel_start|>tools<|content|><toolcalls><call id="ID"><name>fn</name><arguments>{json}</arguments></call>...</toolcalls><|channel_end|>]
//	<|im_end|>
//
// The prompt leaves the analysis channel open (or closes it when reasoning is
// disabled). analysis becomes thinking, final becomes content, and tools is
// parsed into tool calls; any other channel is dropped. Semantics follow the
// fork's common/parsers/olala.cpp.
type OlalaParser struct {
	state        olalaState
	buf          strings.Builder
	channelStart bool // at the start of a channel body: trim leading whitespace
	callIndex    int
	startState   olalaState
}

type olalaState int

const (
	olalaAnalysis olalaState = iota
	olalaBetween
	olalaFinal
	olalaTools
	olalaSkip
)

const (
	olalaPIMStart = "<|im_start|>"
	olalaPIMEnd   = "<|im_end|>"
	olalaPChStart = "<|channel_start|>"
	olalaPChEnd   = "<|channel_end|>"
	olalaPContent = "<|content|>"
)

var olalaCallRe = regexp.MustCompile(`(?s)<call\s+id="([^"]*)"\s*>\s*<name>\s*(.*?)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</call>`)

func (p *OlalaParser) HasToolSupport() bool     { return true }
func (p *OlalaParser) HasThinkingSupport() bool { return true }

func (p *OlalaParser) PreservedTokens() []string {
	return []string{olalaPChStart, olalaPChEnd, olalaPContent, olalaPIMStart, olalaPIMEnd}
}

func (p *OlalaParser) ThinkingClose() []string {
	if p.startState == olalaAnalysis {
		return []string{olalaPChEnd}
	}
	return nil
}

func (p *OlalaParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buf.Reset()
	p.callIndex = 0
	p.channelStart = true

	thinking := true
	if thinkValue != nil {
		switch v := thinkValue.Value.(type) {
		case bool:
			thinking = v
		case string:
			s := strings.ToLower(v)
			thinking = s != "disabled" && s != "none"
		}
	}

	switch {
	case lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != "":
		p.state = olalaFinal
		p.channelStart = false
	case lastMessage != nil && lastMessage.Role == "assistant":
		p.state = olalaAnalysis
		p.channelStart = lastMessage.Thinking == ""
	case thinking:
		p.state = olalaAnalysis
	default:
		p.state = olalaBetween
	}
	p.startState = p.state
	return tools
}

func (p *OlalaParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buf.WriteString(s)
	var contentB, thinkingB strings.Builder

	for {
		progressed := false
		buf := p.buf.String()

		switch p.state {
		case olalaAnalysis, olalaFinal:
			out := &thinkingB
			if p.state == olalaFinal {
				out = &contentB
			}
			if p.channelStart {
				buf = strings.TrimLeftFunc(buf, unicode.IsSpace)
				p.setBuf(buf)
				if buf == "" {
					break
				}
				p.channelStart = false
			}
			idx, marker := olalaFirstMarker(buf, olalaPChEnd, olalaPChStart, olalaPIMEnd)
			if idx >= 0 {
				out.WriteString(strings.TrimRightFunc(buf[:idx], unicode.IsSpace))
				rest := buf[idx:]
				if marker != olalaPChStart {
					rest = rest[len(marker):]
				}
				p.setBuf(rest)
				p.state = olalaBetween
				progressed = true
				break
			}
			if done {
				out.WriteString(buf)
				p.setBuf("")
				break
			}
			// hold back a possible partial marker and trailing whitespace
			hold := max(overlap(buf, olalaPChEnd), overlap(buf, olalaPChStart), overlap(buf, olalaPIMEnd))
			emit := buf[:len(buf)-hold]
			ws := trailingWhitespaceLen(emit)
			emit = emit[:len(emit)-ws]
			out.WriteString(emit)
			p.setBuf(buf[len(emit):])

		case olalaBetween:
			trimmed := strings.TrimLeftFunc(buf, unicode.IsSpace)
			if strings.HasPrefix(trimmed, olalaPIMEnd) {
				p.setBuf(trimmed[len(olalaPIMEnd):])
				progressed = true
				break
			}
			if strings.HasPrefix(trimmed, olalaPChEnd) {
				p.setBuf(trimmed[len(olalaPChEnd):])
				progressed = true
				break
			}
			if strings.HasPrefix(trimmed, olalaPChStart) {
				header := trimmed[len(olalaPChStart):]
				end := strings.Index(header, olalaPContent)
				if end < 0 {
					p.setBuf(trimmed)
					if done {
						p.setBuf("")
					}
					break
				}
				name := strings.TrimSpace(header[:end])
				p.setBuf(header[end+len(olalaPContent):])
				p.channelStart = true
				switch name {
				case "analysis":
					p.state = olalaAnalysis
				case "final":
					p.state = olalaFinal
				case "tools":
					p.state = olalaTools
				default:
					p.state = olalaSkip
				}
				progressed = true
				break
			}
			if trimmed == "" ||
				(!done && (strings.HasPrefix(olalaPChStart, trimmed) || strings.HasPrefix(olalaPIMEnd, trimmed) || strings.HasPrefix(olalaPChEnd, trimmed))) {
				p.setBuf(trimmed)
				break
			}
			// Text outside any channel: treat it as answer content.
			p.setBuf(trimmed)
			p.state = olalaFinal
			p.channelStart = false
			progressed = true

		case olalaTools, olalaSkip:
			idx, marker := olalaFirstMarker(buf, olalaPChEnd, olalaPIMEnd, olalaPChStart)
			if idx < 0 && !done {
				break
			}
			body := buf
			rest := ""
			if idx >= 0 {
				body = buf[:idx]
				rest = buf[idx:]
				if marker != olalaPChStart {
					rest = rest[len(marker):]
				}
			}
			if p.state == olalaTools {
				calls = append(calls, p.parseCalls(body)...)
			}
			p.setBuf(rest)
			p.state = olalaBetween
			progressed = idx >= 0
		}

		if !progressed {
			break
		}
	}

	return contentB.String(), thinkingB.String(), calls, nil
}

func (p *OlalaParser) setBuf(s string) {
	p.buf.Reset()
	p.buf.WriteString(s)
}

func (p *OlalaParser) parseCalls(body string) []api.ToolCall {
	var calls []api.ToolCall
	for _, m := range olalaCallRe.FindAllStringSubmatch(body, -1) {
		args := api.NewToolCallFunctionArguments()
		if strings.TrimSpace(m[3]) != "" {
			if err := json.Unmarshal([]byte(m[3]), &args); err != nil {
				continue
			}
		}
		calls = append(calls, api.ToolCall{
			ID: m[1],
			Function: api.ToolCallFunction{
				Index:     p.callIndex,
				Name:      m[2],
				Arguments: args,
			},
		})
		p.callIndex++
	}
	return calls
}

// olalaFirstMarker returns the index and value of the earliest marker in s.
func olalaFirstMarker(s string, markers ...string) (int, string) {
	best, which := -1, ""
	for _, m := range markers {
		if i := strings.Index(s, m); i >= 0 && (best < 0 || i < best) {
			best, which = i, m
		}
	}
	return best, which
}
