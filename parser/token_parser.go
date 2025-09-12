package parser

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

type TokenParserType int

const (
	TokenParserTypeDefault TokenParserType = iota
	TokenParserTypeHarmony
)

type TokenParser struct {
	messageHandler MessageHandler
	parserEngine   ParserInternals
	toolParser     ToolParser
	lastToken      string
	tokenRepeat    int
	repeatLimit    int
}

const defaultTokenRepeatLimit = 30

type MessageHandler interface {
	AddContent(token string) (content, thinking string, toolContent string)
}

type ParserInternals interface {
	AddImplicitStartOrPrefill(prefillString string)
	ConstraintsAllowed() bool
}

type ToolParser interface {
	Add(token string)
	Drain() (toolName *string, toolContent string)
}

// Default implementation for the TokenParser interface as a no-op passthrough
type defaultMessageHandler struct{}

func (defaultMessageHandler) AddContent(token string) (string, string, string) {
	return token, "", ""
}

type defaultEngine struct{}

func (defaultEngine) AddImplicitStartOrPrefill(prefillString string) {}

func (defaultEngine) ConstraintsAllowed() bool {
	return true
}

type defaultToolParser struct{}

func (defaultToolParser) Add(token string) {}

func (defaultToolParser) Drain() (*string, string) { return nil, "" }

func NewTokenParser(parserType TokenParserType, prefillString string) TokenParser {
	switch parserType {
	case TokenParserTypeHarmony:
		harmonyMessageHandler := harmony.NewHarmonyMessageHandler()
		harmonyMessageHandler.HarmonyParser.AddImplicitStartOrPrefill(prefillString)
		return TokenParser{
			messageHandler: harmonyMessageHandler,
			parserEngine:   harmonyMessageHandler.HarmonyParser,
			toolParser:     harmonyMessageHandler.ToolParser,
			repeatLimit:    defaultTokenRepeatLimit,
		}

	default:
		return TokenParser{
			messageHandler: defaultMessageHandler{},
			parserEngine:   defaultEngine{},
			toolParser:     defaultToolParser{},
			repeatLimit:    30,
		}
	}
}

func (p *TokenParser) AddContent(token string) (string, string, error) {
	if p.repeatLimitReached(token) {
		return "", "", errors.New("token repeat limit reached")
	}
	content, thinking, toolContent := p.messageHandler.AddContent(token)
	p.toolParser.Add(toolContent)
	return content, thinking, nil
}

// repeatLimitReached updates repeat counters and returns true if the repeat limit is reached.
func (p *TokenParser) repeatLimitReached(token string) bool {
	if p == nil {
		return false
	}
	trimmed := strings.TrimSpace(token)
	if trimmed == p.lastToken {
		p.tokenRepeat++
	} else {
		p.tokenRepeat = 0
	}
	p.lastToken = trimmed

	return p.tokenRepeat >= p.repeatLimit
}

func (p *TokenParser) ConstraintsAllowed() bool {
	return p.parserEngine.ConstraintsAllowed()
}

// TODO: update to work with multiple toolcalls - unmarshalling should also happen on parser level
func (p *TokenParser) Drain() []api.ToolCall {
	toolName, toolContent := p.toolParser.Drain()
	if toolName != nil {
		*toolName = strings.TrimPrefix(*toolName, "functions.")
		var args api.ToolCallFunctionArguments
		if err := json.Unmarshal([]byte(toolContent), &args); err != nil {
			return nil
		}
		return []api.ToolCall{
			{
				Function: api.ToolCallFunction{
					Name:      *toolName,
					Arguments: args,
				},
			},
		}
	}
	return nil
}
