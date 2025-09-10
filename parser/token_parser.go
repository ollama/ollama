package parser

import (
	"encoding/json"
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
	messageHandler  MessageHandler
	parserInternals ParserInternals
	toolParser      ToolParser
}

type MessageHandler interface {
	AddContent(token string) (content, thinking string, toolContent string)
}

type ParserInternals interface {
	AddImplicitStartOrPrefill(prefillString string)
}

type ToolParser interface {
	Add(token string)
	Drain() (toolName *string, toolContent string)
}

func NewTokenParser(parserType TokenParserType, prefillString string) *TokenParser {
	switch parserType {
	case TokenParserTypeHarmony:
		harmonyMessageHandler := harmony.NewHarmonyMessageHandler()
		harmonyMessageHandler.HarmonyParser.AddImplicitStartOrPrefill(prefillString)
		return &TokenParser{
			messageHandler:  harmonyMessageHandler,
			parserInternals: harmonyMessageHandler.HarmonyParser,
			toolParser:      harmonyMessageHandler.ToolParser,
		}

	default:
		return nil
	}
}

func (p *TokenParser) AddContent(token string) (string, string) {
	content, thinking, toolContent := p.messageHandler.AddContent(token)
	p.toolParser.Add(toolContent)
	return content, thinking
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
