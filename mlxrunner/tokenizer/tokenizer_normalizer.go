package tokenizer

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/text/unicode/norm"
)

type normalizerStage struct {
	nfc         bool
	literal     *string
	regex       *regexp.Regexp
	replacement string
}

func loadNormalizerSequence(data json.RawMessage) ([]normalizerStage, error) {
	var raw struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			String *string `json:"String"`
			Regex  *string `json:"Regex"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	switch raw.Type {
	case "Sequence":
		var result []normalizerStage
		for _, child := range raw.Normalizers {
			stages, err := loadNormalizerSequence(child)
			if err != nil {
				return nil, err
			}
			result = append(result, stages...)
		}
		return result, nil
	case "NFC":
		return []normalizerStage{{nfc: true}}, nil
	case "Replace":
		stage := normalizerStage{literal: raw.Pattern.String, replacement: raw.Content}
		if raw.Pattern.Regex != nil {
			pattern := regexp.MustCompile(`\\u([0-9A-Fa-f]{4})`).ReplaceAllString(*raw.Pattern.Regex, `\x{${1}}`)
			re, err := regexp.Compile(pattern)
			if err != nil {
				return nil, fmt.Errorf("normalizer regex: %w", err)
			}
			stage.regex = re
		} else if stage.literal == nil {
			return nil, fmt.Errorf("Replace normalizer requires a pattern")
		}
		return []normalizerStage{stage}, nil
	default:
		return nil, fmt.Errorf("unsupported sequence normalizer %q", raw.Type)
	}
}

func (s normalizerStage) apply(text string) string {
	if s.nfc {
		return norm.NFC.String(text)
	}
	if s.literal != nil {
		return strings.ReplaceAll(text, *s.literal, s.replacement)
	}
	return s.regex.ReplaceAllStringFunc(text, func(string) string { return s.replacement })
}
