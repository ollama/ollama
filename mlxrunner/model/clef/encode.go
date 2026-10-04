package clef

import (
	"fmt"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

type tokenSpan struct{ start, end int }

type encodedQuestion struct {
	kind        int32
	instruction tokenSpan
	options     []tokenSpan
}

type encodedRecord struct {
	ids       []int32
	questions []encodedQuestion
}

// Clef's shared encoder supplies prefix, state, then schema and suffix segments.
// The prepared prefix includes image tokens, which count toward the context limit.
func encode(tok *tokenizer.Tokenizer, input llm.ScoreRequest, prefix []int32) (encodedRecord, error) {
	var record encodedRecord
	if len(input.Segments) < 3 {
		return record, fmt.Errorf("Clef requires prefix, state, and schema segments")
	}
	tokens := make([][]int32, len(input.Segments))
	tokens[0] = prefix
	total := len(prefix)
	for i := 1; i < len(tokens); i++ {
		tokens[i] = tok.Encode(input.Segments[i], false)
		total += len(tokens[i])
	}
	if total > input.MaxTokens {
		return record, fmt.Errorf("Clef input has %d tokens; context limit is %d", total, input.MaxTokens)
	}
	offsets := make([]int, len(tokens)+1)
	for i, segment := range tokens {
		record.ids = append(record.ids, segment...)
		offsets[i+1] = len(record.ids)
	}
	span := func(bounds [2]int) (tokenSpan, error) {
		if bounds[0] < 2 || bounds[0] >= bounds[1] || bounds[1] > len(tokens) {
			return tokenSpan{}, fmt.Errorf("Clef question span is outside the schema")
		}
		s := tokenSpan{offsets[bounds[0]], offsets[bounds[1]]}
		if s.start == s.end {
			return tokenSpan{}, fmt.Errorf("Clef question span has no tokens")
		}
		return s, nil
	}
	for _, field := range input.Fields {
		if field.Type < 0 || field.Type > 2 || len(field.Options) < 2 || len(field.Options) > 26 {
			return record, fmt.Errorf("invalid Clef question type or option count")
		}
		question, err := span(field.Question)
		if err != nil {
			return record, err
		}
		q := encodedQuestion{kind: int32(field.Type), instruction: question}
		for _, bounds := range field.Options {
			option, err := span(bounds)
			if err != nil {
				return record, err
			}
			q.options = append(q.options, option)
		}
		record.questions = append(record.questions, q)
	}
	return record, nil
}
