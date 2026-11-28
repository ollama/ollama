package pplxdecider

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
)

type locateSpan struct {
	Start int     `json:"start"`
	End   int     `json:"end"`
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

type locateAnswer struct {
	Type  string       `json:"type"`
	Spans []locateSpan `json:"spans"`
}

type tokenOffset struct {
	start int
	end   int
}

type locatePlan struct {
	prepared *model.PreparedRequest
	markers  []int32
	stateIDs []int32
	offsets  []tokenOffset
	rows     int
}

func byteRuneBoundaries(text string) (floor, ceil []int) {
	floor = make([]int, len(text)+1)
	ceil = make([]int, len(text)+1)
	characters := 0
	for offset := 0; offset < len(text); {
		_, width := utf8.DecodeRuneInString(text[offset:])
		floor[offset] = characters
		ceil[offset] = characters
		for j := offset + 1; j < offset+width; j++ {
			floor[j] = characters
			ceil[j] = characters + 1
		}
		offset += width
		characters++
	}
	floor[len(text)] = characters
	ceil[len(text)] = characters
	return floor, ceil
}

// The tokenizer composes text to NFC before encoding. Preserve offsets in the
// original state even when one encoded character came from multiple runes.
func normalizedRuneBoundaries(text string) (string, []int, []int) {
	normalized := norm.NFC.String(text)
	if normalized == text {
		floor, ceil := byteRuneBoundaries(text)
		return normalized, floor, ceil
	}
	floor := make([]int, len(normalized)+1)
	ceil := make([]int, len(normalized)+1)
	var iter norm.Iter
	iter.InitString(norm.NFC, text)
	sourceOffset, normalizedOffset, characters := 0, 0, 0
	for !iter.Done() {
		part := iter.Next()
		original := text[sourceOffset:iter.Pos()]
		nextCharacters := characters + utf8.RuneCountInString(original)
		// A composed token ending at this boundary maps to its base rune in
		// the publisher tokenizer. A token continuing past it still includes
		// the combining marks in the original state.
		visibleEnd := nextCharacters
		if last, _ := utf8.DecodeLastRune(part); !unicode.IsMark(last) {
			for rest := original; len(rest) > 0; {
				last, width := utf8.DecodeLastRuneInString(rest)
				if !unicode.IsMark(last) {
					break
				}
				visibleEnd--
				rest = rest[:len(rest)-width]
			}
		}
		floor[normalizedOffset] = characters
		for i := 1; i < len(part); i++ {
			floor[normalizedOffset+i] = characters
			ceil[normalizedOffset+i] = visibleEnd
		}
		normalizedOffset += len(part)
		floor[normalizedOffset] = nextCharacters
		ceil[normalizedOffset] = visibleEnd
		sourceOffset = iter.Pos()
		characters = nextCharacters
	}
	return normalized, floor, ceil
}

func (m *Model) locateContext(state, system string) ([]int32, []int32, []tokenOffset, error) {
	contextText, userStart, err := m.scoreContextV4(llm.ScoreRequest{State: state, System: system})
	if err != nil {
		return nil, nil, nil, err
	}
	stateBegin := userStart + len("State:\n")
	stateEnd := len(contextText)
	if stateBegin > stateEnd {
		return nil, nil, nil, fmt.Errorf("Perplexity Locate state is missing from its context")
	}
	escaped := contextText[stateBegin:stateEnd]
	if utf8.RuneCountInString(escaped) != utf8.RuneCountInString(state) {
		return nil, nil, nil, fmt.Errorf("Perplexity Locate escape changed state character positions")
	}
	ids := m.Tokenizer().Encode(contextText, false)
	normalizedContext := norm.NFC.String(contextText)
	normalizedState, byteFloor, byteCeil := normalizedRuneBoundaries(escaped)
	stateBegin = len(norm.NFC.String(contextText[:stateBegin]))
	stateEnd = stateBegin + len(normalizedState)
	var positions []int32
	var offsets []tokenOffset
	cursor := 0
	for i, id := range ids {
		piece := m.Tokenizer().Decode([]int32{id})
		// The general decoder omits NUL, but the publisher tokenizer keeps
		// its separate token and includes it in character offsets.
		if piece == "" && cursor < len(normalizedContext) && normalizedContext[cursor] == 0 {
			piece = "\x00"
		}
		next := cursor + len(piece)
		if next > len(normalizedContext) || normalizedContext[cursor:next] != piece {
			return nil, nil, nil, fmt.Errorf("Perplexity Locate cannot map token %d to state characters", i)
		}
		if cursor < stateEnd && next > stateBegin {
			start := max(cursor, stateBegin) - stateBegin
			end := min(next, stateEnd) - stateBegin
			if start < end {
				positions = append(positions, int32(i))
				offsets = append(offsets, tokenOffset{byteFloor[start], byteCeil[end]})
			}
		}
		cursor = next
	}
	if cursor != len(normalizedContext) {
		return nil, nil, nil, fmt.Errorf("Perplexity Locate token offsets do not cover the state context")
	}
	return append(ids, m.separatorID), positions, offsets, nil
}

func (m *Model) planLocate(input llm.ScoreRequest) ([]locatePlan, error) {
	prefix, positions, offsets, err := m.locateContext(input.State, input.System)
	if err != nil {
		return nil, err
	}
	var split func([]llm.ScoreRow) ([]locatePlan, error)
	split = func(rows []llm.ScoreRow) ([]locatePlan, error) {
		ids := slices.Clone(prefix)
		layout := &qwen3_5.IsolatedLayout{ContextLen: int32(len(prefix))}
		markers := make([]int32, 0, len(rows))
		for _, row := range rows {
			question := locatePromptPrefix + escapeDecisionText(row.Question.Instructions)
			chunk := m.Tokenizer().Encode(question, false)
			ids = append(ids, chunk...)
			ids = append(ids, m.locate.MarkerID)
			layout.Chunks = append(layout.Chunks, int32(len(chunk)+1))
			markers = append(markers, int32(len(ids)-1))
		}
		if len(ids) <= input.MaxTokens {
			return []locatePlan{{
				prepared: &model.PreparedRequest{Tokens: ids, Layout: layout, Isolated: true},
				markers:  markers, stateIDs: positions, offsets: offsets, rows: len(rows),
			}}, nil
		}
		if len(rows) == 1 {
			return nil, fmt.Errorf("Locate input exceeds context limit (%d > %d)", len(ids), input.MaxTokens)
		}
		middle := len(rows) / 2
		left, err := split(rows[:middle])
		if err != nil {
			return nil, err
		}
		right, err := split(rows[middle:])
		if err != nil {
			return nil, err
		}
		return append(left, right...), nil
	}
	return split(input.Rows)
}

func (m *Model) scoreLocate(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) ([][]locateSpan, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	plans, err := m.planLocate(input)
	if err != nil {
		return nil, 0, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	answers := make([][]locateSpan, 0, len(input.Rows))
	usage := 0
	for _, plan := range plans {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		usage += len(plan.prepared.Tokens)
		if len(plan.stateIDs) == 0 {
			for range plan.rows {
				answers = append(answers, []locateSpan{})
			}
			continue
		}
		var found [][]locateSpan
		_, err := forward(ctx, plan.prepared, nil, func(hidden *mlx.Array) error {
			var scoreErr error
			found, scoreErr = m.LocateHead.predict(hidden, plan, input.State, m.locate.Thresholds.Default, m.locate.Head.Proposals, m.locate.Head.MaxSpan)
			return scoreErr
		})
		if err != nil {
			return nil, 0, err
		}
		answers = append(answers, found...)
	}
	return answers, usage, ctx.Err()
}

func (h *spanHead) predict(hidden *mlx.Array, plan locatePlan, state string, cutoff float32, proposals, maxSpan int) ([][]locateSpan, error) {
	width := h.TS.OutputDim()
	amount := len(plan.stateIDs)
	labels := len(plan.markers)
	document := hidden.TakeAxis(mlx.FromValues(plan.stateIDs, amount), 1).Reshape(amount, hidden.Dim(2)).AsType(mlx.DTypeFloat32)
	questions := hidden.TakeAxis(mlx.FromValues(plan.markers, labels), 1).Reshape(labels, hidden.Dim(2)).AsType(mlx.DTypeFloat32)
	ts, te := h.TS.Forward(document), h.TE.Forward(document)
	qs, qe, qp := h.QS.Forward(questions), h.QE.Forward(questions), h.QP.Forward(questions)
	scale := float32(math.Sqrt(float64(width)))
	start := mlx.Add(mlx.DivScalar(mlx.Matmul(qs, ts.Transpose(1, 0)), scale), h.StartBias)
	end := mlx.Add(mlx.DivScalar(mlx.Matmul(qe, te.Transpose(1, 0)), scale), h.EndBias)
	thresholds := h.Thresholds.Forward(questions)
	k := min(proposals, amount)
	startTop := mlx.TopKIndices(start, k)
	endTop := mlx.TopKIndices(end, k)
	mlx.Eval(ts, te, qp, start, end, thresholds, startTop, endTop, h.WidthBias)
	tokens := []rune(state)
	tsValues, teValues, qpValues := ts.Floats(), te.Floats(), qp.Floats()
	starts, ends, biases := start.Floats(), end.Floats(), h.WidthBias.Floats()
	thresholdValues := thresholds.Floats()
	startIndexes, endIndexes := startTop.Ints(), endTop.Ints()
	result := make([][]locateSpan, labels)
	for question := range labels {
		unique := make(map[tokenOffset]float32)
		for i := range k {
			first := int(startIndexes[question*k+i])
			for j := range k {
				last := int(endIndexes[question*k+j])
				widthTokens := last - first + 1
				if widthTokens <= 0 || widthTokens > maxSpan {
					continue
				}
				var pair float32
				for feature := range int(width) {
					pair += tsValues[first*int(width)+feature] *
						float32(math.Tanh(float64(qpValues[question*int(width)+feature]))) *
						teValues[last*int(width)+feature]
				}
				bucket := min(31, int(math.Log2(float64(widthTokens))))
				logit := pair/scale + starts[question*amount+first] + ends[question*amount+last] + biases[bucket] - thresholdValues[question*3+2]
				probability := 1 / (1 + float32(math.Exp(float64(-logit))))
				if probability < cutoff {
					continue
				}
				a, b := plan.offsets[first].start, plan.offsets[last].end
				for a < b && unicode.IsSpace(tokens[a]) {
					a++
				}
				for b > a && unicode.IsSpace(tokens[b-1]) {
					b--
				}
				if a < b {
					span := tokenOffset{a, b}
					unique[span] = max(unique[span], probability)
				}
			}
		}
		spans := make([]tokenOffset, 0, len(unique))
		for span := range unique {
			spans = append(spans, span)
		}
		slices.SortFunc(spans, func(a, b tokenOffset) int {
			if a.start != b.start {
				return a.start - b.start
			}
			return a.end - b.end
		})
		result[question] = make([]locateSpan, 0, len(spans))
		for _, span := range spans {
			result[question] = append(result[question], locateSpan{
				Start: span.start, End: span.end, Text: string(tokens[span.start:span.end]), Score: unique[span],
			})
		}
	}
	return result, nil
}

func encodeLocateAnswers(spans [][]locateSpan) ([]json.RawMessage, error) {
	answers := make([]json.RawMessage, len(spans))
	for i, found := range spans {
		if found == nil {
			found = []locateSpan{}
		}
		answer, err := json.Marshal(locateAnswer{Type: "locate", Spans: found})
		if err != nil {
			return nil, err
		}
		answers[i] = answer
	}
	return answers, nil
}

func hasLocate(rows []llm.ScoreRow) bool {
	for _, row := range rows {
		if row.Question != nil && row.Question.Type == "locate" {
			return true
		}
	}
	return false
}
