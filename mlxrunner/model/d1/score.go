package d1

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lfm3"
)

var _ model.Scorer = (*Model)(nil)

// row is one packed forward pass: the state, then up to set-size questions.
type row struct {
	prepared  *model.PreparedRequest
	segments  []model.Segment
	stateLen  int
	group     []int32 // per token: -1 in the state, else the question's index in the row
	questions []int   // request indexes
	slots     []int   // per question: absolute answer slot position
}

func badRequest(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

// Score answers with a score's probability-weighted level, as System One
// defines it; the publisher's own API defaults to the median level.
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	native := input.Native
	if native == nil {
		return result, badRequest("d1 models answer System One requests only")
	}
	if len(native.Questions) == 0 {
		return result, badRequest("questions must not be empty")
	}
	qs := make([]question, len(native.Questions))
	for i, nq := range native.Questions {
		q, err := parseQuestion(nq.Name, nq.Question)
		if err != nil {
			return result, badRequest("question %q: %v", nq.Name, err)
		}
		qs[i] = q
	}

	text, err := stateText(native.State)
	if err != nil {
		return result, badRequest("%v", err)
	}
	body := m.render.plain(text) + "\n\n" + "\n" + questionHeader

	suffixes := make([]string, len(qs))
	slots := make([]slot, len(qs))
	for i := range qs {
		if suffixes[i], slots[i], err = m.render.suffix(&qs[i]); err != nil {
			return result, badRequest("question %q: %v", qs[i].name, err)
		}
	}
	segments, err := mediaSegments(input.Images)
	if err != nil {
		return result, badRequest("%v", err)
	}

	answers := make([]json.RawMessage, len(qs))
	setSize := m.config.SetSize
	for start := 0; start < len(qs); start += setSize {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		end := min(start+setSize, len(qs))
		r, err := m.buildRow(body, segments, qs, suffixes, slots, start, end)
		if err != nil {
			return result, badRequest("%v", err)
		}
		if len(r.prepared.Tokens) > input.MaxTokens {
			return result, badRequest("decision input needs %d tokens; the context limit is %d", len(r.prepared.Tokens), input.MaxTokens)
		}
		result.InputTokens += len(r.prepared.Tokens)
		var rowAnswers []json.RawMessage
		_, err = forward(ctx, r.prepared, r.segments, func(hidden *mlx.Array) error {
			rowAnswers, err = m.answerRow(r, qs, slots, hidden)
			return err
		})
		if err != nil {
			return result, err
		}
		for i, q := range r.questions {
			answers[q] = rowAnswers[i]
		}
	}
	result.Answers = answers
	return result, ctx.Err()
}

// mediaSegments sends audio clips, detected as chat does, to the audio
// encoder and everything else to the image encoder.
func mediaSegments(media []api.ImageData) ([]model.Segment, error) {
	var segments []model.Segment
	for i, data := range media {
		if llm.DetectMediaKind(data) == llm.MediaKindAudio {
			segments = append(segments, model.Segment{Kind: "audio", Data: data})
			continue
		}
		capped, err := lfm3.CapImagePixels(data, visionMaxPixels)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		segments = append(segments, model.Segment{Kind: "image", Data: capped})
	}
	return segments, nil
}

// buildRow packs questions [start, end) after the state body.
func (m *Model) buildRow(body string, media []model.Segment, qs []question, suffixes []string, slots []slot, start, end int) (*row, error) {
	tok := m.Tokenizer()
	first := tok.Encode(body+suffixes[start], false)
	firstSuffix := tok.Encode(suffixes[start], false)
	bodyLen := len(first) - len(firstSuffix)
	if bodyLen < 0 || !slices.Equal(first[bodyLen:], firstSuffix) {
		return nil, fmt.Errorf("the state and question %q do not tokenize apart", qs[start].name)
	}
	segments := []model.Segment{{Tokens: tok.Encode(userTurn, false)}}
	segments = append(segments, media...)
	segments = append(segments, model.Segment{Tokens: first})
	for j := start + 1; j < end; j++ {
		segments = append(segments, model.Segment{Tokens: tok.Encode(questionHeader+suffixes[j], false)})
	}
	prepared, err := m.PrepareMedia(segments)
	if err != nil {
		return nil, err
	}
	total := len(prepared.Tokens)
	tail := 0
	for j := start + 1; j < end; j++ {
		tail += len(segments[len(media)+2+(j-start-1)].Tokens)
	}
	stateLen := total - tail - len(firstSuffix)

	r := &row{prepared: prepared, segments: segments, stateLen: stateLen, group: make([]int32, total)}
	for i := range stateLen {
		r.group[i] = -1
	}
	pos := stateLen
	for j := start; j < end; j++ {
		n := len(firstSuffix)
		if j > start {
			n = len(segments[len(media)+2+(j-start-1)].Tokens)
		}
		for t := pos; t < pos+n; t++ {
			r.group[t] = int32(j - start)
		}
		r.questions = append(r.questions, j)
		r.slots = append(r.slots, pos+n-1+slots[j].offset)
		pos += n
	}
	prepared.Layout = r
	return r, nil
}

// questionMask lets question tokens attend to the state and their own
// question, in both directions: one row per question token, one column per
// token in the row.
func questionMask(r *row, dtype mlx.DType) *mlx.Array {
	t := len(r.group)
	g := mlx.FromValues(r.group, t)
	gq := mlx.FromValues(r.group[r.stateLen:], t-r.stateLen).Reshape(t-r.stateLen, 1)
	gk := g.Reshape(1, t)
	inState := mlx.FromValues([]int32{-1}, 1)
	zero := mlx.Zeros(dtype, 1)
	blocked := mlx.FromValues([]float32{float32(math.Inf(-1))}, 1).AsType(dtype)
	allowed := mlx.Where(gk.Equal(inState), zero, mlx.Where(gq.Equal(gk), zero, blocked))
	return allowed.Reshape(1, 1, t-r.stateLen, t)
}

// setAttention attends the state to the whole row unmasked and the questions
// through questionMask, so no mask covers the state's rows. Sliding layers
// stay causal within the kernel's own window. A row-sized mask would cost
// memory quadratic in the row: a 64K-token state needs gigabytes per mask.
func setAttention(r *row, window int) func(q, k, v *mlx.Array, scale float32, sliding bool) *mlx.Array {
	return func(q, k, v *mlx.Array, scale float32, sliding bool) *mlx.Array {
		if sliding {
			return mlx.FastScaledDotProductAttention(q, k, v, scale, window, "causal", nil)
		}
		// Built per layer: each layer's arrays are freed with its scope.
		mask := questionMask(r, q.DType())
		b, h, t, d := int32(q.Dim(0)), int32(q.Dim(1)), int32(q.Dim(2)), int32(q.Dim(3))
		s := int32(r.stateLen)
		state := mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{b, h, s, d})
		questions := mlx.SliceStartStop(q, []int32{0, 0, s, 0}, []int32{b, h, t, d})
		return mlx.Concatenate([]*mlx.Array{
			mlx.FastScaledDotProductAttention(state, k, v, scale, mlx.FastSDPANoWindow, "", nil),
			mlx.FastScaledDotProductAttention(questions, k, v, scale, mlx.FastSDPANoWindow, "", mask),
		}, 2)
	}
}

func (m *Model) answerRow(r *row, qs []question, slots []slot, hidden *mlx.Array) ([]json.RawMessage, error) {
	h := hidden.Reshape(len(r.prepared.Tokens), int(m.config.HiddenSize))

	// Every answer slot's vocabulary scores in one gather and transfer.
	positions := make([]int32, len(r.slots))
	var ids []int32
	index := map[int32]int{}
	for k, qi := range r.questions {
		positions[k] = int32(r.slots[k])
		for _, g := range slots[qi].groups {
			for _, id := range g {
				if _, ok := index[id]; !ok {
					index[id] = len(ids)
					ids = append(ids, id)
				}
			}
		}
	}
	rows := h.TakeAxis(mlx.FromValues(positions, len(positions)), 0)
	logits := m.LMHead.Forward(rows).AsType(mlx.DTypeFloat32)
	logz := mlx.Sub(logits, logits.LogsumexpAxis(-1, true))
	// Floats copies the buffer as laid out; a gather may leave it strided.
	picked := mlx.Contiguous(logz.TakeAxis(mlx.FromValues(ids, len(ids)), 1), false)
	mlx.Eval(picked)
	scores := picked.Floats()

	answers := make([]json.RawMessage, len(r.questions))
	for k, qi := range r.questions {
		q := &qs[qi]
		groups := slots[qi].groups
		grouped := make([]float64, len(groups))
		for o, g := range groups {
			best := math.Inf(-1)
			for _, id := range g {
				best = math.Max(best, float64(scores[k*len(ids)+index[id]]))
			}
			grouped[o] = best
		}
		p := softmax(grouped, m.calibration.Default)
		var answer any
		switch q.typ {
		case "noul":
			answer = decision.NoulAnswer{Type: "noul", Noul: p[0]}
		case "choice":
			pick := 0
			probabilities := &decision.Probabilities{}
			for o, opt := range q.options {
				if p[o] > p[pick] {
					pick = o
				}
				probabilities.Set(opt.name, p[o])
			}
			answer = decision.ChoiceAnswer{Type: "choice", Choice: q.options[pick].name, Probabilities: probabilities, Confidence: decision.Concentration(p)}
		case "score":
			probabilities := &decision.Probabilities{}
			legend := &decision.Legend{}
			for i, level := range q.levels {
				key := strconv.Itoa(i)
				probabilities.Set(key, p[i])
				legend.Set(key, level)
			}
			var score float64
			for i, v := range p {
				score += float64(i) * v
			}
			answer = decision.ScoreAnswer{Type: "score", Score: score, Legend: legend, Probabilities: probabilities, Confidence: decision.Concentration(p)}
		default:
			return nil, fmt.Errorf("unsupported question type %q", q.typ)
		}
		data, err := json.Marshal(answer)
		if err != nil {
			return nil, err
		}
		answers[k] = data
	}
	return answers, nil
}

func batchSetAttention(b *batch.Batch, window int) func(q, k, v *mlx.Array, scale float32, sliding bool) *mlx.Array {
	return func(q, k, v *mlx.Array, scale float32, sliding bool) *mlx.Array {
		rows := make([]*mlx.Array, len(b.SeqQueryLens))
		for i, n := range b.SeqQueryLens {
			layout := b.Layout[i].(*row)
			slice := func(a *mlx.Array) *mlx.Array {
				return a.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(0, int(n)), mlx.Slice())
			}
			out := setAttention(layout, window)(slice(q), slice(k), slice(v), scale, sliding)
			if tail := q.Dim(2) - int(n); tail > 0 {
				out = mlx.Concatenate([]*mlx.Array{out, mlx.Zeros(out.DType(), 1, out.Dim(1), tail, out.Dim(3))}, 2)
			}
			rows[i] = out
		}
		return mlx.Concatenate(rows, 0)
	}
}
