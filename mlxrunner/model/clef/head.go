package clef

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type attention struct {
	Q, K, V, Out *nn.Linear
	heads        int
}

type evidenceLayer struct {
	QueryNorm, MemoryNorm, FFNorm *nn.LayerNorm
	Attention                     attention
	Up, Down                      *nn.Linear
}

type fieldLayer struct {
	Norm1, Norm2, Norm3 *nn.LayerNorm
	Self, Cross         attention
	Up, Down            *nn.Linear
}

type head struct {
	HiddenNorm, SummaryNorm, FieldNorm, OptionNorm                         *nn.LayerNorm
	Memory, Question, OptionQuestion, Global, OptionContext, OptionLexical *nn.Linear
	Types                                                                  *mlx.Array
	Evidence                                                               []evidenceLayer
	Fields                                                                 []fieldLayer
	ResidualIn, ResidualOut                                                *nn.Linear
	PriorScale, JointScale, ResidualGate                                   *mlx.Array
}

func (h *head) load(tensors map[string]*mlx.Array, c config) error {
	var loadErr error
	weight := func(name string, shape ...int) *mlx.Array {
		w := tensors[name]
		if w == nil || !slices.Equal(w.Dims(), shape) {
			if loadErr == nil {
				loadErr = fmt.Errorf("Clef weight %s: expected shape %v", name, shape)
			}
			return nil
		}
		if w.DType() != mlx.DTypeBFloat16 && w.DType() != mlx.DTypeFloat16 && w.DType() != mlx.DTypeFloat32 {
			if loadErr == nil {
				loadErr = fmt.Errorf("Clef head requires floating-point weights: %s", name)
			}
			return nil
		}
		return w
	}
	linear := func(name string, out, in int, bias bool) *nn.Linear {
		l := &nn.Linear{Weight: weight(name+".weight", out, in)}
		if bias {
			l.Bias = weight(name+".bias", out)
		}
		return l
	}
	norm := func(name string, size int) *nn.LayerNorm {
		return &nn.LayerNorm{Weight: weight(name+".weight", size), Bias: weight(name+".bias", size), Eps: 1e-5}
	}
	attn := func(name string) attention {
		a := attention{Out: linear(name+".out_proj", c.Width, c.Width, true), heads: c.Heads}
		w, b := weight(name+".in_proj_weight", 3*c.Width, c.Width), weight(name+".in_proj_bias", 3*c.Width)
		if w != nil && b != nil {
			projections := []*nn.Linear{}
			for i := range 3 {
				projections = append(projections, &nn.Linear{Weight: w.Slice(mlx.Slice(i*c.Width, (i+1)*c.Width), mlx.Slice()), Bias: b.Slice(mlx.Slice(i*c.Width, (i+1)*c.Width))})
			}
			a.Q, a.K, a.V = projections[0], projections[1], projections[2]
		}
		return a
	}
	h.HiddenNorm = norm("hidden_norm", c.HiddenSize)
	h.SummaryNorm = norm("option_summary_norm", c.Width)
	h.FieldNorm = norm("field_norm", c.Width)
	h.OptionNorm = norm("option_norm", c.Width)
	h.Memory = linear("memory_projection", c.Width, c.HiddenSize, false)
	h.Question = linear("question_projection", c.Width, c.HiddenSize, false)
	h.OptionQuestion = linear("option_question_projection", c.Width, c.HiddenSize, false)
	h.Global = linear("global_projection", c.Width, c.HiddenSize, false)
	h.OptionContext = linear("option_context_projection", c.Width, c.HiddenSize, false)
	h.OptionLexical = linear("option_lexical_projection", c.Width, c.HiddenSize, false)
	h.Types = weight("type_embedding.weight", 3, c.Width)
	h.ResidualIn = linear("residual_scorer.0", c.Width, 4*c.Width, true)
	h.ResidualOut = linear("residual_scorer.3", 1, c.Width, true)
	h.PriorScale = weight("prior_logit_scale")
	h.JointScale = weight("joint_logit_scale")
	h.ResidualGate = weight("residual_gate")
	h.Evidence = make([]evidenceLayer, c.RoutingLayers)
	for i := range h.Evidence {
		p := fmt.Sprintf("evidence_layers.%d", i)
		h.Evidence[i] = evidenceLayer{QueryNorm: norm(p+".query_norm", c.Width), MemoryNorm: norm(p+".memory_norm", c.Width), FFNorm: norm(p+".feedforward_norm", c.Width), Attention: attn(p + ".attention"), Up: linear(p+".feedforward.0", c.Feedforward, c.Width, true), Down: linear(p+".feedforward.3", c.Width, c.Feedforward, true)}
	}
	h.Fields = make([]fieldLayer, c.Layers)
	for i := range h.Fields {
		p := fmt.Sprintf("layers.%d", i)
		h.Fields[i] = fieldLayer{Norm1: norm(p+".norm1", c.Width), Norm2: norm(p+".norm2", c.Width), Norm3: norm(p+".norm3", c.Width), Self: attn(p + ".self_attn"), Cross: attn(p + ".multihead_attn"), Up: linear(p+".linear1", c.Feedforward, c.Width, true), Down: linear(p+".linear2", c.Width, c.Feedforward, true)}
	}
	return loadErr
}

func (a attention) forward(query, memory *mlx.Array) *mlx.Array {
	project := func(l *nn.Linear, x *mlx.Array) *mlx.Array {
		y := l.Forward(x)
		return y.Reshape(y.Dim(0), y.Dim(1), a.heads, -1).Transpose(0, 2, 1, 3)
	}
	q, k, v := project(a.Q, query), project(a.K, memory), project(a.V, memory)
	out := mlx.FastScaledDotProductAttention(q, k, v, 1/float32(math.Sqrt(float64(q.Dim(3)))), "", nil)
	return a.Out.Forward(out.Transpose(0, 2, 1, 3).Reshape(query.Dim(0), query.Dim(1), -1))
}

func normalize(x *mlx.Array, eps float32) *mlx.Array {
	f := x.AsType(mlx.DTypeFloat32)
	norm := mlx.Sum(mlx.Mul(f, f), -1, true).Sqrt().AsType(x.DType())
	return mlx.Div(x, mlx.Maximum(norm, mlx.FromValue(eps).AsType(x.DType())))
}

func meanSpan(x *mlx.Array, span tokenSpan) *mlx.Array {
	return mlx.Mean(x.Slice(mlx.Slice(span.start, span.end), mlx.Slice()), 0, false)
}

// forward scores the complete schema jointly. The backbone states are causal;
// the head attends to all state tokens and across all questions without a mask.
func (h *head) forward(hidden, ids *mlx.Array, questions []encodedQuestion, embedding nn.EmbeddingLayer) []*mlx.Array {
	sequence := h.HiddenNorm.Forward(hidden).Squeeze(0)
	memory := h.Memory.Forward(sequence).ExpandDims(0)
	global := sequence.Slice(mlx.Slice(sequence.Dim(0)-1, sequence.Dim(0)), mlx.Slice()).Squeeze(0)
	var questionVectors, lexical, queries []*mlx.Array
	var types []int32
	for _, q := range questions {
		questionVectors = append(questionVectors, meanSpan(sequence, q.instruction))
		types = append(types, q.kind)
	}
	question := mlx.Stack(questionVectors, 0)
	for i, q := range questions {
		var contexts, words []*mlx.Array
		for _, span := range q.options {
			contexts = append(contexts, meanSpan(sequence, span))
			tokens := ids.Slice(mlx.Slice(span.start, span.end))
			words = append(words, mlx.Mean(embedding.Forward(tokens), 0, false))
		}
		lexical = append(lexical, mlx.Stack(words, 0))
		queries = append(queries, mlx.Add(mlx.Add(h.OptionContext.Forward(mlx.Stack(contexts, 0)), h.OptionLexical.Forward(lexical[i])), h.OptionQuestion.Forward(questionVectors[i]).ExpandDims(0)))
	}
	routed := mlx.Concatenate(queries, 0).ExpandDims(0)
	for _, l := range h.Evidence {
		routed = mlx.Add(routed, l.Attention.forward(l.QueryNorm.Forward(routed), l.MemoryNorm.Forward(memory)))
		routed = mlx.Add(routed, l.Down.Forward(mlx.GELU(l.Up.Forward(l.FFNorm.Forward(routed)))))
	}
	routed = routed.Squeeze(0)
	base := h.Question.Forward(question)
	var options, summaries []*mlx.Array
	offset := 0
	for i, q := range questions {
		option := routed.Slice(mlx.Slice(offset, offset+len(q.options)), mlx.Slice())
		options = append(options, option)
		field := base.Slice(mlx.Slice(i, i+1), mlx.Slice()).Squeeze(0)
		weights := mlx.SoftmaxAxis(mlx.DivScalar(mlx.Matmul(option, field), float32(math.Sqrt(float64(option.Dim(1))))), 0, true)
		summaries = append(summaries, mlx.Sum(mlx.Mul(weights.ExpandDims(-1), option), 0, false))
		offset += len(q.options)
	}
	fields := mlx.Add(base, h.SummaryNorm.Forward(mlx.Stack(summaries, 0)))
	fields = mlx.Add(fields, h.Global.Forward(global).ExpandDims(0))
	fields = mlx.Add(fields, h.Types.TakeAxis(mlx.FromValues(types, len(types)), 0)).ExpandDims(0)
	for _, l := range h.Fields {
		norm := l.Norm1.Forward(fields)
		fields = mlx.Add(fields, l.Self.forward(norm, norm))
		fields = mlx.Add(fields, l.Cross.forward(l.Norm2.Forward(fields), memory))
		fields = mlx.Add(fields, l.Down.Forward(mlx.GELU(l.Up.Forward(l.Norm3.Forward(fields)))))
	}
	fields = h.FieldNorm.Forward(fields.Squeeze(0))
	scale := func(x *mlx.Array) *mlx.Array {
		return mlx.Exp(mlx.Minimum(x, mlx.FromValue(float32(math.Log(100))).AsType(x.DType())))
	}
	var logits []*mlx.Array
	for i := range questions {
		anchor := normalize(mlx.Add(questionVectors[i], global), 1e-12)
		prior := mlx.Mul(scale(h.PriorScale), mlx.Matmul(normalize(lexical[i], 1e-12), anchor))
		option := h.OptionNorm.Forward(options[i])
		field := fields.Slice(mlx.Slice(i, i+1), mlx.Slice())
		field = mlx.BroadcastTo(field, int32(option.Dim(0)), int32(option.Dim(1)))
		cosine := mlx.Sum(mlx.Mul(normalize(field, 1e-8), normalize(option, 1e-8)), -1, false)
		features := mlx.Concatenate([]*mlx.Array{field, option, mlx.Mul(field, option), mlx.Sub(field, option).Abs()}, -1)
		residual := h.ResidualOut.Forward(mlx.GELU(h.ResidualIn.Forward(features))).Squeeze(-1)
		joint := mlx.Add(mlx.Mul(scale(h.JointScale), cosine), residual)
		logits = append(logits, mlx.Add(prior, mlx.Mul(h.ResidualGate.Sigmoid(), joint)))
	}
	return logits
}
