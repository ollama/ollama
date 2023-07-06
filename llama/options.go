// MIT License

// Copyright (c) 2023 go-skynet authors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package llama

type ModelOptions struct {
	ContextSize int
	Seed        int
	NBatch      int
	F16Memory   bool
	MLock       bool
	MMap        bool
	VocabOnly   bool
	LowVRAM     bool
	Embeddings  bool
	NUMA        bool
	NGPULayers  int
	MainGPU     string
	TensorSplit string
}

type PredictOptions struct {
	Seed, Threads, Tokens, TopK, Repeat, Batch, NKeep int
	TopP, Temperature, Penalty                        float64
	F16KV                                             bool
	DebugMode                                         bool
	StopPrompts                                       []string
	IgnoreEOS                                         bool

	TailFreeSamplingZ float64
	TypicalP          float64
	FrequencyPenalty  float64
	PresencePenalty   float64
	Mirostat          int
	MirostatETA       float64
	MirostatTAU       float64
	PenalizeNL        bool
	LogitBias         string
	TokenCallback     func(string) bool

	MLock, MMap bool
	MainGPU     string
	TensorSplit string
}

type PredictOption func(p *PredictOptions)

type ModelOption func(p *ModelOptions)

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize: 512,
	Seed:        0,
	F16Memory:   false,
	MLock:       false,
	Embeddings:  false,
	MMap:        true,
	LowVRAM:     false,
}

var DefaultOptions PredictOptions = PredictOptions{
	Seed:              -1,
	Threads:           4,
	Tokens:            128,
	Penalty:           1.1,
	Repeat:            64,
	Batch:             512,
	NKeep:             64,
	TopK:              40,
	TopP:              0.95,
	TailFreeSamplingZ: 1.0,
	TypicalP:          1.0,
	Temperature:       0.8,
	FrequencyPenalty:  0.0,
	PresencePenalty:   0.0,
	Mirostat:          0,
	MirostatTAU:       5.0,
	MirostatETA:       0.1,
	MMap:              true,
}

// SetContext sets the context size.
func SetContext(c int) ModelOption {
	return func(p *ModelOptions) {
		p.ContextSize = c
	}
}

func SetModelSeed(c int) ModelOption {
	return func(p *ModelOptions) {
		p.Seed = c
	}
}

// SetContext sets the context size.
func SetMMap(b bool) ModelOption {
	return func(p *ModelOptions) {
		p.MMap = b
	}
}

// SetNBatch sets the  n_Batch
func SetNBatch(n_batch int) ModelOption {
	return func(p *ModelOptions) {
		p.NBatch = n_batch
	}
}

// Set sets the tensor split for the GPU
func SetTensorSplit(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.TensorSplit = maingpu
	}
}

// SetMainGPU sets the main_gpu
func SetMainGPU(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.MainGPU = maingpu
	}
}

// SetPredictionTensorSplit sets the tensor split for the GPU
func SetPredictionTensorSplit(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.TensorSplit = maingpu
	}
}

// SetPredictionMainGPU sets the main_gpu
func SetPredictionMainGPU(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.MainGPU = maingpu
	}
}

var VocabOnly ModelOption = func(p *ModelOptions) {
	p.VocabOnly = true
}

var EnabelLowVRAM ModelOption = func(p *ModelOptions) {
	p.LowVRAM = true
}

var EnableNUMA ModelOption = func(p *ModelOptions) {
	p.NUMA = true
}

var EnableEmbeddings ModelOption = func(p *ModelOptions) {
	p.Embeddings = true
}

var EnableF16Memory ModelOption = func(p *ModelOptions) {
	p.F16Memory = true
}

var EnableF16KV PredictOption = func(p *PredictOptions) {
	p.F16KV = true
}

var Debug PredictOption = func(p *PredictOptions) {
	p.DebugMode = true
}

var EnableMLock ModelOption = func(p *ModelOptions) {
	p.MLock = true
}

// Create a new PredictOptions object with the given options.
func NewModelOptions(opts ...ModelOption) ModelOptions {
	p := DefaultModelOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

var IgnoreEOS PredictOption = func(p *PredictOptions) {
	p.IgnoreEOS = true
}

// SetMlock sets the memory lock.
func SetMlock(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MLock = b
	}
}

// SetMemoryMap sets memory mapping.
func SetMemoryMap(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MMap = b
	}
}

// SetGPULayers sets the number of GPU layers to use to offload computation
func SetGPULayers(n int) ModelOption {
	return func(p *ModelOptions) {
		p.NGPULayers = n
	}
}

// SetTokenCallback sets the prompts that will stop predictions.
func SetTokenCallback(fn func(string) bool) PredictOption {
	return func(p *PredictOptions) {
		p.TokenCallback = fn
	}
}

// SetStopWords sets the prompts that will stop predictions.
func SetStopWords(stop ...string) PredictOption {
	return func(p *PredictOptions) {
		p.StopPrompts = stop
	}
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) PredictOption {
	return func(p *PredictOptions) {
		p.Seed = seed
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) PredictOption {
	return func(p *PredictOptions) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) PredictOption {
	return func(p *PredictOptions) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float64) PredictOption {
	return func(p *PredictOptions) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float64) PredictOption {
	return func(p *PredictOptions) {
		p.Temperature = temp
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float64) PredictOption {
	return func(p *PredictOptions) {
		p.Penalty = penalty
	}
}

// SetRepeat sets the number of times to repeat text generation.
func SetRepeat(repeat int) PredictOption {
	return func(p *PredictOptions) {
		p.Repeat = repeat
	}
}

// SetBatch sets the batch size.
func SetBatch(size int) PredictOption {
	return func(p *PredictOptions) {
		p.Batch = size
	}
}

// SetKeep sets the number of tokens from initial prompt to keep.
func SetNKeep(n int) PredictOption {
	return func(p *PredictOptions) {
		p.NKeep = n
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

// SetTailFreeSamplingZ sets the tail free sampling, parameter z.
func SetTailFreeSamplingZ(tfz float64) PredictOption {
	return func(p *PredictOptions) {
		p.TailFreeSamplingZ = tfz
	}
}

// SetTypicalP sets the typicality parameter, p_typical.
func SetTypicalP(tp float64) PredictOption {
	return func(p *PredictOptions) {
		p.TypicalP = tp
	}
}

// SetFrequencyPenalty sets the frequency penalty parameter, freq_penalty.
func SetFrequencyPenalty(fp float64) PredictOption {
	return func(p *PredictOptions) {
		p.FrequencyPenalty = fp
	}
}

// SetPresencePenalty sets the presence penalty parameter, presence_penalty.
func SetPresencePenalty(pp float64) PredictOption {
	return func(p *PredictOptions) {
		p.PresencePenalty = pp
	}
}

// SetMirostat sets the mirostat parameter.
func SetMirostat(m int) PredictOption {
	return func(p *PredictOptions) {
		p.Mirostat = m
	}
}

// SetMirostatETA sets the mirostat ETA parameter.
func SetMirostatETA(me float64) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatETA = me
	}
}

// SetMirostatTAU sets the mirostat TAU parameter.
func SetMirostatTAU(mt float64) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatTAU = mt
	}
}

// SetPenalizeNL sets whether to penalize newlines or not.
func SetPenalizeNL(pnl bool) PredictOption {
	return func(p *PredictOptions) {
		p.PenalizeNL = pnl
	}
}

// SetLogitBias sets the logit bias parameter.
func SetLogitBias(lb string) PredictOption {
	return func(p *PredictOptions) {
		p.LogitBias = lb
	}
}
