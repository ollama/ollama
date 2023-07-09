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
