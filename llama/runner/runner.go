package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llama"
)

type Sequence struct {
	// number of tokens evaluated
	nPast int

	// batch index
	iBatch int

	// number of tokens predicted so far
	numPredicted int

	// tokens left to evaluate
	tokens []int

	// channel to send responses over
	responses chan string

	// number of tokens to predict
	numPredict int

	samplingCtx *llama.SamplingContext

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// stop sequences
	stop []string

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	doneReason string

	// Metrics
	t_start_process_prompt time.Time
	t_start_genereration   time.Time
	n_decoded              int
	n_prompt_tokens        int
}

func (s *Server) NewSequence(prompt string, numPredict int, stop []string, params *llama.SamplingParams, embedding bool) *Sequence {
	tokens, err := s.lc.Model().Tokenize(prompt, true, true)
	if err != nil {
		panic(err)
	}

	// truncate to last n tokens
	// TODO: this shouldn't happen and will severely impact generation
	// quality. instead we should ensure to cut prompt in the API.
	if len(tokens) > s.numCtx {
		tokens = tokens[:s.numCtx]
	}

	var sc *llama.SamplingContext
	if params != nil {
		sc = llama.NewSamplingContext(*params)
		for _, t := range tokens {
			sc.Accept(s.lc, t, false)
		}
	}

	return &Sequence{
		tokens:          tokens,
		n_prompt_tokens: len(tokens),
		numPredict:      numPredict,
		responses:       make(chan string, 1),
		embedding:       make(chan []float32, 1),
		samplingCtx:     sc,
		embeddingOnly:   embedding,
		stop:            stop,
	}
}

type Server struct {
	model *llama.Model
	lc    *llama.Context
	cc    *llama.ClipContext

	batchSize int

	// parallel is the number of parallel requests to handle
	parallel int

	// seqs is the list of parallel sequences being evaluated
	// TODO (jmorganca): this can probably be moved into run()
	seqs []*Sequence

	// context window size
	numCtx int

	mu sync.Mutex

	cond *sync.Cond

	progress float32

	status string
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func (s *Server) run(ctx context.Context) {
	// TODO - should this be n_ctx / parallel like the old server.cpp setup?
	batch := llama.NewBatch(s.batchSize, 0, s.parallel)
	defer batch.Free()

	// build up stop sequences as we recognize them
	// TODO (jmorganca): simplify this
	pieces := make([][]string, s.parallel)

	for {
		select {
		case <-ctx.Done():
			return
		default:
			slog.Debug("Processing batch", "seqs", len(s.seqs))
			s.mu.Lock()
			for s.allNil() {
				s.cond.Wait() // Wait until an item is added
			}
			s.mu.Unlock()

			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				hitLimit := seq.numPredict > 0 && seq.numPredicted > seq.numPredict

				// if past the num predict limit
				if hitLimit || seq.nPast > s.numCtx {
					seq.doneReason = "limit"
					close(seq.responses)
					s.lc.KvCacheSeqRm(i, 0, -1)
					s.seqs[i] = nil
					continue
				}

				if seq.t_start_process_prompt.IsZero() {
					seq.t_start_process_prompt = time.Now()
				}

				var numTokensProcessed int
				for j, t := range seq.tokens {
					// todo: make this n_batch
					if j >= s.batchSize {
						break
					}
					batch.Add(t, seq.nPast, []int{i}, numTokensProcessed+1 == len(seq.tokens))
					seq.nPast++
					numTokensProcessed++
				}
				seq.tokens = seq.tokens[numTokensProcessed:]
				seq.iBatch = batch.NumTokens() - 1
			}

			if batch.NumTokens() == 0 {
				continue
			}

			err := s.lc.Decode(batch)
			if err != nil {
				slog.Error("failed to decode batch", "error", err)
				panic("Failed to decode")
			}

			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				// don't sample prompt processing
				if len(seq.tokens) != 0 {
					continue
				}

				// if done processing the prompt, generating an embedding and return
				if seq.embeddingOnly {
					embd := s.lc.GetEmbeddingsSeq(i)
					if embd == nil {
						embd = s.lc.GetEmbeddingsIth(seq.iBatch)
					}

					seq.embedding <- embd
					close(seq.embedding)
					s.lc.KvCacheSeqRm(i, 0, -1)
					s.seqs[i] = nil
					continue
				}

				// sample a token
				// logits := s.lc.GetLogitsIth(ibatch[i])
				// token := s.lc.SampleTokenGreedy(logits)
				token := seq.samplingCtx.Sample(s.lc, nil, seq.iBatch)

				seq.samplingCtx.Accept(s.lc, token, true)
				seq.n_decoded += 1
				if seq.n_decoded == 1 {
					seq.t_start_genereration = time.Now()
				}
				piece := s.model.TokenToPiece(token)

				seq.numPredicted++

				slog.Debug("sampled", "piece", piece)

				// if it's an end of sequence token, break
				// TODO: just end this sequence
				if s.model.TokenIsEog(token) {
					// TODO: end the sequence instead of quitting the pool
					s.lc.KvCacheSeqRm(i, 0, -1)

					// TODO (jmorganca): we should send this back
					// as it's important for the /api/generate context
					// seq.responses <- piece

					seq.doneReason = "stop"
					close(seq.responses)
					seq.samplingCtx.Free()
					pieces[i] = []string{}
					s.seqs[i] = nil
					continue
				}

				seq.tokens = []int{token}

				pieces[i] = append(pieces[i], piece)
				sequence := strings.Join(pieces[i], "")
				if ok, stop := findStop(sequence, seq.stop); ok {
					slog.Info("hit stop token", "stop", seq.stop)

					truncated := truncateStop(pieces[i], stop)

					for _, p := range truncated {
						seq.responses <- p
					}

					s.lc.KvCacheSeqRm(i, 0, -1)
					seq.doneReason = "stop"
					close(seq.responses)
					seq.samplingCtx.Free()
					pieces[i] = []string{}
					s.seqs[i] = nil
					continue
				}

				if containsStopSuffix(sequence, seq.stop) {
					continue
				}

				for _, p := range pieces[i] {
					seq.responses <- p
				}

				pieces[i] = []string{}
			}

			batch.Clear()
		}
	}
}

type CompletionRequest struct {
	Prompt  string   `json:"prompt"`
	Images  []string `json:"images"`
	Grammar string   `json:"grammar"`
	Stop    []string `json:"stop"`

	api.Options
}

type Timings struct {
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
}

type CompletionResponse struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`

	Model        string  `json:"model,omitempty"`
	Prompt       string  `json:"prompt,omitempty"`
	StoppedLimit bool    `json:"stopped_limit,omitempty"`
	PredictedN   int     `json:"predicted_n,omitempty"`
	PredictedMS  float64 `json:"predicted_ms,omitempty"`
	PromptN      int     `json:"prompt_n,omitempty"`
	PromptMS     float64 `json:"prompt_ms,omitempty"`

	Timings Timings `json:"timings"`
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req CompletionRequest
	req.Options = api.DefaultOptions()
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	var samplingParams llama.SamplingParams
	samplingParams.TopK = req.TopK
	samplingParams.TopP = req.TopP
	samplingParams.TfsZ = req.TFSZ
	samplingParams.TypicalP = req.TypicalP
	samplingParams.Temp = req.Temperature
	samplingParams.PenaltyRepeat = req.RepeatPenalty
	samplingParams.PenaltyFreq = req.FrequencyPenalty
	samplingParams.PenaltyPresent = req.PresencePenalty
	samplingParams.Mirostat = req.Mirostat
	samplingParams.MirostatTau = req.MirostatTau
	samplingParams.MirostatEta = req.MirostatEta
	samplingParams.PenalizeNl = req.PenalizeNewline
	samplingParams.Seed = uint32(req.Seed)
	samplingParams.Grammar = req.Grammar

	seq := s.NewSequence(req.Prompt, req.NumPredict, req.Stop, &samplingParams, false)

	// TODO (jmorganca): add to sequence queue instead of
	// failing if a slot isn't available
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			s.seqs[i] = seq
			s.cond.Signal()
			break
		}
	}
	s.mu.Unlock()

	// stream the response
	for content := range seq.responses {
		if err := json.NewEncoder(w).Encode(&CompletionResponse{
			Content: content,
		}); err != nil {
			log.Println("Failed to encode result:", err)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		flusher.Flush()
	}

	// Send the stop
	if err := json.NewEncoder(w).Encode(&CompletionResponse{
		Stop: true,
		Timings: Timings{
			PromptN:     seq.n_prompt_tokens,
			PromptMS:    float64(seq.t_start_genereration.Sub(seq.t_start_process_prompt).Milliseconds()),
			PredictedN:  seq.n_decoded,
			PredictedMS: float64(time.Since(seq.t_start_genereration).Milliseconds()),
		},
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	flusher.Flush()
}

type EmbeddingRequest struct {
	Content []string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding [][]float32 `json:"embedding"`
}

// TODO (jmorganca): is it safe to do this concurrently with decoding?
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	slog.Debug("embedding request", "content", req.Content)
	seqs := make([]*Sequence, len(req.Content))
	embeddings := make([][]float32, len(req.Content))
	var processed int
	for i, content := range req.Content {
		seqs[i] = s.NewSequence(content, 0, nil, nil, true)
	}

	// TODO - refactor to go routines to add seq's and drain the responses
	// so we don't stall until each set is iterated through
	for processed < len(seqs) {
		s.mu.Lock()
		for i, sq := range s.seqs {
			if processed >= len(seqs) {
				break
			}
			if sq == nil {
				s.seqs[i] = seqs[processed]
				processed += 1
			}
		}
		s.cond.Signal()
		s.mu.Unlock()

		for i := range processed {
			embeddings[i] = <-seqs[i].embedding
		}
	}

	if err := json.NewEncoder(w).Encode(&EmbeddingResponse{
		Embedding: embeddings,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

// TODO (jmorganca): is it safe to do this concurrently with decoding?
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&HealthResponse{
		Status:   s.status,
		Progress: s.progress,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("mmproj", "", "Path to projector binary file")
	parallel := flag.Int("parallel", 1, "Number of sequences to handle simultaneously")
	batchSize := flag.Int("batch-size", 512, "Batch size")
	nGpuLayers := flag.Int("n-gpu-layers", 0, "Number of layers to offload to GPU")
	mainGpu := flag.Int("main-gpu", 0, "Main GPU")
	flashAttention := flag.Bool("flash-attn", false, "Enable flash attention")
	numCtx := flag.Int("ctx-size", 2048, "Context (or KV cache) size")
	lpath := flag.String("lora", "", "Path to lora layer file")
	port := flag.Int("port", 8080, "Port to expose the server on")
	threads := flag.Int("threads", runtime.NumCPU(), "Number of threads to use during generation")

	// TODO not yet implemented but wired to keep the parsing aligned
	embedding := flag.Bool("embedding", false, "enable embedding vector output (default: disabled)")
	logDisable := flag.Bool("log-disable", false, "disables logging to a file")
	verbose := flag.Bool("verbose", false, "verbose output (default: disabled)")
	f32 := flag.Bool("memory-f32", false, "use f32 instead of f16 for memory key+value (default: disabled) not recommended: doubles context memory required and no measurable increase in quality")
	noMmap := flag.Bool("no-mmap", false, "do not memory-map model (slower load but may reduce pageouts if not using mlock)")
	mlock := flag.Bool("mlock", false, "force system to keep model in RAM rather than swapping or compressing")
	tensorSplit := flag.String("tensor-split", "", "fraction of the model to offload to each GPU, comma-separated list of proportions")

	flag.Parse()
	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})
	slog.SetDefault(slog.New(handler))

	// TODO actually implement...
	if *embedding {
		slog.Warn("embeddings not yet support")
	}
	if *logDisable {
		slog.Info("ignoring --log-disable")
	}
	if *f32 {
		slog.Warn("memory-f32 not yet supported")
	}
	if *noMmap {
		slog.Warn("no-mmap not yet supported")
	}
	if *mlock {
		slog.Warn("mlock not yet supported")
	}
	if *tensorSplit != "" {
		slog.Warn("tensor-split not yet implemented")
	}

	server := &Server{
		numCtx:    *numCtx,
		batchSize: *batchSize,
		parallel:  *parallel,
		seqs:      make([]*Sequence, *parallel),
		status:    "loading",
	}

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams(*nGpuLayers, *mainGpu, func(progress float32) {
		slog.Debug("Loading model", "progress %", math.Round(float64(progress*100)))
		server.progress = progress
	})
	server.model = llama.LoadModelFromFile(*mpath, params)

	if *lpath != "" {
		err := server.model.ApplyLoraFromFile(*lpath, 1.0, "", *threads)
		if err != nil {
			panic(err)
		}
	}

	ctxParams := llama.NewContextParams(*numCtx, *threads, *flashAttention)
	server.lc = llama.NewContextWithModel(server.model, ctxParams)

	if *ppath != "" {
		server.cc = llama.NewClipContext(*ppath)
	}

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/embedding", server.embeddings)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)

	httpServer := http.Server{
		Handler: mux,
	}

	server.status = "ok"

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}

	cancel()
}
