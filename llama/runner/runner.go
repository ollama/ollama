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
	"regexp"
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
	numPast int

	// batch index
	iBatch int

	// number of tokens predicted so far
	numPredicted int

	// tokens left to evaluate
	tokens []int

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	// TODO (jmorganca): simplify this
	pendingResponses []string

	// token cache being used by this sequence
	cache *TokenCacheSlot

	// channel to send responses over
	responses chan string

	// channel to stop decoding (such as if the remote connection is closed)
	quit chan bool

	// number of tokens to predict
	numPredict int

	samplingCtx *llama.SamplingContext

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// stop sequences
	stop []string

	// number of tokens to keep at the beginning when shifting context window
	numKeep int

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	doneReason string

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptTokens     int
}

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
}

func (s *Server) NewSequence(prompt string, params NewSequenceParams) *Sequence {
	s.ready.Wait()

	tokens, err := s.lc.Model().Tokenize(prompt, true, true)
	if err != nil {
		panic(err)
	}

	if params.numKeep < 0 {
		params.numKeep = len(tokens)
	}

	if !params.embedding {
		// Subtracting 4 ensures that at least 1 token can be discarded during shift
		params.numKeep = min(params.numKeep, s.cache.numCtx-4)
		params.numKeep += s.bosToken
	} else {
		// Embeddings are 1 shot - just truncate to the context window, without ever shifting
		params.numKeep = min(params.numKeep, s.cache.numCtx)
	}

	// truncate to fit in context window
	if len(tokens) > s.cache.numCtx {
		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(tokens), "numKeep", params.numKeep)
		newTokens := tokens[:params.numKeep]
		newTokens = append(newTokens, tokens[len(tokens)-s.cache.numCtx+params.numKeep:]...)
		tokens = newTokens
	}

	var sc *llama.SamplingContext
	if params.samplingParams != nil {
		sc = llama.NewSamplingContext(*params.samplingParams)
		for _, t := range tokens {
			sc.Accept(s.lc, t, false)
		}
	}

	return &Sequence{
		tokens:           tokens,
		numPromptTokens:  len(tokens),
		numPredict:       params.numPredict,
		pendingResponses: make([]string, 0),
		responses:        make(chan string, 1),
		quit:             make(chan bool, 1),
		embedding:        make(chan []float32, 1),
		samplingCtx:      sc,
		embeddingOnly:    params.embedding,
		stop:             params.stop,
		numKeep:          params.numKeep,
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

	// KV cache
	cache *TokenCache

	// does this model require a beginning of sequence token?
	bosToken int

	// is the server ready to process requests?
	ready sync.WaitGroup

	mu sync.Mutex

	cond *sync.Cond

	progress float32

	status ServerStatus
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func (s *Server) shiftContext(seq *Sequence) {
	numLeft := seq.numPast - seq.numKeep
	numDiscard := numLeft / 2

	slog.Debug("context limit hit - shifting", "limit", s.cache.numCtx, "numPast", seq.numPast,
		"numKeep", seq.numKeep, "numLeft", numLeft, "numDiscard", numDiscard)

	s.cache.ShiftCacheSlot(seq.cache, seq.numKeep, numDiscard, seq.numPast)

	seq.numPast -= numDiscard
}

func incompleteUnicode(token string) bool {
	incomplete := false

	// check if there is incomplete UTF-8 character at the end
	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			// continuation byte: 10xxxxxx
			continue
		}

		if (c & 0xe0) == 0xc0 {
			// 2-byte character: 110xxxxx ...
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			// 3-byte character: 1110xxxx ...
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			// 4-byte character: 11110xxx ...
			incomplete = i < 4
		}

		// else 1-byte character or invalid byte
		break
	}

	return incomplete
}

func flushPending(seq *Sequence) bool {
	for _, p := range seq.pendingResponses {
		select {
		case seq.responses <- p:
		case <-seq.quit:
			return false
		}
	}

	return true
}

func (s *Server) removeSequence(seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.pendingResponses = []string{}
	seq.cache.inUse = false
	seq.samplingCtx.Free()
	s.seqs[seqIndex] = nil
}

func (s *Server) run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
			s.processBatch()
		}
	}
}

func (s *Server) processBatch() {
	batch := llama.NewBatch(s.batchSize*len(s.seqs), 0, len(s.seqs))
	defer batch.Free()

	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	slog.Debug("Processing batch", "seqs", len(s.seqs))

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted > seq.numPredict {
			s.removeSequence(i, "limit")
			continue
		}

		if seq.numPast+len(seq.tokens) > s.cache.numCtx {
			s.shiftContext(seq)
		}

		if seq.startProcessingTime.IsZero() {
			seq.startProcessingTime = time.Now()
		}

		var numTokensProcessed int
		for j, t := range seq.tokens {
			// todo: make this n_batch
			if j >= s.batchSize {
				break
			}
			batch.Add(t, seq.numPast, []int{seq.cache.id}, numTokensProcessed+1 == len(seq.tokens))
			seq.numPast++
			numTokensProcessed++
		}
		seq.cache.tokens = append(seq.cache.tokens, seq.tokens[:numTokensProcessed]...)
		seq.tokens = seq.tokens[numTokensProcessed:]
		seq.iBatch = batch.NumTokens() - 1
	}

	if batch.NumTokens() == 0 {
		return
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

		// if done processing the prompt, generate an embedding and return
		if seq.embeddingOnly {
			embd := s.lc.GetEmbeddingsSeq(i)
			if embd == nil {
				embd = s.lc.GetEmbeddingsIth(seq.iBatch)
			}

			seq.embedding <- embd
			s.removeSequence(i, "")
			continue
		}

		// sample a token
		token := seq.samplingCtx.Sample(s.lc, nil, seq.iBatch)

		seq.samplingCtx.Accept(s.lc, token, true)
		seq.numDecoded += 1
		if seq.numDecoded == 1 {
			seq.startGenerationTime = time.Now()
		}
		piece := s.model.TokenToPiece(token)

		seq.numPredicted++

		slog.Debug("sampled", "piece", piece)

		// if it's an end of sequence token, break
		if s.model.TokenIsEog(token) {
			// TODO (jmorganca): we should send this back
			// as it's important for the /api/generate context
			// seq.responses <- piece

			s.removeSequence(i, "stop")
			continue
		}

		seq.tokens = []int{token}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := findStop(sequence, seq.stop); ok {
			slog.Info("hit stop token", "stop", seq.stop)

			seq.pendingResponses = truncateStop(seq.pendingResponses, stop)
			s.removeSequence(i, "stop")
			continue
		}

		if containsStopSuffix(sequence, seq.stop) {
			continue
		}

		if incompleteUnicode(sequence) {
			continue
		}

		if !flushPending(seq) {
			seq.pendingResponses = []string{}
			s.removeSequence(i, "connection")
		}
		seq.pendingResponses = []string{}
	}
}

type Options struct {
	api.Runner

	NumKeep          int      `json:"n_keep"`
	Seed             int      `json:"seed"`
	NumPredict       int      `json:"n_predict"`
	TopK             int      `json:"top_k"`
	TopP             float32  `json:"top_p"`
	MinP             float32  `json:"min_p"`
	TFSZ             float32  `json:"tfs_z"`
	TypicalP         float32  `json:"typical_p"`
	RepeatLastN      int      `json:"repeat_last_n"`
	Temperature      float32  `json:"temperature"`
	RepeatPenalty    float32  `json:"repeat_penalty"`
	PresencePenalty  float32  `json:"presence_penalty"`
	FrequencyPenalty float32  `json:"frequency_penalty"`
	Mirostat         int      `json:"mirostat"`
	MirostatTau      float32  `json:"mirostat_tau"`
	MirostatEta      float32  `json:"mirostat_eta"`
	PenalizeNewline  bool     `json:"penalize_nl"`
	Stop             []string `json:"stop"`
}

type CompletionRequest struct {
	Prompt  string   `json:"prompt"`
	Images  []string `json:"images"`
	Grammar string   `json:"grammar"`

	Options
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
	req.Options = Options(api.DefaultOptions())
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
	samplingParams.MinP = req.MinP
	samplingParams.TfsZ = req.TFSZ
	samplingParams.TypicalP = req.TypicalP
	samplingParams.Temp = req.Temperature
	samplingParams.RepeatLastN = req.RepeatLastN
	samplingParams.PenaltyRepeat = req.RepeatPenalty
	samplingParams.PenaltyFreq = req.FrequencyPenalty
	samplingParams.PenaltyPresent = req.PresencePenalty
	samplingParams.Mirostat = req.Mirostat
	samplingParams.MirostatTau = req.MirostatTau
	samplingParams.MirostatEta = req.MirostatEta
	samplingParams.PenalizeNl = req.PenalizeNewline
	samplingParams.Seed = uint32(req.Seed)
	samplingParams.Grammar = req.Grammar

	seq := s.NewSequence(req.Prompt, NewSequenceParams{
		numPredict:     req.NumPredict,
		stop:           req.Stop,
		numKeep:        req.NumKeep,
		samplingParams: &samplingParams,
		embedding:      false,
	})

	// TODO (jmorganca): add to sequence queue instead of
	// failing if a slot isn't available
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.tokens, seq.numPast = s.cache.LoadCacheSlot(seq.tokens)
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
			close(seq.quit)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			close(seq.quit)
			return
		}

		flusher.Flush()
	}

	// Send the stop
	if err := json.NewEncoder(w).Encode(&CompletionResponse{
		Stop:         true,
		StoppedLimit: seq.doneReason == "limit",
		Timings: Timings{
			PromptN:     seq.numPromptTokens,
			PromptMS:    float64(seq.startGenerationTime.Sub(seq.startProcessingTime).Milliseconds()),
			PredictedN:  seq.numDecoded,
			PredictedMS: float64(time.Since(seq.startGenerationTime).Milliseconds()),
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
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	slog.Debug("embedding request", "content", req.Content)

	seq := s.NewSequence(req.Content, NewSequenceParams{embedding: true})

	// TODO (jessegross): Wait for a free slot instead of failing and blocking forever
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.tokens, seq.numPast = s.cache.LoadCacheSlot(seq.tokens)
			s.seqs[i] = seq
			s.cond.Signal()
			break
		}
	}
	s.mu.Unlock()

	embedding := <-seq.embedding

	if err := json.NewEncoder(w).Encode(&EmbeddingResponse{
		Embedding: embedding,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

type ServerStatus int

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusLoadingModel
	ServerStatusError
)

func (s ServerStatus) ToString() string {
	switch s {
	case ServerStatusReady:
		return "ok"
	case ServerStatusLoadingModel:
		return "loading model"
	default:
		return "server error"
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&HealthResponse{
		Status:   s.status.ToString(),
		Progress: s.progress,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

func (s *Server) loadModel(
	params llama.ModelParams,
	mpath string,
	lpath string,
	ppath string,
	kvSize int,
	flashAttention bool,
	threads int,
) {
	llama.BackendInit()

	s.model = llama.LoadModelFromFile(mpath, params)

	ctxParams := llama.NewContextParams(kvSize, s.batchSize, s.parallel, threads, flashAttention)
	s.lc = llama.NewContextWithModel(s.model, ctxParams)

	if lpath != "" {
		err := s.model.ApplyLoraFromFile(s.lc, lpath, 1.0, threads)
		if err != nil {
			panic(err)
		}
	}

	if s.model.AddBOSToken() {
		s.bosToken = 1
	}

	if ppath != "" {
		s.cc = llama.NewClipContext(ppath)
		// TODO (jessegross): Vision model support
		panic("Vision models are not yet supported")
	}

	s.cache = NewTokenCache(s.lc, kvSize, s.parallel)

	s.status = ServerStatusReady
	s.ready.Done()
}

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("mmproj", "", "Path to projector binary file")
	parallel := flag.Int("parallel", 1, "Number of sequences to handle simultaneously")
	batchSize := flag.Int("batch-size", 512, "Batch size")
	nGpuLayers := flag.Int("n-gpu-layers", 0, "Number of layers to offload to GPU")
	mainGpu := flag.Int("main-gpu", 0, "Main GPU")
	flashAttention := flag.Bool("flash-attn", false, "Enable flash attention")
	kvSize := flag.Int("ctx-size", 2048, "Context (or KV cache) size")
	lpath := flag.String("lora", "", "Path to lora layer file")
	port := flag.Int("port", 8080, "Port to expose the server on")
	threads := flag.Int("threads", runtime.NumCPU(), "Number of threads to use during generation")
	verbose := flag.Bool("verbose", false, "verbose output (default: disabled)")
	noMmap := flag.Bool("no-mmap", false, "do not memory-map model (slower load but may reduce pageouts if not using mlock)")
	mlock := flag.Bool("mlock", false, "force system to keep model in RAM rather than swapping or compressing")
	tensorSplit := flag.String("tensor-split", "", "fraction of the model to offload to each GPU, comma-separated list of proportions")

	// These are either ignored by llama.cpp or have no significance to us
	_ = flag.Bool("embedding", false, "enable embedding vector output (default: disabled)")
	_ = flag.Bool("log-disable", false, "disables logging to a file")
	_ = flag.Bool("memory-f32", false, "use f32 instead of f16 for memory key+value (default: disabled) not recommended: doubles context memory required and no measurable increase in quality")

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

	server := &Server{
		batchSize: *batchSize,
		parallel:  *parallel,
		seqs:      make([]*Sequence, *parallel),
		status:    ServerStatusLoadingModel,
	}

	var tensorSplitFloats []float32
	if *tensorSplit != "" {
		stringFloats := regexp.MustCompile(",").Split(*tensorSplit, -1)

		tensorSplitFloats = make([]float32, 0, len(stringFloats))
		for _, s := range stringFloats {
			f, _ := strconv.ParseFloat(s, 32)
			tensorSplitFloats = append(tensorSplitFloats, float32(f))
		}
	}

	params := llama.ModelParams{
		NumGpuLayers: *nGpuLayers,
		MainGpu:      *mainGpu,
		UseMmap:      !*noMmap && *lpath == "",
		UseMlock:     *mlock,
		TensorSplit:  tensorSplitFloats,
		Progress: func(progress float32) {
			slog.Debug("Loading model", "progress %", math.Round(float64(progress*100)))
			server.progress = progress
		},
	}

	server.ready.Add(1)
	go server.loadModel(params, *mpath, *lpath, *ppath, *kvSize, *flashAttention, *threads)

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

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}

	cancel()
}
