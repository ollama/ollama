package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
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

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type input struct {
	token int

	// embed is an image embedding
	embed []float32
}

type Sequence struct {
	// number of inputs evaluated
	numPast int

	// batch index
	iBatch int

	// number of tokens predicted so far
	numPredicted int

	// prompt inputs left to evaluate
	inputs []input

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	pendingResponses []string

	// input cache being used by this sequence
	cache *InputCacheSlot

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

	// number of inputs to keep at the beginning when shifting context window
	numKeep int

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	doneReason string

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptInputs     int
}

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
}

func (s *Server) NewSequence(prompt string, images []ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	startTime := time.Now()

	inputs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = len(inputs)
	}

	if !params.embedding {
		// Subtracting 4 ensures that at least 1 input can be discarded during shift
		params.numKeep = min(params.numKeep, s.cache.numCtx-4)
		params.numKeep += s.bosToken
	} else {
		// Embeddings are 1 shot - just truncate to the context window, without ever shifting
		params.numKeep = min(params.numKeep, s.cache.numCtx)
	}

	// truncate to fit in context window
	if len(inputs) > s.cache.numCtx {
		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "numKeep", params.numKeep)
		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[len(inputs)-s.cache.numCtx+params.numKeep:]...)
		inputs = newInputs
	}

	var sc *llama.SamplingContext
	if params.samplingParams != nil {
		sc = llama.NewSamplingContext(s.model, *params.samplingParams)
		for _, input := range inputs {
			if input.embed == nil {
				sc.Accept(input.token, false)
			}
		}
	}

	return &Sequence{
		inputs:              inputs,
		numPromptInputs:     len(inputs),
		startProcessingTime: startTime,
		numPredict:          params.numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
		embedding:           make(chan []float32, 1),
		samplingCtx:         sc,
		embeddingOnly:       params.embedding,
		stop:                params.stop,
		numKeep:             params.numKeep,
	}, nil
}

// inputs processes the prompt and images into a list of inputs
// by splitting the prompt on [img-<n>] tags, tokenizing text and
// generating image embeddings for each image
func (s *Server) inputs(prompt string, images []ImageData) ([]input, error) {
	var inputs []input

	re := regexp.MustCompile(`\[img-(\d+)\]`)
	parts := re.Split(prompt, -1)
	matches := re.FindAllStringSubmatch(prompt, -1)

	for i, part := range parts {
		// text - tokenize
		if strings.TrimSpace(part) != "" {
			tokens, err := s.lc.Model().Tokenize(part, i == 0, true)
			if err != nil {
				return nil, err
			}

			for _, t := range tokens {
				inputs = append(inputs, input{token: t})
			}
		}

		// image - generate image embedding
		if i < len(matches) {
			n, _ := strconv.Atoi(matches[i][1])

			imageIndex := -1
			for j := range images {
				if images[j].ID == n {
					imageIndex = j
					break
				}
			}

			if imageIndex < 0 {
				return nil, fmt.Errorf("invalid image index: %d", n)
			}

			hash := s.cache.HashImage(images[imageIndex].Data)

			// Vision models cannot be accessed concurrently
			s.clip.mu.Lock()
			embed, err := s.cache.FindImage(hash)
			if err != nil {
				embed = llama.NewLlavaImageEmbed(s.lc, s.clip.cc, images[imageIndex].Data)
				s.cache.AddImage(hash, embed)
			}
			s.clip.mu.Unlock()

			for _, e := range embed {
				inputs = append(inputs, input{embed: e})
			}
		}
	}

	return inputs, nil
}

type clip struct {
	cc *llama.ClipContext
	mu sync.Mutex
}

type Server struct {
	model *llama.Model
	lc    *llama.Context

	// required for image embeddings
	clip clip

	batchSize int

	// parallel is the number of parallel requests to handle
	parallel int

	// seqs is the list of parallel sequences being evaluated
	// TODO (jmorganca): this can probably be moved into run()
	seqs []*Sequence

	// KV cache
	cache *InputCache

	// does this model require a beginning of sequence token?
	bosToken int

	// next sequence for prompt processing to avoid starvation
	nextSeq int

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

func flushPending(seq *Sequence) bool {
	for _, p := range seq.pendingResponses {
		select {
		case seq.responses <- p:
		case <-seq.quit:
			seq.pendingResponses = []string{}
			return false
		}
	}

	seq.pendingResponses = []string{}
	return true
}

func (s *Server) removeSequence(seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
}

func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	// logically these batches are used only within the context of processBatch
	// but it is better for performance to allocate them once here
	tokenBatch := llama.NewBatch(s.batchSize*len(s.seqs), 0, len(s.seqs))
	defer tokenBatch.Free()

	embedBatch := llama.NewBatch(s.batchSize*len(s.seqs), s.lc.Model().NEmbd(), len(s.seqs))
	defer embedBatch.Free()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			s.processBatch(tokenBatch, embedBatch)
			tokenBatch.Clear()
			embedBatch.Clear()
		}
	}
}

// TODO (jmorganca): processBatch should be simplified, removing:
// * sampling
// * stop token checking
// * metrics
// these should instead be handled by the handlers
// it should only be responsible for accepting tokens or embeddings and
// processing batches as fast as possible
func (s *Server) processBatch(tokenBatch *llama.Batch, embedBatch *llama.Batch) {
	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	var batch *llama.Batch

	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]

		if seq == nil {
			continue
		}

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted > seq.numPredict {
			s.removeSequence(seqIdx, "limit")
			continue
		}

		if seq.numPast+len(seq.inputs) > s.cache.numCtx {
			s.shiftContext(seq)
		}

		var numInputsProcessed int
		for i, input := range seq.inputs {
			embedding := input.embed != nil

			// If we don't currently have a batch, use one of the correct type and
			// fill it up as much as possible across all sequences. If we encounter an
			// input of the opppsite type, stop for that sequence but then pick up from
			// there for the next batch, ensuring that we alternate types
			if batch == nil {
				if !embedding {
					batch = tokenBatch
				} else {
					batch = embedBatch
				}
			} else if embedding != batch.IsEmbedding() {
				s.nextSeq = seqIdx
				break
			}

			// todo: make this n_batch
			if i >= s.batchSize {
				break
			}

			batch.Add(input.token, input.embed, seq.numPast, []int{seq.cache.Id}, numInputsProcessed+1 == len(seq.inputs))
			seq.numPast++
			numInputsProcessed++
		}

		if numInputsProcessed > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.inputs[:numInputsProcessed]...)
			seq.inputs = seq.inputs[numInputsProcessed:]
			seq.iBatch = batch.NumTokens() - 1
		}
	}

	if batch == nil || batch.NumTokens() == 0 {
		return
	}

	err := s.lc.Decode(batch)
	if err != nil {
		slog.Error("failed to decode batch", "error", err)
		return
	}

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// don't sample prompt processing
		if len(seq.inputs) != 0 {
			continue
		}

		seq.numDecoded += 1
		if seq.numDecoded == 1 {
			seq.startGenerationTime = time.Now()
		}

		// if done processing the prompt, generate an embedding and return
		if seq.embeddingOnly {
			embed := s.lc.GetEmbeddingsSeq(i)
			if embed == nil {
				embed = s.lc.GetEmbeddingsIth(seq.iBatch)
			}

			seq.embedding <- embed
			s.removeSequence(i, "")
			continue
		}

		// sample a token
		token := seq.samplingCtx.Sample(s.lc, seq.iBatch)
		seq.samplingCtx.Accept(token, true)
		piece := s.model.TokenToPiece(token)

		seq.numPredicted++

		// if it's an end of sequence token, break
		if s.model.TokenIsEog(token) {
			// TODO (jmorganca): we should send this back
			// as it's important for the /api/generate context
			// seq.responses <- piece

			s.removeSequence(i, "stop")
			continue
		}

		seq.inputs = []input{{token: token}}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := findStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = truncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Update the cache based on the tokens that will be returned:
			// - We have 1 token more than is currently in the cache because
			// the last one generated wasn't submitted to Decode
			// - Remove any stop sequences that we stripped out
			// - If truncateStop removed a portion of a token, drop that
			// - As defense-in-depth, if truncatedToken didn't find a stop token
			// remove the extra one that we added to the cache len
			tokenLen := len(seq.cache.Inputs) + 1
			tokenLen -= origLen - newLen
			if tokenTruncated || origLen == newLen {
				tokenLen--
			}
			seq.cache.Inputs = seq.cache.Inputs[:tokenLen]

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
			s.removeSequence(i, "connection")
		}
	}
}

// TODO (jmorganca): use structs from the api package to avoid duplication
// this way the api acts as a proxy instead of using a different api for the
// runner
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

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

type CompletionRequest struct {
	Prompt      string      `json:"prompt"`
	Images      []ImageData `json:"image_data"`
	Grammar     string      `json:"grammar"`
	CachePrompt bool        `json:"cache_prompt"`

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

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

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

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:     req.NumPredict,
		stop:           req.Stop,
		numKeep:        req.NumKeep,
		samplingParams: &samplingParams,
		embedding:      false,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// TODO (jmorganca): add to sequence queue instead of
	// failing if a slot isn't available
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, seq.numPast, err = s.cache.LoadCacheSlot(seq.inputs, req.CachePrompt)
			if err != nil {
				s.mu.Unlock()
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}
			s.seqs[i] = seq
			s.cond.Signal()
			break
		}
	}
	s.mu.Unlock()

	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case content, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Content: content,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				// Send the final response
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Stop:         true,
					StoppedLimit: seq.doneReason == "limit",
					Timings: Timings{
						PromptN:     seq.numPromptInputs,
						PromptMS:    float64(seq.startGenerationTime.Sub(seq.startProcessingTime).Milliseconds()),
						PredictedN:  seq.numDecoded,
						PredictedMS: float64(time.Since(seq.startGenerationTime).Milliseconds()),
					},
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
	}
}

type EmbeddingRequest struct {
	Content     string `json:"content"`
	CachePrompt bool   `json:"cache_prompt"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	slog.Debug("embedding request", "content", req.Content)

	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{embedding: true})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// TODO (jessegross): Wait for a free slot instead of failing and blocking forever
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, seq.numPast, err = s.cache.LoadCacheSlot(seq.inputs, req.CachePrompt)
			if err != nil {
				s.mu.Unlock()
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}
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
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
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
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
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
	multiUserCache bool,
) {
	llama.BackendInit()

	s.model = llama.LoadModelFromFile(mpath, params)

	ctxParams := llama.NewContextParams(kvSize, s.batchSize*s.parallel, s.parallel, threads, flashAttention)
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
		s.clip.cc = llama.NewClipContext(ppath)
	}

	s.cache = NewInputCache(s.lc, kvSize, s.parallel, multiUserCache)

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
	multiUserCache := flag.Bool("multiuser-cache", false, "optimize input cache algorithm for multiple users")
	// Expose requirements as a JSON output to stdout
	requirements := flag.Bool("requirements", false, "print json requirement information")

	// These are either ignored by llama.cpp or have no significance to us
	_ = flag.Bool("embedding", false, "enable embedding vector output (default: disabled)")
	_ = flag.Bool("log-disable", false, "disables logging to a file")
	_ = flag.Bool("memory-f32", false, "use f32 instead of f16 for memory key+value (default: disabled) not recommended: doubles context memory required and no measurable increase in quality")

	flag.Parse()
	if *requirements {
		printRequirements(os.Stdout)
		return
	}
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
	slog.Info("starting go runner")
	slog.Debug("system info", "cpu", llama.PrintSystemInfo(), "threads", *threads)

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
			server.progress = progress
		},
	}

	server.ready.Add(1)
	go server.loadModel(params, *mpath, *lpath, *ppath, *kvSize, *flashAttention, *threads, *multiUserCache)

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
