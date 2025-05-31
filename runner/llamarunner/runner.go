package llamarunner

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
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/runner/common"
)

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type input struct {
	token int

	// embed is an image embedding
	embed []float32
}

type Sequence struct {
	// batch index
	iBatch int

	// number of tokens predicted so far
	numPredicted int

	// prompt inputs left to evaluate
	inputs []input

	// inputs that have been added to a batch but not yet submitted to Decode
	pendingInputs []input

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	pendingResponses []llm.CompletionResponse

	// input cache being used by this sequence
	cache *InputCacheSlot

	// channel to send responses over
	responses chan llm.CompletionResponse

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

	doneReason llm.DoneReason

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptInputs     int
}

var _ common.ResponseSequence = (*Sequence)(nil)

func (seq *Sequence) Pending() []llm.CompletionResponse {
	return seq.pendingResponses
}

func (seq *Sequence) Clear() {
	seq.pendingResponses = []llm.CompletionResponse{}
}

func (seq *Sequence) Send(resp llm.CompletionResponse) bool {
	if len(resp.Content) > 0 || resp.Done {
		select {
		case seq.responses <- resp:
			// Successfully sent
			return true
		case <-seq.quit:
			return false
		}
	}
	return true
}

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
}

func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
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

	if s.model.AddBOSToken() {
		params.numKeep += 1
	}

	// Ensure that at least 1 input can be discarded during shift
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	if len(inputs) > s.cache.numCtx {
		discard := len(inputs) - s.cache.numCtx
		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[params.numKeep+discard:]...)

		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	var sc *llama.SamplingContext
	if params.samplingParams != nil {
		sc, err = llama.NewSamplingContext(s.model, *params.samplingParams)
		if err != nil {
			return nil, err
		}
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
		pendingResponses:    make([]llm.CompletionResponse, 0),
		responses:           make(chan llm.CompletionResponse, 100),
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
func (s *Server) inputs(prompt string, images []llm.ImageData) ([]input, error) {
	var inputs []input
	var parts []string
	var matches [][]string

	if s.image != nil {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
	} else {
		parts = []string{prompt}
	}

	for i, part := range parts {
		// text - tokenize
		tokens, err := s.lc.Model().Tokenize(part, i == 0, true)
		if err != nil {
			return nil, err
		}

		for _, t := range tokens {
			inputs = append(inputs, input{token: t})
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

			embed, err := s.image.NewEmbed(s.lc, images[imageIndex].Data)
			if err != nil {
				return nil, err
			}

			for _, e := range embed {
				inputs = append(inputs, input{embed: e})
			}
		}
	}

	return inputs, nil
}

type Server struct {
	// is the server ready to process requests?
	// protects access to model and image
	ready sync.WaitGroup

	// loaded model
	model *llama.Model

	// image model context for multi-modal models
	image *ImageContext

	// status for external health reporting - loading, ready to serve, etc.
	status llm.ServerStatus

	// current progress on loading the model
	progress float32

	// number of simultaneous requests to handle
	parallel int

	// maximum number of elements in a batch (per sequence)
	// TODO (jmorganca): make this n_batch
	batchSize int

	// protects access to everything below this line
	// this is context state needed for decoding
	mu sync.Mutex

	// indicates that data is ready for processing
	cond *sync.Cond

	// decoding state
	lc *llama.Context

	// the list of simultaneous sequences being evaluated
	seqs []*Sequence

	// seqs can have a maximum of parallel entries, which
	// is enfoced by seqSem
	seqsSem *semaphore.Weighted

	// KV cache
	cache *InputCache

	// next sequence for prompt processing to avoid starvation
	nextSeq int
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func (s *Server) removeSequence(seqIndex int, reason llm.DoneReason) {
	seq := s.seqs[seqIndex]

	common.FlushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
	s.seqsSem.Release(1)
}

func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	// Logically these batches are used only within the context of processBatch
	// but it is better for performance to allocate them once here
	tokenBatch, err := llama.NewBatch(s.batchSize, len(s.seqs), 0)
	if err != nil {
		panic(err)
	}
	defer tokenBatch.Free()

	var embedBatch *llama.Batch
	embedBatchSize := s.image.BatchSize(s.batchSize)
	if embedBatchSize != 0 {
		embedBatch, err = llama.NewBatch(embedBatchSize, len(s.seqs), s.image.EmbedSize(s.lc))
		if err != nil {
			panic(err)
		}
		defer embedBatch.Free()
	} else {
		embedBatch = &llama.Batch{}
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
			err := s.processBatch(tokenBatch, embedBatch)
			if err != nil {
				panic(err)
			}

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
func (s *Server) processBatch(tokenBatch *llama.Batch, embedBatch *llama.Batch) error {
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
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(seqIdx, llm.DoneReasonLength)
			continue
		}

		for i, input := range seq.inputs {
			if len(seq.cache.Inputs)+len(seq.pendingInputs)+1 > s.cache.numCtx {
				if len(seq.pendingInputs) == 0 {
					err := s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
					if err != nil {
						var reprocess *ErrReprocessInputs
						if errors.As(err, &reprocess) {
							// Prepend these inputs to the sequence's inputs queue for reprocessing
							seq.inputs = append(reprocess.Inputs, seq.inputs...)
							// Continue processing as normal
							continue
						} else {
							return err
						}
					}
				} else {
					break
				}
			}

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

			if i >= batch.Size() {
				break
			}

			batch.Add(input.token, input.embed, len(seq.cache.Inputs)+len(seq.pendingInputs), i+1 == len(seq.inputs), seq.cache.Id)
			seq.pendingInputs = append(seq.pendingInputs, input)
			seq.iBatch = batch.NumTokens() - 1
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	if batch == nil || batch.NumTokens() == 0 {
		return nil
	}

	err := s.lc.Decode(batch)
	if err != nil {
		return fmt.Errorf("failed to decode batch: %w", err)
	}

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// After calling Decode, pending inputs are now in the cache
		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []input{}
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
			embed := s.lc.GetEmbeddingsSeq(seq.cache.Id)
			if embed == nil {
				embed = s.lc.GetEmbeddingsIth(seq.iBatch)
			}

			seq.embedding <- embed
			s.removeSequence(i, llm.DoneReasonStop)
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

			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		seq.inputs = []input{{token: token}}

		seq.pendingResponses = append(seq.pendingResponses, llm.CompletionResponse{Content: piece})
		sequence := ""
		for _, r := range seq.pendingResponses {
			sequence += r.Content
		}

		if ok, stop := common.FindStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = common.TruncateStop(seq.pendingResponses, stop)
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

			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		if common.ContainsStopSuffix(sequence, seq.stop) {
			continue
		}

		if common.IncompleteUnicode(sequence) {
			continue
		}

		if !common.FlushPending(seq) {
			s.removeSequence(i, llm.DoneReasonConnectionClosed)
		}
	}

	return nil
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req llm.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Extract options from the CompletionRequest
	samplingParams := llama.SamplingParams{
		TopK:           req.Options.TopK,
		TopP:           req.Options.TopP,
		MinP:           req.Options.MinP,
		TypicalP:       req.Options.TypicalP,
		Temp:           req.Options.Temperature,
		RepeatLastN:    req.Options.RepeatLastN,
		PenaltyRepeat:  req.Options.RepeatPenalty,
		PenaltyFreq:    req.Options.FrequencyPenalty,
		PenaltyPresent: req.Options.PresencePenalty,
		Seed:           uint32(req.Options.Seed),
		Grammar:        req.Grammar,
	}

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:     req.Options.NumPredict,
		stop:           req.Options.Stop,
		numKeep:        req.Options.NumKeep,
		samplingParams: &samplingParams,
		embedding:      false,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Ensure there is a place to put the sequence, released when removed from s.seqs
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("Failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, true)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}

			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case content, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&content); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Done:               true,
					DoneReason:         seq.doneReason,
					PromptEvalCount:    seq.numPromptInputs,
					PromptEvalDuration: seq.startGenerationTime.Sub(seq.startProcessingTime),
					EvalCount:          seq.numDecoded,
					EvalDuration:       time.Since(seq.startGenerationTime),
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
	}
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req llm.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{embedding: true})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Ensure there is a place to put the sequence, released when removed from s.seqs
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embeddings request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("Failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, false)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}
			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	embedding := <-seq.embedding

	if err := json.NewEncoder(w).Encode(&llm.EmbeddingResponse{
		Embedding: embedding,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&llm.ServerStatusResponse{
		Status:   s.status,
		Progress: s.progress,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

type multiLPath []string

func (m *multiLPath) Set(value string) error {
	*m = append(*m, value)
	return nil
}

func (m *multiLPath) String() string {
	return strings.Join(*m, ", ")
}

func (s *Server) loadModel(
	params llama.ModelParams,
	mpath string,
	lpath multiLPath,
	ppath string,
	kvSize int,
	kvCacheType string,
	flashAttention bool,
	threads int,
	multiUserCache bool,
) {
	var err error
	s.model, err = llama.LoadModelFromFile(mpath, params)
	if err != nil {
		panic(err)
	}

	ctxParams := llama.NewContextParams(kvSize, s.batchSize*s.parallel, s.parallel, threads, flashAttention, kvCacheType)
	s.lc, err = llama.NewContextWithModel(s.model, ctxParams)
	if err != nil {
		panic(err)
	}

	if lpath.String() != "" {
		for _, path := range lpath {
			err := s.model.ApplyLoraFromFile(s.lc, path, 1.0, threads)
			if err != nil {
				panic(err)
			}
		}
	}

	if ppath != "" {
		var err error
		s.image, err = NewImageContext(s.lc, ppath)
		if err != nil {
			panic(err)
		}
	}

	s.cache, err = NewInputCache(s.lc, kvSize, s.parallel, multiUserCache)
	if err != nil {
		panic(err)
	}

	s.status = llm.ServerStatusReady
	s.ready.Done()
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	ppath := fs.String("mmproj", "", "Path to projector binary file")
	parallel := fs.Int("parallel", 1, "Number of sequences to handle simultaneously")
	batchSize := fs.Int("batch-size", 512, "Batch size")
	nGpuLayers := fs.Int("n-gpu-layers", 0, "Number of layers to offload to GPU")
	mainGpu := fs.Int("main-gpu", 0, "Main GPU")
	flashAttention := fs.Bool("flash-attn", false, "Enable flash attention")
	kvSize := fs.Int("ctx-size", 2048, "Context (or KV cache) size")
	kvCacheType := fs.String("kv-cache-type", "", "quantization type for KV cache (default: f16)")
	port := fs.Int("port", 8080, "Port to expose the server on")
	threads := fs.Int("threads", runtime.NumCPU(), "Number of threads to use during generation")
	_ = fs.Bool("verbose", false, "verbose output (default: disabled)")
	noMmap := fs.Bool("no-mmap", false, "do not memory-map model (slower load but may reduce pageouts if not using mlock)")
	tensorSplit := fs.String("tensor-split", "", "fraction of the model to offload to each GPU, comma-separated list of proportions")
	multiUserCache := fs.Bool("multiuser-cache", false, "optimize input cache algorithm for multiple users")

	var lpaths multiLPath
	fs.Var(&lpaths, "lora", "Path to lora layer file (can be specified multiple times)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting go runner")

	llama.BackendInit()

	server := &Server{
		batchSize: *batchSize,
		parallel:  *parallel,
		seqs:      make([]*Sequence, *parallel),
		seqsSem:   semaphore.NewWeighted(int64(*parallel)),
		status:    llm.ServerStatusLoadingModel,
	}

	var tensorSplitFloats []float32
	if *tensorSplit != "" {
		splits := strings.Split(*tensorSplit, ",")
		tensorSplitFloats = make([]float32, len(splits))
		for i, s := range splits {
			f, _ := strconv.ParseFloat(s, 32)
			tensorSplitFloats[i] = float32(f)
		}
	}

	params := llama.ModelParams{
		NumGpuLayers: *nGpuLayers,
		MainGpu:      *mainGpu,
		UseMmap:      !*noMmap && lpaths.String() == "",
		TensorSplit:  tensorSplitFloats,
		Progress: func(progress float32) {
			server.progress = progress
		},
	}

	server.ready.Add(1)
	go server.loadModel(params, *mpath, lpaths, *ppath, *kvSize, *kvCacheType, *flashAttention, *threads, *multiUserCache)

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
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
		return err
	}

	return nil
}
