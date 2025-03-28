package ollamarunner

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/maphash"
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
	"unicode/utf8"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/runner/common"
	"github.com/ollama/ollama/sample"

	_ "github.com/ollama/ollama/model/models"
)

type contextList struct {
	list []ml.Context
}

type Sequence struct {
	// ctxs are used for allocating tensors that last the lifetime of the sequence, such as
	// multimodal embeddings
	ctxs *contextList

	// batch index
	iBatch int

	// prompt inputs left to evaluate
	inputs []input.Input

	// inputs that have been added to a batch but not yet submitted to Forward
	pendingInputs []input.Input

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

	// sampler with transforms to run on generated logits
	sampler sample.Sampler

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// stop sequences
	stop []string

	// number of inputs to keep at the beginning when shifting context window
	numKeep int32

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	doneReason string

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numPredicted        int
	numPromptInputs     int
}

type NewSequenceParams struct {
	numPredict int
	stop       []string
	numKeep    int32
	sampler    sample.Sampler
	embedding  bool
}

func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	startTime := time.Now()

	inputs, ctxs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = int32(len(inputs))
	}

	// TODO(jessegross): We should ensure that we always leave minBatch of context space to shift,
	// otherwise we might truncate or split the batch against the model's wishes

	// Ensure that at least 1 input can be discarded during shift
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	if int32(len(inputs)) > s.cache.numCtx {
		discard := int32(len(inputs)) - s.cache.numCtx
		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[params.numKeep+discard:]...)

		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	// TODO(jessegross): Ingest cached history for grammar

	return &Sequence{
		ctxs:                ctxs,
		inputs:              inputs,
		numPromptInputs:     len(inputs),
		startProcessingTime: startTime,
		numPredict:          params.numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
		embedding:           make(chan []float32, 1),
		sampler:             params.sampler,
		embeddingOnly:       params.embedding,
		stop:                params.stop,
		numKeep:             params.numKeep,
	}, nil
}

// inputs processes the prompt and images into a list of inputs
// by splitting the prompt on [img-<n>] tags, tokenizing text and
// decoding images
func (s *Server) inputs(prompt string, images []llm.ImageData) ([]input.Input, *contextList, error) {
	var inputs []input.Input
	var parts []string
	var matches [][]string

	multimodalProcessor, visionModel := s.model.(model.MultimodalProcessor)

	if visionModel {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
	} else {
		parts = []string{prompt}
	}

	var contexts contextList
	runtime.AddCleanup(&contexts, func(ctxs []ml.Context) {
		for _, ctx := range ctxs {
			ctx.Close()
		}
	}, contexts.list)

	postTokenize := false
	for i, part := range parts {
		// text - tokenize
		tokens, err := s.model.(model.TextProcessor).Encode(part, i == 0)
		if err != nil {
			return nil, nil, err
		}

		for _, t := range tokens {
			inputs = append(inputs, input.Input{Token: t})
		}

		// image - decode and store
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
				return nil, nil, fmt.Errorf("invalid image index: %d", n)
			}

			ctx := s.model.Backend().NewContext()
			contexts.list = append(contexts.list, ctx)
			imageEmbeddings, err := multimodalProcessor.EncodeMultimodal(ctx, images[imageIndex].Data)
			if err != nil {
				return nil, nil, err
			}

			s.multimodalHash.Reset()
			_, _ = s.multimodalHash.Write(images[imageIndex].Data)
			imageHash := s.multimodalHash.Sum64()

			inputs = append(inputs, input.Input{Multimodal: imageEmbeddings, MultimodalHash: imageHash})
			postTokenize = true
		}
	}

	if visionModel && postTokenize {
		var err error
		inputs, err = multimodalProcessor.PostTokenize(inputs)
		if err != nil {
			return nil, nil, err
		}
	}

	return inputs, &contexts, nil
}

type Server struct {
	// is the server ready to process requests?
	// protects access to model and image
	ready sync.WaitGroup

	// loaded model
	model model.Model

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

	// the list of simultaneous sequences being evaluated
	seqs []*Sequence

	// seqs can have a maximum of parallel entries, which
	// is enfoced by seqSem
	seqsSem *semaphore.Weighted

	// KV cache
	cache *InputCache

	// multimodalHash generates hashes for comparing equality
	// of non-text data
	multimodalHash maphash.Hash

	// vocab is a llama.cpp vocab required for gammar-based
	// constrained generation (json mode, structured outputs)
	// TODO: this is temporary until Ollama sampling supports
	// constrained generation
	vocab *sample.Vocab
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	seq.pendingResponses = []string{}

	// Check if there are any partial UTF-8 characters remaining.
	// We already check and queue as we are generating but some may
	// still make it here:
	// - Sequence is ending, e.g. generation limit has been hit
	// - Invalid characters in the middle of a string
	// This is a stricter check to ensure we never output invalid Unicode.
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- joined:
		return true
	case <-seq.quit:
		return false
	}
}

func (s *Server) removeSequence(seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
	s.seqsSem.Release(1)
}

func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			err := s.processBatch()
			if err != nil {
				panic(err)
			}
		}
	}
}

func (s *Server) processBatch() error {
	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	var batchInputs []int32
	var batch input.Batch

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(i, "limit")
			continue
		}

		if !s.cache.enabled {
			seq.inputs = append(seq.cache.Inputs, seq.inputs...)
			seq.cache.Inputs = []input.Input{}
		}

		batchSize := s.batchSize

		for j, inp := range seq.inputs {
			// If we are required to put following inputs into a single batch then extend the
			// batch size. Since we are only extending the size the minimum amount possible, this
			// will cause a break if we have pending inputs.
			minBatch := 1 + inp.SameBatch
			if minBatch > batchSize {
				batchSize = minBatch
			}

			if len(seq.pendingInputs)+minBatch > batchSize {
				break
			}

			// If the sum of our working set (already processed tokens, tokens we added to this
			// batch, required following tokens) exceeds the context size, then trigger a shift
			// now so we don't have to do one later when we can't break the batch.
			if int32(len(seq.cache.Inputs)+len(seq.pendingInputs)+minBatch) > s.cache.numCtx {
				if len(seq.pendingInputs) != 0 {
					break
				}

				err := s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
				if err != nil {
					return err
				}
			}

			batchInputs = append(batchInputs, inp.Token)
			if inp.Multimodal != nil {
				batch.Multimodal = append(batch.Multimodal, input.MultimodalIndex{Index: len(batchInputs) - 1, Multimodal: inp.Multimodal})
			}

			batch.Positions = append(batch.Positions, int32(len(seq.cache.Inputs)+len(seq.pendingInputs)))
			batch.Sequences = append(batch.Sequences, seq.cache.Id)

			seq.iBatch = len(batch.Outputs)
			if j+1 == len(seq.inputs) {
				batch.Outputs = append(batch.Outputs, int32(len(batchInputs)-1))
			}
			seq.pendingInputs = append(seq.pendingInputs, inp)
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	if len(batchInputs) == 0 {
		return nil
	}

	ctx := s.model.Backend().NewContext()
	defer ctx.Close()

	modelOutput, err := model.Forward(ctx, s.model, batchInputs, batch)
	if err != nil {
		return fmt.Errorf("failed to decode batch: %w", err)
	}

	logits := modelOutput.Floats()

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// After calling Forward, pending inputs are now in the cache
		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []input.Input{}
		}

		// don't sample prompt processing
		if len(seq.inputs) != 0 {
			if !s.cache.enabled {
				return errors.New("caching disabled but unable to fit entire input in a batch")
			}
			continue
		}

		seq.numPredicted++
		if seq.numPredicted == 1 {
			seq.startGenerationTime = time.Now()
		}

		// if done processing the prompt, generate an embedding and return
		if seq.embeddingOnly {
			// TODO(jessegross): Embedding support
			slog.Warn("generation of embedding outputs not yet supported")
			s.removeSequence(i, "")
			continue
		}

		// sample a token
		vocabSize := len(logits) / len(batch.Outputs)

		token, err := seq.sampler.Sample(logits[seq.iBatch*vocabSize : (seq.iBatch+1)*vocabSize])
		if err != nil {
			return fmt.Errorf("failed to sample token: %w", err)
		}

		// if it's an end of sequence token, break
		if s.model.(model.TextProcessor).Is(token, model.SpecialEOS) {
			// TODO (jmorganca): we should send this back
			// as it's important for the /api/generate context
			// seq.responses <- piece

			s.removeSequence(i, "stop")
			continue
		}

		piece, err := s.model.(model.TextProcessor).Decode([]int32{token})
		if err != nil {
			return err
		}

		seq.inputs = []input.Input{{Token: token}}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

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

			s.removeSequence(i, "stop")
			continue
		}

		if common.ContainsStopSuffix(sequence, seq.stop) {
			continue
		}

		if common.IncompleteUnicode(sequence) {
			continue
		}

		if !flushPending(seq) {
			s.removeSequence(i, "connection")
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

	var grammar *sample.Grammar
	var err error
	if req.Grammar != "" {
		grammar, err = sample.NewGrammar(s.vocab, req.Grammar)
		if err != nil {
			http.Error(w, "failed to load model vocabulary required for format", http.StatusInternalServerError)
			return
		}
	}

	sampler := sample.NewSampler(
		req.Options.Temperature,
		req.Options.TopK,
		req.Options.TopP,
		req.Options.MinP,
		req.Options.Seed,
		grammar,
	)

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict: req.Options.NumPredict,
		stop:       req.Options.Stop,
		numKeep:    int32(req.Options.NumKeep),
		sampler:    sampler,
		embedding:  false,
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
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs)
			if err != nil {
				s.mu.Unlock()
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
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Content: content,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				// Send the final response
				doneReason := "stop"
				if seq.doneReason == "limit" {
					doneReason = "length"
				}
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Done:               true,
					DoneReason:         doneReason,
					PromptEvalCount:    seq.numPromptInputs,
					PromptEvalDuration: seq.startGenerationTime.Sub(seq.startProcessingTime),
					EvalCount:          seq.numPredicted,
					EvalDuration:       time.Since(seq.startGenerationTime),
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
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
	ctx context.Context,
	mpath string,
	params ml.BackendParams,
	lpath multiLPath,
	parallel int,
	kvCacheType string,
	kvSize int,
	multiUserCache bool,
) {
	var err error
	s.model, err = model.New(ctx, mpath, params)
	if err != nil {
		panic(err)
	}

	s.vocab = sample.NewVocab(mpath)

	// TODO(jessegross): LoRA loading
	if lpath.String() != "" {
		panic("loras are not yet implemented")
	}

	s.cache, err = NewInputCache(s.model, kvCacheType, int32(kvSize), parallel, s.batchSize, multiUserCache)
	if err != nil {
		panic(err)
	}

	if !s.cache.enabled && parallel > 1 {
		parallel = 1
		slog.Warn("model does not support caching, disabling parallel processing")
	}

	s.parallel = parallel
	s.seqs = make([]*Sequence, s.parallel)
	s.seqsSem = semaphore.NewWeighted(int64(s.parallel))

	s.status = llm.ServerStatusReady
	s.ready.Done()
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	parallel := fs.Int("parallel", 1, "Number of sequences to handle simultaneously")
	batchSize := fs.Int("batch-size", 512, "Batch size")
	numGPULayers := fs.Int("n-gpu-layers", 0, "Number of layers to offload to GPU")
	mainGPU := fs.Int("main-gpu", 0, "Main GPU")
	flashAttention := fs.Bool("flash-attn", false, "Enable flash attention")
	kvSize := fs.Int("ctx-size", 2048, "Context (or KV cache) size")
	kvCacheType := fs.String("kv-cache-type", "", "quantization type for KV cache (default: f16)")
	port := fs.Int("port", 8080, "Port to expose the server on")
	threads := fs.Int("threads", runtime.NumCPU(), "Number of threads to use during generation")
	verbose := fs.Bool("verbose", false, "verbose output (default: disabled)")
	_ = fs.Bool("no-mmap", false, "do not memory-map model (slower load but may reduce pageouts if not using mlock)")
	_ = fs.Bool("mlock", false, "force system to keep model in RAM rather than swapping or compressing")
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
	slog.Info("starting ollama engine")

	server := &Server{
		batchSize: *batchSize,
		status:    llm.ServerStatusLoadingModel,
	}

	// TODO(jessegross): Parameters that need to be implemented:
	//	no-mmap
	//	mlock

	var tensorSplitFloats []float32
	if *tensorSplit != "" {
		splits := strings.Split(*tensorSplit, ",")
		tensorSplitFloats = make([]float32, len(splits))
		for i, s := range splits {
			f, _ := strconv.ParseFloat(s, 32)
			tensorSplitFloats[i] = float32(f)
		}
	}

	params := ml.BackendParams{
		Progress: func(progress float32) {
			server.progress = progress
		},
		NumThreads:     *threads,
		NumGPULayers:   *numGPULayers,
		MainGPU:        *mainGPU,
		TensorSplit:    tensorSplitFloats,
		FlashAttention: *flashAttention,
	}

	server.ready.Add(1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go server.loadModel(ctx, *mpath, params, lpaths, *parallel, *kvCacheType, *kvSize, *multiUserCache)

	server.cond = sync.NewCond(&server.mu)

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
	}
	defer listener.Close()

	mux := http.NewServeMux()
	// TODO: support embeddings
	mux.HandleFunc("POST /embedding", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "this model does not support embeddings", http.StatusNotImplemented)
	})

	mux.HandleFunc("POST /completion", server.completion)
	mux.HandleFunc("GET /health", server.health)

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
