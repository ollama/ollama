package llamarunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/spf13/pflag"
	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/runner/common"
)

// response contains a piece of generated text along with optional logprobs
type response struct {
	content  string
	logprobs []llm.Logprob
}

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type input struct {
	token int

	// embed is an image embedding (single token for non-M-RoPE, or all image embeddings for M-RoPE)
	embed []float32

	// For M-RoPE models (Qwen3-VL), these store the image grid dimensions
	// When set, embed contains ALL embeddings for the image (nx * ny * embedSize floats)
	imageNx int
	imageNy int
}

// isImageMRoPE returns true if this input represents a complete M-RoPE image
func (inp *input) isImageMRoPE() bool {
	return inp.imageNx > 0 && inp.imageNy > 0
}

// numTokens returns the number of tokens this input represents
func (inp *input) numTokens() int {
	if inp.isImageMRoPE() {
		return inp.imageNx * inp.imageNy
	}
	return 1
}

// numPos returns the position advance for this input.
// For M-RoPE images, this is max(nx, ny) because the temporal dimension
// advances by the maximum of the spatial dimensions.
// For regular tokens/embeddings, this equals numTokens().
func (inp *input) numPos() int {
	if inp.isImageMRoPE() {
		if inp.imageNx > inp.imageNy {
			return inp.imageNx
		}
		return inp.imageNy
	}
	return 1
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
	pendingResponses []string

	// logprobs for tokens that haven't been returned yet
	pendingLogprobs []llm.Logprob

	// input cache being used by this sequence
	cache *InputCacheSlot

	// channel to send responses over
	responses chan response

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

	// shift if context window is exceeded
	shift bool

	doneReason llm.DoneReason

	// logprobs configuration
	logprobs    bool
	topLogprobs int

	// Metrics
	processingDuration time.Duration
	generationDuration time.Duration
	numDecoded         int
	numPromptInputs    int

	// Repetition loop detection
	recentTokens     []int // circular buffer of recent tokens
	recentTokensIdx  int   // current index in circular buffer
	repetitionCount  int   // count of detected repetitions
}

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
	shift          bool
	truncate       bool
	logprobs       bool
	topLogprobs    int
}

var errorInputTooLong = errors.New("the input length exceeds the context length")

func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

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
		if !params.truncate {
			return nil, errorInputTooLong
		}

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
		inputs:           inputs,
		numPromptInputs:  len(inputs),
		numPredict:       params.numPredict,
		pendingResponses: make([]string, 0),
		responses:        make(chan response, 100),
		quit:             make(chan bool, 1),
		embedding:        make(chan []float32, 1),
		samplingCtx:      sc,
		embeddingOnly:    params.embedding,
		stop:             params.stop,
		numKeep:          params.numKeep,
		shift:            params.shift,
		logprobs:         params.logprobs,
		topLogprobs:      params.topLogprobs,
		recentTokens:     make([]int, 256), // buffer for last 256 tokens
		recentTokensIdx:  0,
		repetitionCount:  0,
	}, nil
}

// detectRepetitionLoop checks if the model is stuck in a repetition loop
// by looking for repeating patterns in the recent token buffer
// Returns true if a repetition loop is detected
func (seq *Sequence) detectRepetitionLoop(token int) bool {
	bufSize := len(seq.recentTokens)
	
	// Store the token in the circular buffer
	seq.recentTokens[seq.recentTokensIdx] = token
	seq.recentTokensIdx = (seq.recentTokensIdx + 1) % bufSize
	
	// Only check after we have enough tokens (at least 64)
	if seq.numPredicted < 64 {
		return false
	}
	
	// Check for repeating patterns of various lengths (8 to 64 tokens)
	for patternLen := 8; patternLen <= 64; patternLen++ {
		if seq.numPredicted < patternLen*3 {
			continue
		}
		
		// Check if the last patternLen tokens match the previous patternLen tokens
		matches := 0
		for rep := 1; rep <= 2; rep++ { // check 2 repetitions
			allMatch := true
			for i := 0; i < patternLen; i++ {
				// Calculate indices in circular buffer
				idx1 := (seq.recentTokensIdx - 1 - i + bufSize) % bufSize
				idx2 := (seq.recentTokensIdx - 1 - i - (patternLen * rep) + bufSize) % bufSize
				
				if seq.recentTokens[idx1] != seq.recentTokens[idx2] {
					allMatch = false
					break
				}
			}
			if allMatch {
				matches++
			}
		}
		
		// If we found 2 consecutive repetitions of the same pattern
		if matches >= 2 {
			seq.repetitionCount++
			if seq.repetitionCount >= 3 { // Trigger after 3 detections
				slog.Warn("repetition loop detected", "patternLen", patternLen, 
					"numPredicted", seq.numPredicted, "repetitionCount", seq.repetitionCount)
				return true
			}
		}
	}
	
	return false
}

// calculateLogprobsLlama converts raw logits to log probabilities and finds top K tokens
func calculateLogprobsLlama(logits []float32, selectedToken int, topK int, model *llama.Model) []llm.Logprob {
	return common.CalculateLogprobs(logits, selectedToken, topK, model.TokenToPiece)
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

			chunks, err := s.image.MultimodalTokenize(s.lc, images[imageIndex].Data)
			if err != nil {
				return nil, err
			}

			for _, c := range chunks {
				if len(c.Embed) != 0 {
					// Check if this is an M-RoPE image chunk (has grid dimensions)
					if c.Nx > 0 && c.Ny > 0 {
						// M-RoPE: add as single input with all embeddings and grid info
						inputs = append(inputs, input{
							embed:   c.Embed,
							imageNx: c.Nx,
							imageNy: c.Ny,
						})
					} else {
						// Non-M-RoPE: single embedding per token
						inputs = append(inputs, input{embed: c.Embed})
					}
				} else {
					for _, t := range c.Tokens {
						inputs = append(inputs, input{token: t})
					}
				}
			}
		}
	}

	return inputs, nil
}

type Server struct {
	// modelPath is the location of the model to be loaded
	modelPath string

	extraModelPaths []string

	// loadMu prevents more than one load attempt from occurring at a time
	loadMu sync.Mutex

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

// cleanup frees all resources held by the server.
// This method should be called when shutting down or when handling close operations.
func (s *Server) cleanup() {
	// Free image context resources
	if s.image != nil {
		s.image.Free(s.modelPath)
		s.image = nil
	}

	// Free input cache
	if s.cache != nil {
		s.cache = nil
	}

	// Free llama context
	if s.lc != nil {
		s.lc.Free()
		s.lc = nil
	}

	// Free model
	if s.model != nil {
		llama.FreeModel(s.model)
		s.model = nil
	}
}

func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	logprobs := seq.pendingLogprobs
	seq.pendingResponses = []string{}
	seq.pendingLogprobs = []llm.Logprob{}

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
	case seq.responses <- response{content: joined, logprobs: logprobs}:
		return true
	case <-seq.quit:
		return false
	}
}

func (s *Server) removeSequence(seqIndex int, reason llm.DoneReason) {
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
		// Use M-RoPE batch if the vision model requires it (Qwen3-VL, Qwen2-VL)
		if s.image.UsesMRoPE() {
			slog.Info("using M-RoPE batch for vision model")
			embedBatch, err = llama.NewBatchMRoPE(embedBatchSize, len(s.seqs), s.image.EmbedSize(s.lc))
		} else {
			embedBatch, err = llama.NewBatch(embedBatchSize, len(s.seqs), s.image.EmbedSize(s.lc))
		}
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
	var numOutputs int
	var mropeBatchReady bool // Set when M-RoPE image is added; must not add more to batch

	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]

		if seq == nil {
			continue
		}

		// If we've added an M-RoPE image, don't add anything else to the batch
		// because the position layout is fixed to the current n_tokens
		if mropeBatchReady {
			break
		}

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(seqIdx, llm.DoneReasonLength)
			continue
		}

		for i, input := range seq.inputs {
			// Calculate how many tokens this input represents
			inputTokenCount := input.numTokens()

			// Check if we have enough space in context
			// Count actual tokens, not just inputs (M-RoPE images have many tokens per input)
			cachedTokens := 0
			for _, ci := range seq.cache.Inputs {
				cachedTokens += ci.numTokens()
			}
			pendingTokens := 0
			for _, pi := range seq.pendingInputs {
				pendingTokens += pi.numTokens()
			}
			if cachedTokens+pendingTokens+inputTokenCount > s.cache.numCtx {
				if len(seq.pendingInputs) == 0 {
					if !seq.shift {
						s.removeSequence(seqIdx, llm.DoneReasonLength)
						break
					}

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

			// Check batch capacity (for M-RoPE images, we need space for all tokens)
			if batch.NumTokens()+inputTokenCount > batch.Size() {
				break
			}

			output := i+1 == len(seq.inputs)

			// Calculate position for this input
			pos := 0
			for _, pi := range seq.cache.Inputs {
				pos += pi.numPos()
			}
			for _, pi := range seq.pendingInputs {
				pos += pi.numPos()
			}

			// Add input to batch
			if input.isImageMRoPE() && batch.IsMRoPE() {
				// M-RoPE image: use special method that sets 2D positions
				batch.AddImageMRoPE(input.embed, pos, input.imageNx, input.imageNy, output, seq.cache.Id)

				// For M-RoPE, we must process the batch immediately after adding the image
				// because the position array layout depends on the total token count.
				// Adding more tokens would corrupt the M-RoPE position encoding.
				if output {
					numOutputs++
				}
				seq.pendingInputs = append(seq.pendingInputs, input)
				seq.iBatch = batch.NumTokens() - 1
				s.nextSeq = seqIdx
				mropeBatchReady = true // Signal to break outer loop too
				// Don't process more inputs - let the outer code handle seq.inputs update
				break
			}

			// Regular token or non-M-RoPE embedding
			batch.Add(input.token, input.embed, pos, output, seq.cache.Id)

			if output {
				numOutputs++
			}

			seq.pendingInputs = append(seq.pendingInputs, input)
			seq.iBatch = batch.NumTokens() - 1
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	if batch == nil || batch.NumTokens() == 0 {
		return nil
	}

	t := time.Now()
	if err := s.lc.Decode(batch); err != nil {
		return fmt.Errorf("failed to decode batch: %w", err)
	}

	if numOutputs > 0 {
		s.lc.Synchronize()
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
			seq.processingDuration += time.Since(t)
			continue
		}

		seq.numDecoded++
		if seq.numDecoded > 1 {
			seq.generationDuration += time.Since(t)
		} else {
			seq.processingDuration += time.Since(t)
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

		// Check for repetition loop - stop if model is stuck repeating
		if seq.detectRepetitionLoop(token) {
			slog.Warn("stopping generation due to repetition loop", "numPredicted", seq.numPredicted)
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		// Calculate logprobs if requested (after EOS check to avoid logprobs for EOS tokens)
		if seq.logprobs {
			logits := s.lc.GetLogitsIth(seq.iBatch)
			if logits != nil {
				logprobs := calculateLogprobsLlama(logits, token, seq.topLogprobs, s.model)
				seq.pendingLogprobs = append(seq.pendingLogprobs, logprobs...)
			}
		}

		seq.inputs = []input{{token: token}}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := common.FindStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = common.TruncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Truncate logprobs to match the truncated responses
			if seq.logprobs {
				origLogprobsLen := len(seq.pendingLogprobs)
				numTokensRemoved := origLen - newLen
				newLogprobsLen := origLogprobsLen - numTokensRemoved
				if newLogprobsLen < 0 {
					newLogprobsLen = 0
				}
				seq.pendingLogprobs = seq.pendingLogprobs[:newLogprobsLen]
			}

			// Update the cache based on the tokens that will be returned:
			// - We have 1 token more than is currently in the cache because
			// the last one generated wasn't submitted to Decode
			// - Remove any stop sequences that we stripped out
			// - If truncateStop removed a portion of a token, drop that
			// - As defense-in-depth, if truncatedToken didn't find a stop token
			// remove the extra one that we added to the cache len
			// NOTE: This assumes each recent input is a single token. This is true
			// during text generation (post-prompt), but M-RoPE images would have
			// multiple tokens per input. Since stop detection only happens during
			// text generation, this assumption is safe.
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

		if !flushPending(seq) {
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
		shift:          req.Shift,
		truncate:       req.Truncate,
		logprobs:       req.Logprobs,
		topLogprobs:    req.TopLogprobs,
	})
	if err != nil {
		if errors.Is(err, errorInputTooLong) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
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
		case resp, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Content:  resp.content,
					Logprobs: resp.logprobs,
				}); err != nil {
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
					PromptEvalDuration: seq.processingDuration,
					EvalCount:          seq.numDecoded,
					EvalDuration:       seq.generationDuration,
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

	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{
		embedding: true,

		// TODO (jmorganca): this should be provided by the server via the
		// request options and truncated here in the runner, instead of relying on
		// the server's truncate logic
		truncate: true,
	})
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

// loadModel allocates memory based on the given parameters and loads the weights. The
// memory allocated is worst case for text models but not for vision.
func (s *Server) loadModel(
	params llama.ModelParams,
	mpath string,
	empath []string,
	lpath []string,
	ppath string,
	kvSize int,
	kvCacheType string,
	flashAttention bool,
	threads int,
	multiUserCache bool,
) {
	var err error
	s.model, err = llama.LoadModelFromFile(mpath, empath, params)
	if err != nil {
		panic(err)
	}

	// For multimodal models with projector, use larger batch size to accommodate
	// M-RoPE images which can have 4000+ tokens that must be processed together.
	// The batch size must be set before creating the context.
	actualBatchSize := s.batchSize
	if ppath != "" {
		const multimodalMinBatchSize = 8192
		if actualBatchSize < multimodalMinBatchSize {
			slog.Info("increasing batch size for multimodal model",
				"original", actualBatchSize,
				"new", multimodalMinBatchSize,
				"reason", "M-RoPE images require processing all tokens in single batch")
			actualBatchSize = multimodalMinBatchSize
			s.batchSize = actualBatchSize // Update server's batchSize for BatchSize() method
		}
	}

	ctxParams := llama.NewContextParams(kvSize, actualBatchSize*s.parallel, s.parallel, threads, flashAttention, kvCacheType)
	s.lc, err = llama.NewContextWithModel(s.model, ctxParams)
	if err != nil {
		panic(err)
	}

	for _, path := range lpath {
		err := s.model.ApplyLoraFromFile(s.lc, path, 1.0, threads)
		if err != nil {
			panic(err)
		}
	}

	// Free previous image context if exists (cleanup from previous load)
	if s.image != nil {
		s.image.Free(s.modelPath)
		s.image = nil
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

// load is the handler called by the Ollama server to process different
// load operations
func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	if s.status != llm.ServerStatusLaunched {
		http.Error(w, "model already loaded", http.StatusInternalServerError)
		return
	}

	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	slog.Info("load", "request", req)

	switch req.Operation {
	// LoadOperationFit and LoadOperationAlloc have no meaning here - just return a successful response

	case llm.LoadOperationCommit:
		s.batchSize = req.BatchSize
		s.parallel = req.Parallel
		s.seqs = make([]*Sequence, s.parallel)
		s.seqsSem = semaphore.NewWeighted(int64(s.parallel))

		numGPU := 0
		var tensorSplit []float32
		var llamaIDs []uint64

		gpuIDs := llama.EnumerateGPUs()
		sort.Sort(req.GPULayers)
		for _, layers := range req.GPULayers {
			for i := range gpuIDs {
				if gpuIDs[i].DeviceID == layers.DeviceID {
					numGPU += len(layers.Layers)
					tensorSplit = append(tensorSplit, float32(len(layers.Layers)))
					llamaIDs = append(llamaIDs, gpuIDs[i].LlamaID)
				}
			}
		}

		params := llama.ModelParams{
			Devices:      llamaIDs,
			NumGpuLayers: numGPU,
			MainGpu:      req.MainGPU,
			UseMmap:      req.UseMmap && len(req.LoraPath) == 0,
			TensorSplit:  tensorSplit,
			Progress: func(progress float32) {
				s.progress = progress
			},
		}

		s.status = llm.ServerStatusLoadingModel
		go s.loadModel(params, s.modelPath, s.extraModelPaths, req.LoraPath, req.ProjectorPath, req.KvSize, req.KvCacheType, req.FlashAttention, req.NumThreads, req.MultiUserCache)

	case llm.LoadOperationClose:
		s.cleanup()

		if err := json.NewEncoder(w).Encode(&llm.LoadResponse{}); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		}
		return
	}

	resp := llm.LoadResponse{Success: true}
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

func Execute(args []string) error {
	fs := pflag.NewFlagSet("runner", pflag.ExitOnError)
	mpaths := fs.StringArray("model", []string{""}, "Path to model binary file. May be repeatedly specified to provide other splits of model binaries.")
	port := fs.Int("port", 8080, "Port to expose the server on")
	_ = fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		// sadly pflag does not expose out(). Fallback to os.Stderr which should perform identically as we don't set fs.output
		fmt.Fprintf(os.Stderr, "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting go runner")

	llama.BackendInit()

	server := &Server{
		modelPath:       (*mpaths)[0],
		extraModelPaths: (*mpaths)[1:],
		status:          llm.ServerStatusLaunched,
	}

	server.ready.Add(1)

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handler to free resources on SIGINT/SIGTERM
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		server.mu.Lock()
		server.cleanup()
		server.mu.Unlock()
		cancel()
		os.Exit(0)
	}()

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", server.load)
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
