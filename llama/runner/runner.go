package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llama"
)

type Sequence struct {
	// number of tokens evaluated
	nPast int

	// tokens left to evaluate
	tokens []int

	// channel to send responses over
	responses chan string

	samplingCtx *llama.SamplingContext

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool
}

// prompt returns true if the prompt is still being processed
func (s *Sequence) prompt() bool {
	return s.nPast < len(s.tokens)-1
}

func (s *Server) NewSequence(prompt string, params *llama.SamplingParams, embedding bool) *Sequence {
	tokens, err := s.lc.Model().Tokenize(prompt, 2048, false, true)
	if err != nil {
		panic(err)
	}

	var sc *llama.SamplingContext
	if params != nil {
		sc = llama.NewSamplingContext(*params)
		for _, t := range tokens {
			sc.Accept(s.lc, t, false)
		}
	}

	return &Sequence{
		tokens:        tokens,
		responses:     make(chan string, 1),
		embedding:     make(chan []float32, 1),
		samplingCtx:   sc,
		embeddingOnly: embedding,
	}
}

type Server struct {
	model *llama.Model
	lc    *llama.Context
	cc    *llama.ClipContext

	// parallel is the number of parallel requests to handle
	parallel int

	// seqs is the list of parallel sequences being evaluated
	seqs []*Sequence

	mu sync.Mutex

	cond *sync.Cond
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
	batch := llama.NewBatch(512, 0, s.parallel)
	defer batch.Free()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			slog.Info("Processing batch", "seqs", len(s.seqs))
			s.mu.Lock()
			for s.allNil() {
				s.cond.Wait() // Wait until an item is added
			}
			s.mu.Unlock()

			// prepare the batch
			ibatch := make([]int, s.parallel)
			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				for j, t := range seq.tokens {
					// todo: make this n_batch
					if j > 512 {
						break
					}

					batch.Add(t, seq.nPast, []int{i}, !seq.prompt())
					seq.nPast++

					if seq.prompt() {
						ibatch[i] = batch.NumTokens() + 1
					}
				}
			}

			err := s.lc.Decode(batch)
			if err != nil {
				panic("Failed to decode")
			}

			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				// don't sample prompt processing
				if seq.prompt() {
					if len(seq.tokens) < 512 {
						seq.tokens = []int{}
					} else {
						seq.tokens = seq.tokens[512:]
					}

					continue
				}

				// if done processing the prompt, generating an embedding and return
				if seq.embeddingOnly {
					embd := s.lc.GetEmbeddingsSeq(i)
					if embd == nil {
						embd = s.lc.GetEmbeddingsIth(ibatch[i])
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
				token := seq.samplingCtx.Sample(s.lc, nil, ibatch[i])
				seq.samplingCtx.Accept(s.lc, token, true)

				seq.responses <- s.model.TokenToPiece(token)
				seq.tokens = []int{token}

				// if it's an end of sequence token, break
				// TODO: just end this sequence
				if s.model.TokenIsEog(token) {
					// TODO: end the sequence instead of quitting the pool
					s.lc.KvCacheSeqRm(i, 0, -1)
					close(seq.responses)
					seq.samplingCtx.Free()
					s.seqs[i] = nil
					continue
				}
			}

			batch.Clear()
		}
	}
}

type CompletionRequest struct {
	Prompt  string   `json:"prompt"`
	Images  []string `json:"images"`
	Grammar string   `json:"grammar"`

	api.Options
}

type CompletionResponse struct {
	Token string `json:"token"`
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

	seq := s.NewSequence(req.Prompt, &samplingParams, false)

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
	for token := range seq.responses {
		if err := json.NewEncoder(w).Encode(&CompletionResponse{
			Token: token,
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
}

type EmbeddingRequest struct {
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// TODO (jmorganca): is it safe to do this concurrently with decoding?
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	seq := s.NewSequence(req.Prompt, nil, true)

	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
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

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("projector", "", "Path to projector binary file")
	parallel := flag.Int("parallel", 1, "Number of sequences to handle simultaneously")
	port := flag.Int("port", 8080, "Port to expose the server on")
	flag.Parse()

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams()
	model := llama.LoadModelFromFile(*mpath, params)
	ctxParams := llama.NewContextParams()
	lc := llama.NewContextWithModel(model, ctxParams)
	if lc == nil {
		panic("Failed to create context")
	}

	var cc *llama.ClipContext
	if *ppath != "" {
		cc = llama.NewClipContext(*ppath)
		if cc == nil {
			panic("Failed to create clip context")
		}
	}

	server := &Server{
		model:    model,
		lc:       lc,
		cc:       cc,
		parallel: *parallel,
		seqs:     make([]*Sequence, *parallel),
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
	mux.HandleFunc("/embeddings", server.embeddings)
	mux.HandleFunc("/completion", server.completion)

	httpServer := http.Server{
		Handler: mux,
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}

	cancel()
}
