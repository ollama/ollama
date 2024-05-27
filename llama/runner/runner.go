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

	responses chan string

	samplingCtx *llama.SamplingContext
}

// prompt returns true if the prompt is still being processed
func (s *Sequence) prompt() bool {
	return s.nPast < len(s.tokens)-1
}

func (s *Server) NewSequence(r Request, w http.ResponseWriter) *Sequence {
	var samplingParams llama.SamplingParams
	samplingParams.TopK = r.TopK
	samplingParams.TopP = r.TopP
	samplingParams.TfsZ = r.TFSZ
	samplingParams.TypicalP = r.TypicalP
	samplingParams.Temp = r.Temperature
	samplingParams.PenaltyRepeat = r.RepeatPenalty
	samplingParams.PenaltyFreq = r.FrequencyPenalty
	samplingParams.PenaltyPresent = r.PresencePenalty
	samplingParams.Mirostat = r.Mirostat
	samplingParams.MirostatTau = r.MirostatTau
	samplingParams.MirostatEta = r.MirostatEta
	samplingParams.PenalizeNl = r.PenalizeNewline
	samplingParams.Seed = uint32(r.Seed)
	samplingParams.Grammar = r.Grammar

	tokens, err := s.lc.Model().Tokenize(r.Prompt, 2048, false, true)
	if err != nil {
		panic(err)
	}

	sc := llama.NewSamplingContext(samplingParams)

	for _, t := range tokens {
		sc.Accept(s.lc, t, false)
	}

	return &Sequence{
		tokens:      tokens,
		responses:   make(chan string, 1),
		samplingCtx: sc,
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

type Request struct {
	Prompt  string   `json:"prompt"`
	Images  []string `json:"images"`
	Grammar string   `json:"grammar"`

	api.Options
}

type Response struct {
	Token string `json:"token"`
}

func (s *Server) handler(w http.ResponseWriter, r *http.Request) {
	var request Request
	request.Options = api.DefaultOptions()
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	seq := s.NewSequence(request, w)

	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			s.seqs[i] = seq
			fmt.Println("signal")
			s.cond.Signal()
			break
		}
	}
	s.mu.Unlock()

	for token := range seq.responses {
		if err := json.NewEncoder(w).Encode(&Response{
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

	httpServer := http.Server{
		Handler: http.HandlerFunc(server.handler),
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}

	cancel()
}
