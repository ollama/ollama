package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"sync"

	"github.com/ollama/ollama/llama"
)

type Request struct {
	Prompt string `json:"prompt"`
}

type Response struct {
	Token string `json:"token"`
}

type Server struct {
	model *llama.Model
	lc    *llama.Context
	batch *llama.Batch

	queue chan Sequence
	seqs  []*Sequence

	// mu guards seqs
	mu sync.Mutex
}

type Sequence struct {
	prompt []llama.Token
	out    chan string
}

func schedule(parallel int, queue <-chan Sequence) {
	// Fill sequences from the queue

	// once a sequence finishes, remove it from and add a new one from the queue
}

func process() {
	// loop through the sequences, fill a batch, decode and sample tokens, responding to appropriate requests
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	var request Request
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	tokens, err := s.model.Tokenize(request.Prompt, 2048, true, true)
	if err != nil {
		panic(err)
	}

	seq := Sequence{prompt: tokens}
	s.queue <- seq

	// listen for the sequence to finish
	for {
		str := <-seq.out
		if err := json.NewEncoder(w).Encode(&Response{Token: str}); err != nil {
			log.Println("Failed to encode result:", err)
			return
		}
		w.(http.Flusher).Flush()
	}

	// prompt eval
	for i, t := range tokens {
		batch.Add(t, llama.Pos(i), []llama.SeqId{0}, true)
	}

	// main loop
	for n := batch.NumTokens(); n < 2048; n++ {
		mu.Lock()
		err = s.lc.Decode(batch)
		if err != nil {
			panic("Failed to decode")
		}

		// sample a token
		token := s.lc.SampleTokenGreedy(batch)
		mu.Unlock()

		// if it's an end of sequence token, break
		if s.model.TokenIsEog(token) {
			break
		}

		// print the token
		str := s.model.TokenToPiece(token)

		if err := enc.Encode(&Response{Token: str}); err != nil {
			log.Println("Failed to encode result:", err)
			return
		}
		w.(http.Flusher).Flush()

		batch.Clear()
		batch.Add(token, llama.Pos(n), []llama.SeqId{0}, true)
	}
}

func main() {
	mp := flag.String("model", "", "Path to model binary file")
	parallel := flag.Int("parallel", 1, "Number of parallel requests to handle")
	flag.Parse()

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams()
	model := llama.LoadModelFromFile(*mp, params)
	ctxParams := llama.NewContextParams()
	lc := llama.NewContextWithModel(model, ctxParams)
	if lc == nil {
		panic("Failed to create context")
	}

	server := &Server{
		model: model,
		lc:    lc,
		queue: make(chan Sequence, 256),
		seqs:  make([]*Sequence, *parallel),
	}

	addr := "127.0.0.1:8080"
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return
	}
	defer listener.Close()

	httpServer := http.Server{
		Handler: http.HandlerFunc(server.stream),
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}
}
