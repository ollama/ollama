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
}

var mu sync.Mutex

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

	enc := json.NewEncoder(w)

	// main loop
	tokens, err := s.model.Tokenize(request.Prompt, 2048, true, true)
	if err != nil {
		panic(err)
	}

	batch := llama.NewBatch(512, 0, 1)

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
