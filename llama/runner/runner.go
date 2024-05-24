package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"sync"

	"github.com/ollama/ollama/llama"
)

type Request struct {
	Prompt string   `json:"prompt"`
	Images []string `json:"images"`
}

type Response struct {
	Token string `json:"token"`
}

type Server struct {
	model *llama.Model
	lc    *llama.Context
	cc    *llama.ClipContext
}

var mu sync.Mutex

func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	var request Request
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	mu.Lock()
	defer mu.Unlock()

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)

	// create embeddings for each image
	var embeddings []*llama.LlavaImageEmbed
	if s.cc != nil {
		for _, img := range request.Images {
			data, err := base64.StdEncoding.DecodeString(img)
			if err != nil {
				http.Error(w, "Failed to decode image", http.StatusBadRequest)
				return
			}

			embd := llama.NewLlavaImageEmbed(s.cc, data)
			embeddings = append(embeddings, embd)
		}
	}

	var nPast int

	// eval the prompt
	re := regexp.MustCompile(`\[\s*img-(\d+)\s*\]`)
	matches := re.FindAllStringSubmatchIndex(request.Prompt, -1)

	// eval each chunk including images
	pos := 0
	for _, match := range matches {
		part := request.Prompt[pos:match[0]]
		fmt.Println("Text part:", part)

		// eval text before image
		err := s.evalText(part, &nPast)
		if err != nil {
			log.Println("Failed to eval text:", err)
			return
		}

		// eval image
		imgIndexStr := request.Prompt[match[2]:match[3]]
		imgIndex, err := strconv.Atoi(imgIndexStr)
		if err != nil {
			slog.Warn("Failed to parse image index", "index", imgIndexStr)
			continue
		}

		fmt.Println("Tag index:", imgIndex)
		if imgIndex <= len(embeddings) {
			slog.Info("evaluating image", "index", imgIndex)
			llama.LlavaEvalImageEmbed(s.lc, embeddings[imgIndex], 512, &nPast)
		}

		pos = match[1]
	}

	// eval remaining text
	if pos < len(request.Prompt) {
		s.evalText(request.Prompt[pos:], &nPast)
	}

	batch := llama.NewBatch(512, 0, 1)
	defer batch.Free()

	// main loop
	for n := nPast; n < 2048; n++ {
		// sample a token
		token := s.lc.SampleTokenGreedy(batch)

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
		batch.Add(token, n, []int{0}, true)

		err := s.lc.Decode(batch)
		if err != nil {
			panic("Failed to decode")
		}
	}

	s.lc.KvCacheClear()
}

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("projector", "", "Path to projector binary file")
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
	if ppath != nil {
		cc = llama.NewClipContext(*ppath)
		if cc == nil {
			panic("Failed to create clip context")
		}
	}

	server := &Server{
		model: model,
		lc:    lc,
		cc:    cc,
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

func (s *Server) evalText(text string, nPast *int) error {
	// eval before
	batch := llama.NewBatch(512, 0, 1)
	defer batch.Free()

	tokens, err := s.lc.Model().Tokenize(text, 2048, true, true)
	if err != nil {
		return fmt.Errorf("tokenize failed: %w", err)
	}

	// prompt eval
	for _, t := range tokens {
		batch.Add(t, *nPast, []int{0}, true)
		*nPast++
	}

	err = s.lc.Decode(batch)
	if err != nil {
		return fmt.Errorf("decode failed: %w", err)
	}

	return nil
}
