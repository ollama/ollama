package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	buildkite "github.com/buildkite/go-buildkite/v4"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	apiToken = kingpin.Flag("token", "API token (or set BUILDKITE_TOKEN)").String()
	org      = kingpin.Flag("org", "Organization slug").Required().String()
	addr     = kingpin.Flag("addr", "HTTP listen address").Default(":8080").String()
)

func safeStr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

type ChatRequest struct {
	Message string `json:"message"`
}

type ChatResponse struct {
	Reply string `json:"reply"`
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	// Very simple example: echo back the message. Replace this with your model/LLM integration.
	resp := ChatResponse{Reply: "Echo: " + req.Message}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("failed to write response: %v", err)
	}
}

func main() {
	kingpin.Parse()

	// allow token from env as a fallback
	token := ""
	if apiToken != nil && *apiToken != "" {
		token = *apiToken
	} else if v, ok := os.LookupEnv("BUILDKITE_TOKEN"); ok {
		token = v
	} else {
		log.Fatalf("API token required: provide --token or set BUILDKITE_TOKEN")
	}

	client, err := buildkite.NewOpts(
		buildkite.WithTokenAuth(token),
	)
	if err != nil {
		log.Fatalf("client config failed: %s", err)
	}

	// Note: List is paginated. If you have many pipelines you should iterate pages.
	// This example requests the first page with default page size.
	pipelines, _, err := client.Pipelines.List(
		*org,
		&buildkite.PipelineListOptions{},
	)
	if err != nil {
		log.Fatalf("failed to list pipelines: %s", err)
	}

	for _, p := range pipelines {
		log.Printf("Pipeline: %s (%s)", safeStr(p.Name), safeStr(p.Slug))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", chatHandler)

	srv := &http.Server{
		Addr:    *addr,
		Handler: mux,
	}

	// start server
	go func() {
		log.Printf("starting server on %s", *addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen failed: %v", err)
		}
	}()

	// graceful shutdown on SIGINT/SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Printf("shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}