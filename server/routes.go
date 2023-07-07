package server

import (
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"path"
	"runtime"
	"strings"
	"text/template"

	"github.com/gin-gonic/gin"
	"github.com/lithammer/fuzzysearch/fuzzy"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
)

//go:embed templates/*
var templatesFS embed.FS
var templates = template.Must(template.ParseFS(templatesFS, "templates/*.prompt"))

func generate(c *gin.Context) {
	// TODO: these should be request parameters
	gpulayers := 1
	tokens := 512
	threads := runtime.NumCPU()

	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	model, err := llama.New(req.Model, llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(gpulayers))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		return
	}
	defer model.Free()

	templateNames := make([]string, 0, len(templates.Templates()))
	for _, template := range templates.Templates() {
		templateNames = append(templateNames, template.Name())
	}

	match, _ := matchRankOne(path.Base(req.Model), templateNames)
	if template := templates.Lookup(match); template != nil {
		var sb strings.Builder
		if err := template.Execute(&sb, req); err != nil {
			fmt.Println("Prompt template failed:", err.Error())
			return
		}

		req.Prompt = sb.String()
	}

	ch := make(chan string)

	go func() {
		defer close(ch)
		_, err := model.Predict(req.Prompt, llama.Debug, llama.SetTokenCallback(func(token string) bool {
			ch <- token
			return true
		}), llama.SetTokens(tokens), llama.SetThreads(threads), llama.SetTopK(90), llama.SetTopP(0.86), llama.SetStopWords("llama"))
		if err != nil {
			panic(err)
		}
	}()

	c.Stream(func(w io.Writer) bool {
		token, ok := <-ch
		if !ok {
			return false
		}

		resp := api.GenerateResponse{
			Response: token,
		}

		bts, err := json.Marshal(resp)
		if err != nil {
			return false
		}

		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			return false
		}

		return true
	})
}

func Serve(ln net.Listener) error {
	r := gin.Default()

	r.POST("api/pull", func(c *gin.Context) {
		var req api.PullRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
			return
		}

		progressCh := make(chan api.PullProgress)
		go func() {
			defer close(progressCh)
			if err := pull(req.Model, progressCh); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
				return
			}
		}()

		c.Stream(func(w io.Writer) bool {
			progress, ok := <-progressCh
			if !ok {
				return false
			}

			bts, err := json.Marshal(progress)
			if err != nil {
				return false
			}

			bts = append(bts, '\n')
			if _, err := w.Write(bts); err != nil {
				return false
			}

			return true
		})
	})

	r.POST("/api/generate", generate)

	log.Printf("Listening on %s", ln.Addr())
	s := &http.Server{
		Handler: r,
	}

	return s.Serve(ln)
}

func matchRankOne(source string, targets []string) (bestMatch string, bestRank int) {
	bestRank = math.MaxInt
	for _, target := range targets {
		if rank := fuzzy.LevenshteinDistance(source, target); bestRank > rank {
			bestRank = rank
			bestMatch = target
		}
	}

	return
}
