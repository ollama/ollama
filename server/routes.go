package server

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"runtime"

	"github.com/gin-gonic/gin"
	llama "github.com/jmorganca/ollama/llama"

	"github.com/jmorganca/ollama/api"
)

func generate(c *gin.Context) {
	// TODO: these should be request parameters
	gpulayers := 1
	tokens := 512
	threads := runtime.NumCPU()
	// TODO: set prompt from template

	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	l, err := llama.New(req.Model, llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(gpulayers))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		return
	}

	ch := make(chan string)

	go func() {
		defer close(ch)
		_, err := l.Predict(req.Prompt, llama.Debug, llama.SetTokenCallback(func(token string) bool {
			ch <- token
			return true
		}), llama.SetTokens(tokens), llama.SetThreads(threads), llama.SetTopK(90), llama.SetTopP(0.86), llama.SetStopWords("llama"))
		if err != nil {
			panic(err)
		}
	}()

	c.Stream(func(w io.Writer) bool {
		tok, ok := <-ch
		if !ok {
			return false
		}
		c.SSEvent("token", tok)
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

		progressCh := make(chan string)
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
			c.SSEvent("progress", progress)
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
