package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
)

func cacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama")
}

func generate(c *gin.Context) {
	start := time.Now()

	req := api.GenerateRequest{
		Options: api.DefaultOptions(),
		Prompt:  "",
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	model, err := GetModel(req.Model)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	templ, err := template.New("").Parse(model.Prompt)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var sb strings.Builder
	if err = templ.Execute(&sb, req); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Prompt = sb.String()

	fmt.Printf("prompt = >>>%s<<<\n", req.Prompt)

	llm, err := llama.New(model.ModelPath, req.Options)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer llm.Close()

	ch := make(chan any)
	go func() {
		defer close(ch)
		llm.Predict(req.Context, req.Prompt, func(r api.GenerateResponse) {
			r.Model = req.Model
			r.CreatedAt = time.Now().UTC()
			if r.Done {
				r.TotalDuration = time.Since(start)
			}

			ch <- r
		})
	}()

	streamResponse(c, ch)
}

func pull(c *gin.Context) {
	var req api.PullRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(status, digest string, total, completed int, percent float64) {
			ch <- api.PullProgress{
				Status:    status,
				Digest:    digest,
				Total:     total,
				Completed: completed,
				Percent:   percent,
			}
		}
		if err := PullModel(req.Name, req.Username, req.Password, fn); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
	}()

	streamResponse(c, ch)
}

func push(c *gin.Context) {
	var req api.PushRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(status, digest string, total, completed int, percent float64) {
			ch <- api.PushProgress{
				Status:    status,
				Digest:    digest,
				Total:     total,
				Completed: completed,
				Percent:   percent,
			}
		}
		if err := PushModel(req.Name, req.Username, req.Password, fn); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
	}()

	streamResponse(c, ch)
}

func create(c *gin.Context) {
	var req api.CreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	// NOTE consider passing the entire Modelfile in the json instead of the path to it

	file, err := os.Open(req.Path)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}
	defer file.Close()

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(status string) {
			ch <- api.CreateProgress{
				Status: status,
			}
		}

		if err := CreateModel(req.Name, file, fn); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
			return
		}
	}()

	streamResponse(c, ch)
}

func Serve(ln net.Listener) error {
	r := gin.Default()

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})

	r.POST("/api/pull", pull)
	r.POST("/api/generate", generate)
	r.POST("/api/create", create)
	r.POST("/api/push", push)

	log.Printf("Listening on %s", ln.Addr())
	s := &http.Server{
		Handler: r,
	}

	return s.Serve(ln)
}

func streamResponse(c *gin.Context, ch chan any) {
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		bts, err := json.Marshal(val)
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
