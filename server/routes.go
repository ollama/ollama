package server

import (
	"embed"
	"encoding/json"
	"errors"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"path"
	"strings"
	"text/template"

	"github.com/gin-gonic/gin"
	"github.com/lithammer/fuzzysearch/fuzzy"
	"golang.org/x/sync/errgroup"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
)

//go:embed templates/*
var templatesFS embed.FS
var templates = template.Must(template.ParseFS(templatesFS, "templates/*.prompt"))

func cacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama")
}

func generate(c *gin.Context) {
	req := api.GenerateRequest{
		Options: api.DefaultOptions(),
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if remoteModel, _ := getRemote(req.Model); remoteModel != nil {
		req.Model = remoteModel.FullName()
	}
	if _, err := os.Stat(req.Model); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		req.Model = path.Join(cacheDir(), "models", req.Model+".bin")
	}

	llm, err := llama.New(req.Model, req.Options)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer llm.Close()

	templateNames := make([]string, 0, len(templates.Templates()))
	for _, template := range templates.Templates() {
		templateNames = append(templateNames, template.Name())
	}

	match, _ := matchRankOne(path.Base(req.Model), templateNames)
	if template := templates.Lookup(match); template != nil {
		var sb strings.Builder
		if err := template.Execute(&sb, req); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		req.Prompt = sb.String()
	}

	ch := make(chan any)
	g, _ := errgroup.WithContext(c.Request.Context())
	g.Go(func() error {
		defer close(ch)
		return llm.Predict(req.Prompt, func(s string) {
			ch <- api.GenerateResponse{Response: s}
		})
	})

	g.Go(func() error {
		stream(c, ch)
		return nil
	})

	if err := g.Wait(); err != nil && !errors.Is(err, io.EOF) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func pull(c *gin.Context) {
	var req api.PullRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	remote, err := getRemote(req.Model)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	g, _ := errgroup.WithContext(c.Request.Context())
	g.Go(func() error {
		defer close(ch)
		return saveModel(remote, func(total, completed int64) {
			ch <- api.PullProgress{
				Total:     total,
				Completed: completed,
				Percent:   float64(total) / float64(completed) * 100,
			}
		})
	})

	g.Go(func() error {
		stream(c, ch)
		return nil
	})

	if err := g.Wait(); err != nil && !errors.Is(err, io.EOF) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func Serve(ln net.Listener) error {
	r := gin.Default()

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})

	r.POST("api/pull", pull)
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

func stream(c *gin.Context, ch chan any) {
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
