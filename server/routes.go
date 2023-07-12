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

	ch := make(chan any)
	go stream(c, ch)

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

	llm, err := llama.New(req.Model, req.Options)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer llm.Close()

	fn := func(s string) {
		ch <- api.GenerateResponse{Response: s}
	}

	if err := llm.Predict(req.Prompt, fn); err != nil {
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
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go stream(c, ch)

	fn := func(total, completed int64) {
		ch <- api.PullProgress{
			Total:     total,
			Completed: completed,
			Percent:   float64(completed) / float64(total) * 100,
		}
	}

	if err := saveModel(remote, fn); err != nil {
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
