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

	ch := make(chan string)
	g, _ := errgroup.WithContext(c.Request.Context())
	g.Go(func() error {
		defer close(ch)
		return llm.Predict(req.Prompt, func(s string) {
			ch <- s
		})
	})

	g.Go(func() error {
		c.Stream(func(w io.Writer) bool {
			s, ok := <-ch
			if !ok {
				return false
			}

			bts, err := json.Marshal(api.GenerateResponse{Response: s})
			if err != nil {
				return false
			}

			bts = append(bts, '\n')
			if _, err := w.Write(bts); err != nil {
				return false
			}

			return true
		})

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

	r.POST("api/pull", func(c *gin.Context) {
		var req api.PullRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		progressCh := make(chan api.PullProgress)
		go func() {
			defer close(progressCh)
			if err := pull(req.Model, progressCh); err != nil {
				var opError *net.OpError
				if errors.As(err, &opError) {
					c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
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
