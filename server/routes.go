package server

import (
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"dario.cat/mergo"
	"github.com/gin-gonic/gin"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
)

func generate(c *gin.Context) {
	start := time.Now()

	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	model, err := GetModel(req.Model)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	opts := api.DefaultOptions()
	if err := mergo.Merge(&opts, model.Options, mergo.WithOverride); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if err := mergo.Merge(&opts, req.Options, mergo.WithOverride); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	prompt, err := model.Prompt(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	llm, err := llama.New(model.ModelPath, opts)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer llm.Close()

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.GenerateResponse) {
			r.Model = req.Model
			r.CreatedAt = time.Now().UTC()
			if r.Done {
				r.TotalDuration = time.Since(start)
			}

			ch <- r
		}

		if err := llm.Predict(req.Context, prompt, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
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
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		if err := PullModel(req.Name, req.Username, req.Password, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
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
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		if err := PushModel(req.Name, req.Username, req.Password, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
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

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(status string) {
			ch <- api.CreateProgress{
				Status: status,
			}
		}

		if err := CreateModel(req.Name, req.Path, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	streamResponse(c, ch)
}

func list(c *gin.Context) {
	var models []api.ListResponseModel
	fp, err := GetManifestPath()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	err = filepath.Walk(fp, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			fi, err := os.Stat(path)
			if err != nil {
				log.Printf("skipping file: %s", fp)
				return nil
			}
			path := path[len(fp)+1:]
			slashIndex := strings.LastIndex(path, "/")
			if slashIndex == -1 {
				return nil
			}
			tag := path[:slashIndex] + ":" + path[slashIndex+1:]
			mp := ParseModelPath(tag)
			manifest, err := GetManifest(mp)
			if err != nil {
				log.Printf("skipping file: %s", fp)
				return nil
			}
			model := api.ListResponseModel{
				Name:       mp.GetShortTagname(),
				Size:       manifest.GetTotalSize(),
				ModifiedAt: fi.ModTime(),
			}
			models = append(models, model)
		}
		return nil
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, api.ListResponse{models})
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
	r.GET("/api/tags", list)

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
