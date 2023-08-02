package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
)

var loaded struct {
	mu sync.Mutex

	llm *llama.LLM

	expireAt    time.Time
	expireTimer *time.Timer

	digest  string
	options api.Options
}

func GenerateHandler(c *gin.Context) {
	loaded.mu.Lock()
	defer loaded.mu.Unlock()

	checkpointStart := time.Now()

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

	if model.Digest != loaded.digest || !reflect.DeepEqual(loaded.options, req.Options) {
		if loaded.llm != nil {
			loaded.llm.Close()
			loaded.llm = nil
			loaded.digest = ""
		}

		opts := api.DefaultOptions()
		if err := opts.FromMap(model.Options); err != nil {
			log.Printf("could not load model options: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if err := opts.FromMap(req.Options); err != nil {
			log.Printf("could not merge model options: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		llm, err := llama.New(model.ModelPath, opts)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		loaded.llm = llm
		loaded.digest = model.Digest
	}

	sessionDuration := 5 * time.Minute

	loaded.expireAt = time.Now().Add(sessionDuration)
	if loaded.expireTimer == nil {
		loaded.expireTimer = time.AfterFunc(sessionDuration, func() {
			loaded.mu.Lock()
			defer loaded.mu.Unlock()

			if time.Now().Before(loaded.expireAt) {
				return
			}

			if loaded.llm == nil {
				return
			}

			loaded.llm.Close()
			loaded.llm = nil
			loaded.digest = ""
		})
	}
	loaded.expireTimer.Reset(sessionDuration)

	checkpointLoaded := time.Now()

	prompt, err := model.Prompt(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.GenerateResponse) {
			loaded.expireAt = time.Now().Add(sessionDuration)
			loaded.expireTimer.Reset(sessionDuration)

			r.Model = req.Model
			r.CreatedAt = time.Now().UTC()
			if r.Done {
				r.TotalDuration = time.Since(checkpointStart)
				r.LoadDuration = checkpointLoaded.Sub(checkpointStart)
			}

			ch <- r
		}

		if err := loaded.llm.Predict(req.Context, prompt, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	streamResponse(c, ch)
}

func PullModelHandler(c *gin.Context) {
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

		regOpts := &RegistryOptions{
			Insecure: req.Insecure,
			Username: req.Username,
			Password: req.Password,
		}

		if err := PullModel(req.Name, regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	streamResponse(c, ch)
}

func PushModelHandler(c *gin.Context) {
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

		regOpts := &RegistryOptions{
			Insecure: req.Insecure,
			Username: req.Username,
			Password: req.Password,
		}

		if err := PushModel(req.Name, regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	streamResponse(c, ch)
}

func CreateModelHandler(c *gin.Context) {
	var req api.CreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(resp api.ProgressResponse) {
			ch <- resp
		}

		if err := CreateModel(req.Name, req.Path, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	streamResponse(c, ch)
}

func DeleteModelHandler(c *gin.Context) {
	var req api.DeleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := DeleteModel(req.Name); err != nil {
		if os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Name)})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
}

func ListModelsHandler(c *gin.Context) {
	var models []api.ListResponseModel
	fp, err := GetManifestPath()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	err = filepath.Walk(fp, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				log.Printf("manifest file does not exist: %s", fp)
				return nil
			}
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

	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func CopyModelHandler(c *gin.Context) {
	var req api.CopyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := CopyModel(req.Source, req.Destination); err != nil {
		if os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Source)})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
}

func Serve(ln net.Listener) error {
	config := cors.DefaultConfig()
	config.AllowWildcard = true
	// only allow http/https from localhost
	config.AllowOrigins = []string{
		"http://localhost",
		"http://localhost:*",
		"https://localhost",
		"https://localhost:*",
		"http://127.0.0.1",
		"http://127.0.0.1:*",
		"https://127.0.0.1",
		"https://127.0.0.1:*",
	}

	r := gin.Default()
	r.Use(cors.New(config))

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})
	r.HEAD("/", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/pull", PullModelHandler)
	r.POST("/api/generate", GenerateHandler)
	r.POST("/api/create", CreateModelHandler)
	r.POST("/api/push", PushModelHandler)
	r.POST("/api/copy", CopyModelHandler)
	r.GET("/api/tags", ListModelsHandler)
	r.DELETE("/api/delete", DeleteModelHandler)

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
			log.Printf("streamResponse: json.Marshal failed with %s", err)
			return false
		}

		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			log.Printf("streamResponse: w.Write failed with %s", err)
			return false
		}

		return true
	})
}
