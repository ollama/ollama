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
	var req api.GenerateRequest
	req.ModelOptions = api.DefaultModelOptions
	req.PredictOptions = api.DefaultPredictOptions
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	if remoteModel, _ := getRemote(req.Model); remoteModel != nil {
		req.Model = remoteModel.FullName()
	}

	modelOpts := getModelOpts(req)
	modelOpts.NGPULayers = 1  // hard-code this for now

	model, err := llama.New(req.Model, modelOpts)
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
	model.SetTokenCallback(func(token string) bool {
		ch <- token
		return true
	})

	predictOpts := getPredictOpts(req)

	go func() {
		defer close(ch)
		_, err := model.Predict(req.Prompt, predictOpts)
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

func getModelOpts(req api.GenerateRequest) llama.ModelOptions {
	var opts llama.ModelOptions
	opts.ContextSize = req.ModelOptions.ContextSize
	opts.Seed = req.ModelOptions.Seed
	opts.F16Memory = req.ModelOptions.F16Memory
	opts.MLock = req.ModelOptions.MLock
	opts.Embeddings = req.ModelOptions.Embeddings
	opts.MMap = req.ModelOptions.MMap
	opts.LowVRAM = req.ModelOptions.LowVRAM

	opts.NBatch = req.ModelOptions.NBatch
	opts.VocabOnly = req.ModelOptions.VocabOnly
	opts.NUMA = req.ModelOptions.NUMA
	opts.NGPULayers = req.ModelOptions.NGPULayers
	opts.MainGPU = req.ModelOptions.MainGPU
	opts.TensorSplit = req.ModelOptions.TensorSplit

	return opts
}

func getPredictOpts(req api.GenerateRequest) llama.PredictOptions {
	var opts llama.PredictOptions

	if req.PredictOptions.Threads == -1 {
		opts.Threads = runtime.NumCPU()
	} else {
		opts.Threads = req.PredictOptions.Threads
	}

	opts.Seed = req.PredictOptions.Seed
	opts.Tokens = req.PredictOptions.Tokens
	opts.Penalty = req.PredictOptions.Penalty
	opts.Repeat = req.PredictOptions.Repeat
	opts.Batch = req.PredictOptions.Batch
	opts.NKeep = req.PredictOptions.NKeep
	opts.TopK = req.PredictOptions.TopK
	opts.TopP = req.PredictOptions.TopP
	opts.TailFreeSamplingZ = req.PredictOptions.TailFreeSamplingZ
	opts.TypicalP = req.PredictOptions.TypicalP
	opts.Temperature = req.PredictOptions.Temperature
	opts.FrequencyPenalty = req.PredictOptions.FrequencyPenalty
	opts.PresencePenalty = req.PredictOptions.PresencePenalty
	opts.Mirostat = req.PredictOptions.Mirostat
	opts.MirostatTAU = req.PredictOptions.MirostatTAU
	opts.MirostatETA = req.PredictOptions.MirostatETA
	opts.MMap = req.PredictOptions.MMap

	return opts
}
