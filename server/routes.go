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

func Serve(ln net.Listener) error {
	r := gin.Default()

	// TODO: these should be request parameters
	gpulayers := 0
	tokens := 512
	threads := runtime.NumCPU()

	r.POST("/api/generate", func(c *gin.Context) {
		// TODO: set prompt from template
		fmt.Println("Generating text...")

		var req api.GenerateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
			return
		}

		fmt.Println(req)

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
	})

	log.Printf("Listening on %s", ln.Addr())
	s := &http.Server{
		Handler: r,
	}

	return s.Serve(ln)
}
