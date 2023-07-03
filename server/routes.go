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

	var l *llama.LLama

	gpulayers := 1
	tokens := 512
	threads := runtime.NumCPU()
	model := "/Users/pdevine/.cache/gpt4all/GPT4All-13B-snoozy.ggmlv3.q4_0.bin"

	r.POST("/api/load", func(c *gin.Context) {
		var err error
		l, err = llama.New(model, llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(gpulayers))
		if err != nil {
			fmt.Println("Loading the model failed:", err.Error())
		}
	})

	r.POST("/api/unload", func(c *gin.Context) {
	})

	r.POST("/api/generate", func(c *gin.Context) {
		var req api.GenerateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
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

/*
                embeds, err := l.Embeddings(text)
                if err != nil {
                        fmt.Printf("Embeddings: error %s \n", err.Error())
                }
*/
		
	})

	log.Printf("Listening on %s", ln.Addr())
	s := &http.Server{
		Handler: r,
	}

	return s.Serve(ln)
}
