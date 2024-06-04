package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/ollama/ollama/llama"
)

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("projector", "", "Path to projector binary file")
	image := flag.String("image", "", "Path to image file")
	prompt := flag.String("prompt", "", "Prompt including <image> tag")
	flag.Parse()

	if *mpath == "" {
		panic("model path is required")
	}

	if *prompt == "" {
		panic("prompt is required")
	}

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams(999, 0, func(p float32) {
		fmt.Printf("loading... %f\n", p)
	})
	model := llama.LoadModelFromFile(*mpath, params)
	ctxParams := llama.NewContextParams(2048, runtime.NumCPU(), false)

	// language model context
	lc := llama.NewContextWithModel(model, ctxParams)

	// eval before
	batch := llama.NewBatch(512, 0, 1)
	var nPast int

	// clip context
	var clipCtx *llama.ClipContext

	// multi-modal
	if *ppath != "" {
		clipCtx = llama.NewClipContext(*ppath)

		// open image file
		file, err := os.Open(*image)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		data, err := io.ReadAll(file)
		if err != nil {
			log.Fatal(err)
		}

		embedding := llama.NewLlavaImageEmbed(clipCtx, data)

		parts := strings.Split(*prompt, "<image>")
		if len(parts) != 2 {
			panic("prompt must contain exactly one <image>")
		}

		beforeTokens, err := lc.Model().Tokenize(parts[0], true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range beforeTokens {
			batch.Add(t, nPast, []int{0}, true)
			nPast++
		}

		err = lc.Decode(batch)
		if err != nil {
			panic(err)
		}

		llama.LlavaEvalImageEmbed(lc, embedding, 512, &nPast)

		afterTokens, err := lc.Model().Tokenize(parts[1], true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range afterTokens {
			batch.Add(t, nPast, []int{0}, true)
			nPast++
		}
	} else {
		tokens, err := lc.Model().Tokenize(*prompt, true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range tokens {
			batch.Add(t, nPast, []int{0}, true)
			nPast++
		}
	}

	// main loop
	for n := nPast; n < 4096; n++ {
		err := lc.Decode(batch)
		if err != nil {
			panic(err)
		}

		// sample a token
		logits := lc.GetLogitsIth(batch.NumTokens() - 1)
		token := lc.SampleTokenGreedy(logits)

		// if it's an end of sequence token, break
		if lc.Model().TokenIsEog(token) {
			break
		}

		// print the token
		str := lc.Model().TokenToPiece(token)
		fmt.Print(str)
		batch.Clear()
		batch.Add(token, n, []int{0}, true)
	}
}
