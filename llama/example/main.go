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
	params := llama.ModelParams{
		NumGpuLayers: 999,
		MainGpu:      0,
		UseMmap:      true,
		Progress: func(p float32) {
			fmt.Printf("loading... %f\n", p)
		},
	}
	model := llama.LoadModelFromFile(*mpath, params)
	ctxParams := llama.NewContextParams(2048, 1024, 1, runtime.NumCPU(), false)

	// language model context
	lc := llama.NewContextWithModel(model, ctxParams)

	// eval before
	batch := llama.NewBatch(1024, 0, 1)
	var nPast int

	// multi-modal
	if *ppath != "" {
		// clip context
		clipCtx := llama.NewClipContext(*ppath)

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

		// generate embeddings for the image
		embedding := llama.NewLlavaImageEmbed(lc, clipCtx, data)

		parts := strings.Split(*prompt, "<image>")
		if len(parts) != 2 {
			panic("prompt must contain exactly one <image>")
		}

		// process text before the image
		beforeTokens, err := lc.Model().Tokenize(parts[0], true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range beforeTokens {
			batch.Add(t, nil, nPast, []int{0}, false)
			nPast++
		}

		err = lc.Decode(batch)
		if err != nil {
			panic(err)
		}
		batch.Clear()

		// set up a separate batch for image embeddings
		imageBatch := llama.NewBatch(1024, lc.Model().NEmbd(), 1)
		for _, e := range embedding {
			imageBatch.Add(0, e, nPast, []int{0}, false)
			nPast++
		}

		err = lc.Decode(imageBatch)
		if err != nil {
			panic(err)
		}

		// process text after the image
		afterTokens, err := lc.Model().Tokenize(parts[1], true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range afterTokens {
			batch.Add(t, nil, nPast, []int{0}, true)
			nPast++
		}
	} else {
		tokens, err := lc.Model().Tokenize(*prompt, true, true)
		if err != nil {
			panic(err)
		}

		for _, t := range tokens {
			batch.Add(t, nil, nPast, []int{0}, true)
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
		batch.Add(token, nil, n, []int{0}, true)
	}
}
