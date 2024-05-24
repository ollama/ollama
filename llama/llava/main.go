package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/ollama/ollama/llama"
)

func main() {
	mp := flag.String("model", "", "Path to model binary file")
	pp := flag.String("projector", "", "Path to projector binary file")
	image := flag.String("image", "", "Path to image file")
	prompt := flag.String("prompt", " [INST] What is in the picture? <image> [/INST]", "Prompt including <image> tag")
	flag.Parse()

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams()
	model := llama.LoadModelFromFile(*mp, params)
	ctxParams := llama.NewContextParams()

	// language model context
	lc := llama.NewContextWithModel(model, ctxParams)

	// clip context
	clipCtx := llama.NewClipContext(*pp)

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

	err = eval(lc, parts[0], embedding, parts[1])
	if err != nil {
		panic(err)
	}
}

func eval(lc *llama.Context, before string, embedding *llama.LlavaImageEmbed, after string) error {
	beforeTokens, err := lc.Model().Tokenize(before, 2048, true, true)
	if err != nil {
		return err
	}

	afterTokens, err := lc.Model().Tokenize(after, 2048, true, true)
	if err != nil {
		return err
	}

	// eval before
	batch := llama.NewBatch(512, 0, 1)

	var nPast int

	// prompt eval
	for _, t := range beforeTokens {
		batch.Add(t, nPast, []int{0}, true)
		nPast++
	}

	err = lc.Decode(batch)
	if err != nil {
		return err
	}

	// batch.Clear()

	llama.LlavaEvalImageEmbed(lc, embedding, 512, &nPast)

	batch = llama.NewBatch(512, 0, 1)
	for _, t := range afterTokens {
		batch.Add(t, nPast, []int{0}, true)
	}

	// main loop
	for n := nPast; n < 4096; n++ {
		err = lc.Decode(batch)
		if err != nil {
			panic("Failed to decode")
		}

		// sample a token
		token := lc.SampleTokenGreedy(batch)

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

	return nil
}
