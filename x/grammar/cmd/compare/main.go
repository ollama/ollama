//go:build mlx

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/x/grammar"
	"github.com/ollama/ollama/x/grammar/schema"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

const jsonGBNF = `
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws "}" | "{" members "}"
members ::= member ("," member)*
member ::= ws string ws ":" element
array ::= "[" ws "]" | "[" elements "]"
elements ::= element ("," element)*
element ::= ws value ws
string ::= "\"" characters "\""
characters ::= character*
character ::= [^"\\] | "\\" escape
escape ::= ["\\bfnrt]
number ::= "-"? integer fraction? exponent?
integer ::= "0" | [1-9] [0-9]*
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
`

type result struct {
	vocabSize           int    `json:"vocab_size"`
	Iterations          int    `json:"iterations"`
	Warmup              int    `json:"warmup"`
	ConstrainedSource   string `json:"constrained_source"`
	LlamaSource         string `json:"llama_source"`
	LlamaApply          string `json:"llama_apply"`
	ConstrainedGraph    string `json:"constrained_graph"`
	ConstrainedWithEval string `json:"constrained_with_eval,omitempty"`
	EvalOnly            string `json:"eval_only,omitempty"`
	ConstrainedEvalNet  string `json:"constrained_eval_net,omitempty"`
}

func main() {
	var (
		vocabSize  = flag.Int("vocab-size", 128000, "Vocabulary size")
		iterations = flag.Int("iterations", 500, "Benchmark iterations")
		warmup     = flag.Int("warmup", 50, "Warmup iterations")
		withEval   = flag.Bool("eval", true, "Measure ApplyMask with mlx.Eval")
		gbnfPath   = flag.String("gbnf", "", "GBNF grammar file for llama.cpp")
		schemaPath = flag.String("schema", "", "JSON Schema file for grammar constraints")
		ebnfPath   = flag.String("ebnf", "", "EBNF grammar file for grammar constraints")
		startRule  = flag.String("start", "root", "Start rule for EBNF")
	)
	flag.Parse()

	if *vocabSize <= 0 || *iterations <= 0 || *warmup < 0 {
		fmt.Fprintln(os.Stderr, "invalid flags")
		os.Exit(2)
	}

	vocab := createVocab(*vocabSize)

	if *schemaPath != "" && *ebnfPath != "" {
		fmt.Fprintln(os.Stderr, "only one of -schema or -ebnf may be set")
		os.Exit(2)
	}

	var constrainedSource string
	var compiled *grammar.Grammar
	var err error
	switch {
	case *schemaPath != "":
		data, readErr := os.ReadFile(*schemaPath)
		if readErr != nil {
			fmt.Fprintf(os.Stderr, "read schema: %v\n", readErr)
			os.Exit(1)
		}
		compiled, err = schema.Grammar(string(data))
		constrainedSource = "schema:" + *schemaPath
	case *ebnfPath != "":
		data, readErr := os.ReadFile(*ebnfPath)
		if readErr != nil {
			fmt.Fprintf(os.Stderr, "read ebnf: %v\n", readErr)
			os.Exit(1)
		}
		compiled, err = grammar.ParseEBNF(string(data), *startRule)
		constrainedSource = "ebnf:" + *ebnfPath
	default:
		compiled, err = grammar.JSONGrammar()
		constrainedSource = "json"
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "grammar: %v\n", err)
		os.Exit(1)
	}
	engine, err := grammar.NewEngine(compiled, vocab)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	logits := mlx.Ones(int32(*vocabSize))
	mlx.Keep(logits)

	for i := 0; i < *warmup; i++ {
		masked := engine.ApplyMask(logits)
		if *withEval {
			mlx.Eval(masked)
		}
	}

	graphAvg := measure(*iterations, func() {
		_ = engine.ApplyMask(logits)
	})

	var evalAvg time.Duration
	var evalOnlyAvg time.Duration
	if *withEval {
		evalOnlyAvg = measure(*iterations, func() {
			baseline := mlx.MulScalar(logits, 1)
			mlx.Eval(baseline)
			baseline.Free()
		})

		evalAvg = measure(*iterations, func() {
			masked := engine.ApplyMask(logits)
			mlx.Eval(masked)
		})
	}

	vocabIDs := make([]uint32, *vocabSize)
	for i := range vocabIDs {
		vocabIDs[i] = uint32(i)
	}
	eogTokens := []int32{0}

	gbnf := jsonGBNF
	llamaSource := "json"
	if *gbnfPath != "" {
		data, readErr := os.ReadFile(*gbnfPath)
		if readErr != nil {
			fmt.Fprintf(os.Stderr, "read gbnf: %v\n", readErr)
			os.Exit(1)
		}
		gbnf = string(data)
		llamaSource = *gbnfPath
	}

	llamaGrammar := llama.NewGrammar(gbnf, vocabIDs, vocab, eogTokens)
	if llamaGrammar == nil {
		fmt.Fprintln(os.Stderr, "llama grammar initialization failed")
		os.Exit(1)
	}
	defer llamaGrammar.Free()

	llamaTokens := make([]llama.TokenData, *vocabSize)

	for i := 0; i < *warmup; i++ {
		for j := range llamaTokens {
			llamaTokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(llamaTokens)
	}

	llamaAvg := measure(*iterations, func() {
		for j := range llamaTokens {
			llamaTokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(llamaTokens)
	})

	out := result{
		vocabSize:         *vocabSize,
		Iterations:        *iterations,
		Warmup:            *warmup,
		LlamaApply:        llamaAvg.String(),
		ConstrainedGraph:  graphAvg.String(),
		ConstrainedSource: constrainedSource,
		LlamaSource:       llamaSource,
	}
	if *withEval {
		out.ConstrainedWithEval = evalAvg.String()
		out.EvalOnly = evalOnlyAvg.String()
		if evalAvg > evalOnlyAvg {
			out.ConstrainedEvalNet = (evalAvg - evalOnlyAvg).String()
		} else {
			out.ConstrainedEvalNet = "0s"
		}
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
}

func measure(iterations int, fn func()) time.Duration {
	start := time.Now()
	for i := 0; i < iterations; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(iterations)
}

func createVocab(size int) []string {
	vocab := make([]string, size)

	jsonTokens := []string{
		"{", "}", "[", "]", ":", ",",
		"true", "false", "null",
		" ", "\n", "\t", "\r",
		"\"",
	}
	for i, t := range jsonTokens {
		if i < size {
			vocab[i] = t
		}
	}

	for i := len(jsonTokens); i < size; i++ {
		vocab[i] = fmt.Sprintf("tok%d", i)
	}

	return vocab
}
