package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/ollama/ollama/template"
)

func main() {
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "usage: %s <template.gotmpl>\n", os.Args[0])
		os.Exit(2)
	}
	path := flag.Arg(0)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}

	out, err := template.Format(string(data))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(out)
}
