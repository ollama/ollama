package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("Run as 'go test -bench=.' to run the benchmarks")
	os.Exit(1)
}
