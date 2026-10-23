// Command forkversion prints the latest or next ds41 release tag.
//
//	go run ./scripts/forkversion latest
//	go run ./scripts/forkversion next patch
//	go run ./scripts/forkversion next v0.40.5-b11724-ds41.4
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/ollama/ollama/app/forkversion"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	tags, err := gitTags()
	if err != nil {
		die(err)
	}
	switch os.Args[1] {
	case "latest":
		if len(os.Args) != 2 {
			usage()
		}
		tag, err := forkversion.Latest(tags)
		if err != nil {
			die(err)
		}
		fmt.Println(tag)
	case "next":
		if len(os.Args) != 3 {
			usage()
		}
		tag, err := forkversion.Next(os.Args[2], tags)
		if err != nil {
			die(err)
		}
		fmt.Println(tag)
	default:
		usage()
	}
}

func gitTags() ([]string, error) {
	out, err := exec.Command("git", "tag", "-l").Output()
	if err != nil {
		return nil, err
	}
	var tags []string
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			tags = append(tags, line)
		}
	}
	return tags, nil
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run ./scripts/forkversion latest|next <patch|vX.Y.Z-bNNNNN-ds41.N>\n")
	os.Exit(2)
}

func die(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
