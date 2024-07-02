package cmd

import (
	"testing"
)

func BenchmarkCreateLocalBlob(b *testing.B) {
	for i := 0; i < b.N; i++ {
		dest := b.TempDir() + "/hi"

		createBlobLocal("/Users/joshyan/.ollama/models/blobs/sha256-edd739ebd0b09f4e9345e8dc76d06ec37d08a080246560e57f7f1443fa3e57af", dest)
	}
}
