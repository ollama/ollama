package tokenizerloader

import "context"

// Compiled only when running `go test` because of the `_test.go` suffix.

func ResetForTest() {
	reset()
}

func SetOpenVocabOnlyForTest(f func(context.Context, string) (Tokenizer, error)) {
	openVocabOnly = f
}
