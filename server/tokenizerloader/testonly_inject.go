//go:build test

package tokenizerloader

import "context"

// NOTE: In loader.go, ensure openVocabOnly is declared as a *var*, e.g.:
//    var openVocabOnly = func(ctx context.Context, model string) (Tokenizer, error) { ... }

func SetOpenVocabOnlyForTest(f func(context.Context, string) (Tokenizer, error)) {
	openVocabOnly = f
}
