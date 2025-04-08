package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestDownloadBlobCheckDigest(t *testing.T) {
	testOpts := downloadOpts{
		mp: ModelPath{
			ProtocolScheme: "https",
			Registry:       "registry.ollama.ai",
			Namespace:      "library",
			Repository:     "tensorflow",
			Tag:            "latest",
		},
		regOpts: nil,
		fn:      func(api.ProgressResponse) {},
	}

	testOpts.digest = "foo"
	if _, err := downloadBlob(context.TODO(), testOpts); err != nil {
		if err.Error() != "invalid digest" {
			t.Fatal(err)
		}
	}

	testOpts.digest = "sha256:foo"
	if _, err := downloadBlob(context.TODO(), testOpts); err != nil {
		if err.Error() != "invalid digest" {
			t.Fatal(err)
		}
	}
}
