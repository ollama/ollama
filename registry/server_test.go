package registry

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os/exec"
	"strings"
	"testing"
	"time"

	"bllamo.com/registry/apitype"
	"github.com/kr/pretty"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"kr.dev/diff"
)

func TestPush(t *testing.T) {
	startMinio(t)

	s := &Server{}
	hs := httptest.NewServer(s)
	t.Cleanup(hs.Close)
	c := &Client{BaseURL: hs.URL}

	manifest := []byte(`{
		"layers": [
			{"digest": "sha256-1", "size": 1},
			{"digest": "sha256-2", "size": 2},
			{"digest": "sha256-3", "size": 3}
		]
	}`)

	got, err := c.Push(context.Background(), "x+y", manifest)
	if err != nil {
		t.Fatal(err)
	}

	diff.Test(t, t.Errorf, got, []apitype.Requirement{
		{Digest: "sha256-1", Size: 1},
		{Digest: "sha256-2", Size: 2},
		{Digest: "sha256-3", Size: 3},
	}, diff.ZeroFields[apitype.Requirement]("URL"))

	for _, r := range got {
		body := strings.NewReader(strings.Repeat("x", int(r.Size)))
		if err := PushLayer(context.Background(), r.URL, r.Size, body); err != nil {
			t.Fatal(err)
		}
	}

	got, err = c.Push(context.Background(), "x+y", manifest)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 0 {
		t.Fatalf("unexpected requirements: % #v", pretty.Formatter(got))
	}

	mc, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		t.Fatal(err)
	}

	var paths []string
	keys := mc.ListObjects(context.Background(), "test", minio.ListObjectsOptions{
		Recursive: true,
	})
	for k := range keys {
		paths = append(paths, k.Key)
	}

	t.Logf("paths: %v", paths)

	diff.Test(t, t.Errorf, paths, []string{
		"blobs/sha256-1",
		"blobs/sha256-2",
		"blobs/sha256-3",
		"manifests/registry.ollama.ai/x/latest/Y",
	})

	obj, err := mc.GetObject(context.Background(), "test", "manifests/registry.ollama.ai/x/latest/Y", minio.GetObjectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer obj.Close()

	var gotM apitype.Manifest
	if err := json.NewDecoder(obj).Decode(&gotM); err != nil {
		t.Fatal(err)
	}

	diff.Test(t, t.Errorf, gotM, apitype.Manifest{
		Layers: []apitype.Layer{
			{Digest: "sha256-1", Size: 1},
			{Digest: "sha256-2", Size: 2},
			{Digest: "sha256-3", Size: 3},
		},
	})
}

func startMinio(t *testing.T) {
	t.Helper()

	dir := t.TempDir()
	cmd := exec.Command("minio", "server", "--address", "localhost:9000", dir)

	// TODO(bmizerany): wait delay etc...
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		if err := cmd.Wait(); err != nil {
			t.Log(err)
		}
	})

	mc, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		t.Fatal(err)
	}

	// wait for server to start
	// TODO(bmizerany): use backoff
	for {
		_, err := mc.ListBuckets(context.Background())
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if err := mc.MakeBucket(context.Background(), "test", minio.MakeBucketOptions{}); err != nil {
		t.Fatal(err)
	}
}
