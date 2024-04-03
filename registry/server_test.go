package registry

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"bllamo.com/registry/apitype"
	"bllamo.com/utils/backoff"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"kr.dev/diff"
)

const abc = "abcdefghijklmnopqrstuvwxyz"

func testPush(t *testing.T, chunkSize int64) {
	t.Run(fmt.Sprintf("chunkSize=%d", chunkSize), func(t *testing.T) {
		mc := startMinio(t, false)

		manifest := []byte(`{
			"layers": [
				{"digest": "sha256-1", "size": 1},
				{"digest": "sha256-2", "size": 2},
				{"digest": "sha256-3", "size": 3}
			]
		}`)

		const ref = "registry.ollama.ai/x/y:latest+Z"

		hs := httptest.NewServer(&Server{
			minioClient:     mc,
			UploadChunkSize: chunkSize,
		})
		t.Cleanup(hs.Close)
		c := &Client{BaseURL: hs.URL}

		requirements, err := c.Push(context.Background(), ref, manifest, nil)
		if err != nil {
			t.Fatal(err)
		}

		if len(requirements) < 3 {
			t.Fatalf("expected at least 3 requirements; got %d", len(requirements))
			t.Logf("requirements: %v", requirements)
		}

		var uploaded []apitype.CompletePart
		for i, r := range requirements {
			t.Logf("[%d] pushing layer: offset=%d size=%d", i, r.Offset, r.Size)

			body := strings.NewReader(abc)
			etag, err := PushLayer(context.Background(), r.URL, r.Offset, r.Size, body)
			if err != nil {
				t.Fatal(err)
			}
			uploaded = append(uploaded, apitype.CompletePart{
				URL:  r.URL,
				ETag: etag,
			})
		}

		requirements, err = c.Push(context.Background(), ref, manifest, &PushParams{
			Uploaded: uploaded,
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(requirements) != 0 {
			t.Fatalf("unexpected requirements: %v", requirements)
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
			"manifests/registry.ollama.ai/x/y/latest/Z",
		})

		obj, err := mc.GetObject(context.Background(), "test", "manifests/registry.ollama.ai/x/y/latest/Z", minio.GetObjectOptions{})
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

		// checksum the blobs
		for i, l := range gotM.Layers {
			obj, err := mc.GetObject(context.Background(), "test", "blobs/"+l.Digest, minio.GetObjectOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer obj.Close()

			info, err := obj.Stat()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("[%d] layer info: name=%q l.Size=%d size=%d", i, info.Key, l.Size, info.Size)

			data, err := io.ReadAll(obj)
			if err != nil {
				t.Fatal(err)
			}

			got := string(data)
			want := abc[:l.Size]
			if got != want {
				t.Errorf("[%d] got layer data = %q; want %q", i, got, want)
			}
		}
	})
}

func TestPush(t *testing.T) {
	testPush(t, 0)
	testPush(t, 1)
}

func availableAddr() string {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func startMinio(t *testing.T, debug bool) *minio.Client {
	t.Helper()

	dir := t.TempDir()
	t.Logf(">> minio data dir: %s", dir)
	addr := availableAddr()
	cmd := exec.Command("minio", "server", "--address", addr, dir)
	cmd.Env = os.Environ()

	if debug {
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		doneLogging := make(chan struct{})
		t.Cleanup(func() {
			<-doneLogging
		})
		go func() {
			defer close(doneLogging)
			sc := bufio.NewScanner(stdout)
			for sc.Scan() {
				t.Logf("minio: %s", sc.Text())
			}
		}()
	}

	// TODO(bmizerany): wait delay etc...
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		if err := cmd.Wait(); err != nil {
			var e *exec.ExitError
			if errors.As(err, &e) && e.Exited() {
				t.Logf("minio stderr: %s", e.Stderr)
				t.Logf("minio exit status: %v", e.ExitCode())
				t.Logf("minio exited: %v", e.Exited())
				t.Error(err)
			}
		}
	})

	mc, err := minio.New(addr, &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	deadline, ok := t.Deadline()
	if ok {
		ctx, cancel = context.WithDeadline(ctx, deadline.Add(-100*time.Millisecond))
		defer cancel()
	}

	// wait for server to start with exponential backoff
	for _, err := range backoff.Upto(ctx, 1*time.Second) {
		if err != nil {
			t.Fatal(err)
		}
		if mc.IsOnline() {
			break
		}
	}

	if err := mc.MakeBucket(context.Background(), "test", minio.MakeBucketOptions{}); err != nil {
		t.Fatal(err)
	}

	return mc
}

// contextForTest returns a context that is canceled when the test deadline,
// if any, is reached. The returned doneLogging function should be called
// after all Log/Error/Fatalf calls are done before the test returns.
func contextForTest(t *testing.T) (_ context.Context, doneLogging func()) {
	done := make(chan struct{})
	deadline, ok := t.Deadline()
	if !ok {
		return context.Background(), func() {}
	}

	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-100*time.Millisecond))
	t.Cleanup(func() {
		cancel()
		<-done
	})
	return ctx, func() { close(done) }
}
