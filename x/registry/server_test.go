package registry

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"bllamo.com/registry/apitype"
	"bllamo.com/utils/backoff"
	"bllamo.com/utils/upload"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"kr.dev/diff"
)

const abc = "abcdefghijklmnopqrstuvwxyz"

func testPush(t *testing.T, chunkSize int64) {
	t.Run(fmt.Sprintf("chunkSize=%d", chunkSize), func(t *testing.T) {
		mc := startMinio(t, false)

		const MB = 1024 * 1024
		const FiveMB = 5 * MB

		// Upload two small layers and one large layer that will
		// trigger a multipart upload.
		manifest := []byte(`{
			"layers": [
				{"digest": "sha256-1", "size": 1},
				{"digest": "sha256-2", "size": 2},
				{"digest": "sha256-3", "size": 11000000}
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

func pushLayer(body io.ReaderAt, url string, off, n int64) (apitype.CompletePart, error) {
	var zero apitype.CompletePart
	if off < 0 {
		return zero, errors.New("off must be >0")
	}

	file := io.NewSectionReader(body, off, n)
	req, err := http.NewRequest("PUT", url, file)
	if err != nil {
		return zero, err
	}
	req.ContentLength = n

	// TODO(bmizerany): take content type param
	req.Header.Set("Content-Type", "text/plain")

	if n >= 0 {
		req.Header.Set("x-amz-copy-source-range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return zero, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		e := parseS3Error(res)
		return zero, fmt.Errorf("unexpected status code: %d; %w", res.StatusCode, e)
	}
	etag := strings.Trim(res.Header.Get("ETag"), `"`)
	cp := apitype.CompletePart{
		URL:  url,
		ETag: etag,
		// TODO(bmizerany): checksum
	}
	return cp, nil
}

// TestBasicPresignS3MultipartReferenceDoNotDelete tests the basic flow of
// presigning a multipart upload, uploading the parts, and completing the
// upload. It is for future reference and should not be deleted. This flow
// is tricky and if we get it wrong in our server, we can refer back to this
// as a "back to basics" test/reference.
func TestBasicPresignS3MultipartReferenceDoNotDelete(t *testing.T) {
	// t.Skip("skipping reference test; unskip when needed")

	mc := startMinio(t, true)
	mcc := &minio.Core{Client: mc}

	uploadID, err := mcc.NewMultipartUpload(context.Background(), "test", "theKey", minio.PutObjectOptions{})
	if err != nil {
		t.Fatal(err)
	}

	var completed []minio.CompletePart
	const size int64 = 10 * 1024 * 1024
	const chunkSize = 5 * 1024 * 1024

	for partNumber, c := range upload.Chunks(size, chunkSize) {
		u, err := mcc.Presign(context.Background(), "PUT", "test", "theKey", 15*time.Minute, url.Values{
			"partNumber": {strconv.Itoa(partNumber)},
			"uploadId":   {uploadID},
		})
		if err != nil {
			t.Fatalf("[partNumber=%d]: %v", partNumber, err)
		}
		t.Logf("[partNumber=%d]: %v", partNumber, u)

		var body abcReader
		cp, err := pushLayer(&body, u.String(), c.Offset, c.N)
		if err != nil {
			t.Fatalf("[partNumber=%d]: %v", partNumber, err)
		}
		t.Logf("completed part: %v", cp)

		// behave like server here (don't cheat and use partNumber)
		// instead get partNumber from the URL
		retPartNumber, err := strconv.Atoi(u.Query().Get("partNumber"))
		if err != nil {
			t.Fatalf("[partNumber=%d]: %v", partNumber, err)
		}

		completed = append(completed, minio.CompletePart{
			PartNumber: retPartNumber,
			ETag:       cp.ETag,
		})
	}

	defer func() {
		// fail if there are any incomplete uploads
		for x := range mcc.ListIncompleteUploads(context.Background(), "test", "theKey", true) {
			t.Errorf("incomplete: %v", x)
		}
	}()

	info, err := mcc.CompleteMultipartUpload(context.Background(), "test", "theKey", uploadID, completed, minio.PutObjectOptions{})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("completed: %v", info)

	// Check key in bucket
	obj, err := mc.GetObject(context.Background(), "test", "theKey", minio.GetObjectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer obj.Close()

	h := sha256.New()
	if _, err := io.Copy(h, obj); err != nil {
		t.Fatal(err)
	}
	gotSum := h.Sum(nil)

	h.Reset()
	var body abcReader
	if _, err := io.CopyN(h, &body, size); err != nil {
		t.Fatal(err)
	}
	wantSum := h.Sum(nil)

	if !bytes.Equal(gotSum, wantSum) {
		t.Errorf("got sum = %x; want %x", gotSum, wantSum)
	}
}

func availableAddr() string {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func startMinio(t *testing.T, trace bool) *minio.Client {
	t.Helper()

	// Trace is enabled by setting the OLLAMA_MINIO_TRACE environment or
	// explicitly setting trace to true.
	trace = cmp.Or(trace, os.Getenv("OLLAMA_MINIO_TRACE") != "")

	dir := t.TempDir() + "-keep" // prevent tempdir from auto delete

	t.Cleanup(func() {
		// TODO(bmizerany): trim temp dir based on dates so that
		// future runs may be able to inspect results for some time.
	})

	waitAndMaybeLogError := func(cmd *exec.Cmd) {
		if err := cmd.Wait(); err != nil {
			var e *exec.ExitError
			if errors.As(err, &e) {
				if !e.Exited() {
					// died due to our signal
					return
				}
				t.Errorf("%s stderr: %s", cmd.Path, e.Stderr)
				t.Errorf("%s exit status: %v", cmd.Path, e.ExitCode())
				t.Errorf("%s exited: %v", cmd.Path, e.Exited())
				t.Errorf("%s stderr: %s", cmd.Path, e.Stderr)
			} else {
				t.Errorf("%s exit error: %v", cmd.Path, err)
			}
		}
	}

	// Cancel must be called first so do wait to add to Cleanup
	// stack as last cleanup.
	ctx, cancel := context.WithCancel(context.Background())
	deadline, ok := t.Deadline()
	if ok {
		ctx, cancel = context.WithDeadline(ctx, deadline.Add(-100*time.Millisecond))
	}

	t.Logf(">> minio: minio server %s", dir)
	addr := availableAddr()
	cmd := exec.CommandContext(ctx, "minio", "server", "--address", addr, dir)
	cmd.Env = os.Environ()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cancel()
		waitAndMaybeLogError(cmd)
	})

	mc, err := minio.New(addr, &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		t.Fatal(err)
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

	if trace {
		cmd := exec.CommandContext(ctx, "mc", "admin", "trace", "--verbose", "test")
		cmd.Env = append(os.Environ(),
			"MC_HOST_test=http://minioadmin:minioadmin@"+addr,
		)

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}

		doneLogging := make(chan struct{})
		sc := bufio.NewScanner(stdout)
		go func() {
			defer close(doneLogging)

			// Scan lines until the process exits.
			for sc.Scan() {
				t.Logf("mc trace: %s", sc.Text())
			}
			_ = sc.Err() // ignore (not important)
		}()
		t.Cleanup(func() {
			cancel()
			waitAndMaybeLogError(cmd)

			// Make sure we do not log after test exists to
			// avoid panic.
			<-doneLogging
		})
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

// abcReader repeats the string s infinitely.
type abcReader struct {
	pos int
}

const theABCs = "abcdefghijklmnopqrstuvwxyz"

func (r *abcReader) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = theABCs[r.pos]
		r.pos++
		if r.pos == len(theABCs) {
			r.pos = 0
		}
	}
	return len(p), nil
}

func (r *abcReader) ReadAt(p []byte, off int64) (n int, err error) {
	for i := range p {
		p[i] = theABCs[(off+int64(i))%int64(len(theABCs))]
	}
	return len(p), nil
}
