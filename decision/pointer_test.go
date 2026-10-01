package decision

import "testing"

func TestIsPointerFamily(t *testing.T) {
	if !IsPointerFamily(PointerHeadFamily, nil) {
		t.Fatal("family should select the pointer-head runner")
	}
	if !IsPointerFamily("qwen3", []string{PointerHeadFamily}) {
		t.Fatal("families should select the pointer-head runner")
	}
	if IsPointerFamily("qwen3", []string{"qwen3"}) {
		t.Fatal("a normal architecture must keep letter-token scoring")
	}
}

func TestPointerRunnerEndpoint(t *testing.T) {
	endpoint, err := PointerRunnerEndpoint(" http://127.0.0.1:8000 ")
	if err != nil || endpoint != "http://127.0.0.1:8000/v1/systemone" {
		t.Fatalf("endpoint = %q, %v", endpoint, err)
	}
	endpoint, err = PointerRunnerEndpoint("http://localhost:8000/v1/systemone")
	if err != nil || endpoint != "http://localhost:8000/v1/systemone" {
		t.Fatalf("existing path = %q, %v", endpoint, err)
	}
	if _, err := PointerRunnerEndpoint("http://example.com:8000"); err == nil {
		t.Fatal("non-loopback runner must be rejected")
	}
	if _, err := PointerRunnerEndpoint("file:///tmp/runner"); err == nil {
		t.Fatal("non-http runner must be rejected")
	}
	endpoint, err = PointerRunnerEndpoint("")
	if err != nil || endpoint != "" {
		t.Fatalf("empty runner = %q, %v", endpoint, err)
	}
}
