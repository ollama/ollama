package decisiontest

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/decision"
)

const fixture = `{"id":"one","dataset":"test","request":{"state":"A test message.","questions":{"answer":{"type":"noul","instructions":"Is this a message?"}}},"expected":{"answer":{"noul":true}}}`

func FuzzResponse(f *testing.F) {
	var c Case
	if err := json.Unmarshal([]byte(fixture), &c); err != nil {
		f.Fatal(err)
	}
	c.Request.Model = "test"
	f.Add([]byte(`{"model":"test","usage":{"input_tokens":1,"output_tokens":0},"answers":{"answer":{"type":"noul","noul":1}}}`))
	f.Add([]byte(`{"model":"test","answers":{"answer":{"type":"noul","noul":null}}}`))
	f.Fuzz(func(t *testing.T, body []byte) {
		if response, err := decode(c.Request, body); err == nil {
			c.Mismatches(response)
			p := response.Answers["answer"].Noul
			if p == nil || *p < 0 || *p > 1 {
				t.Fatalf("accepted invalid probability: %v", p)
			}
		}
	})
}

func TestLoadCorpus(t *testing.T) {
	dir := t.TempDir()
	var pngData bytes.Buffer
	if err := png.Encode(&pngData, image.NewRGBA(image.Rect(0, 0, 2, 3))); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "image.png"), pngData.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "cases.jsonl")
	withImage := strings.TrimSuffix(fixture, "}") + `,"image_files":["image.png"]}`
	if err := os.WriteFile(file, []byte(withImage+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	if !c.HasImages() || c.Cases[0].ImagePixels != 6 || c.Cases[0].ImageBytes != pngData.Len() || c.Cases[0].Options() != 2 {
		t.Fatalf("bad image metadata: %+v", c.Cases[0])
	}
	if !bytes.Equal(c.Cases[0].Request.Images[0], pngData.Bytes()) {
		t.Fatal("image contents changed")
	}
	firstHash := c.SHA256
	if err := os.WriteFile(filepath.Join(dir, "image.png"), append(pngData.Bytes(), 0), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err = Load(file)
	if err != nil || c.SHA256 == firstHash {
		t.Fatalf("image change was not fingerprinted: %v", err)
	}
}

func TestRejectInvalidCorpus(t *testing.T) {
	for name, data := range map[string]string{
		"empty":                  "\n",
		"duplicate id":           fixture + "\n" + fixture,
		"trailing object":        fixture + `{}`,
		"unknown field":          strings.TrimSuffix(fixture, "}") + `,"unexpected":{}}`,
		"missing expectation":    strings.Replace(fixture, `"answer":{"noul":true}`, `"other":{"noul":true}`, 1),
		"wrong expectation type": strings.Replace(fixture, `"noul":true`, `"choice":"yes"`, 1),
		"model in data":          strings.Replace(fixture, `"state":`, `"model":"ignored","state":`, 1),
		"missing image":          strings.TrimSuffix(fixture, "}") + `,"image_files":["missing.png"]}`,
		"escaping image":         strings.TrimSuffix(fixture, "}") + `,"image_files":["../private.png"]}`,
	} {
		t.Run(name, func(t *testing.T) {
			file := filepath.Join(t.TempDir(), "cases.jsonl")
			if err := os.WriteFile(file, []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(file); err == nil {
				t.Fatal("accepted invalid corpus")
			}
		})
	}
}

func TestCorpusHashFramesImageContents(t *testing.T) {
	dir := t.TempDir()
	var data bytes.Buffer
	if err := png.Encode(&data, image.NewRGBA(image.Rect(0, 0, 2, 3))); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "cases.jsonl")
	if err := os.WriteFile(file, []byte(strings.TrimSuffix(fixture, "}")+`,"image_files":["a.png","b.png"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var hashes []string
	for _, sizes := range [][2]int{{1, 2}, {2, 1}} {
		for i, name := range []string{"a.png", "b.png"} {
			if err := os.WriteFile(filepath.Join(dir, name), bytes.Repeat(data.Bytes(), sizes[i]), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		corpus, err := Load(file)
		if err != nil {
			t.Fatal(err)
		}
		hashes = append(hashes, corpus.SHA256)
	}
	if hashes[0] == hashes[1] {
		t.Fatal("different image inputs with the same concatenated bytes have the same corpus hash")
	}
}

func TestResponseContractAndLabels(t *testing.T) {
	var req decision.Request
	if err := json.Unmarshal([]byte(`{"model":"test","state":"A test.","questions":{
		"yes":{"type":"noul"},
		"topic":{"type":"choice","criteria":{"a":"A","b":"B"}},
		"rating":{"type":"score","criteria":["Low","High"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	valid := `{"model":"test","total_duration":100,"load_duration":20,"usage":{"input_tokens":10,"output_tokens":0},"answers":{
		"yes":{"type":"noul","noul":0},
		"topic":{"type":"choice","choice":"a","probabilities":{"a":1,"b":0},"confidence":1},
		"rating":{"type":"score","score":0,"probabilities":{"0":1,"1":0},"confidence":1,"legend":{"0":"Low","1":"High"}}}}`
	for _, mutation := range []struct{ old, replacement string }{
		{`"noul":0`, `"noul":null`},
		{`"input_tokens":10,`, ""},
		{`"output_tokens":0`, `"output_tokens":null`},
		{`"a":1`, `"a":0.7`},
		{`"choice":"a"`, `"choice":"b"`},
		{`"confidence":1`, `"confidence":0`},
		{`"score":0`, `"score":1`},
		{`"0":"Low"`, `"0":"Wrong"`},
		{`"total_duration":100`, `"total_duration":-1`},
		{`"load_duration":20`, `"load_duration":-1`},
		{`"load_duration":20`, `"load_duration":101`},
	} {
		t.Run(mutation.old, func(t *testing.T) {
			if _, err := decode(req, []byte(strings.Replace(valid, mutation.old, mutation.replacement, 1))); err == nil {
				t.Fatal("accepted broken response contract")
			}
		})
	}
	result, err := decode(req, []byte(valid))
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := decode(req, []byte(strings.Replace(valid, `"total_duration":100,"load_duration":20,`, "", 1)))
	if err != nil || legacy.TotalDuration != nil || legacy.LoadDuration != nil {
		t.Fatalf("response without timings: %+v, %v", legacy, err)
	}
	no, a, low := false, "a", 0
	c := Case{Expected: map[string]Expected{"yes": {Noul: &no}, "topic": {Choice: &a}, "rating": {Level: &low}}}
	if wrong := c.Mismatches(result); len(wrong) != 0 {
		t.Fatalf("correct labels failed: %v", wrong)
	}
	a = "b"
	if wrong := c.Mismatches(result); len(wrong) != 1 {
		t.Fatalf("wrong label did not fail: %v", wrong)
	}
	if _, err := Mode("skip-low-confidence"); err == nil {
		t.Fatal("accepted an unknown evaluation policy")
	}
}
