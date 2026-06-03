package model

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// fakeSafetensorsBlob writes a header-only safetensors-format blob to t.TempDir
// (no tensor body needed since the scanners only parse the header).
func fakeSafetensorsBlob(t *testing.T, header map[string]any) string {
	t.Helper()
	hdr, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	path := filepath.Join(t.TempDir(), "blob.safetensors")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(hdr))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(hdr); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadBlobTensorQuantInfo_OllamaNativeSingularNaming(t *testing.T) {
	hdr := map[string]any{
		"__metadata__":     map[string]string{"quant_type": "INT4", "group_size": "32"},
		"foo.weight":       map[string]any{"dtype": "U32", "shape": []int{4, 8}, "data_offsets": []int{0, 0}},
		"foo.weight.scale": map[string]any{"dtype": "U8", "shape": []int{4, 2}, "data_offsets": []int{0, 0}},
	}
	infos, gqt, ggs, err := readBlobTensorQuantInfo(fakeSafetensorsBlob(t, hdr))
	if err != nil {
		t.Fatal(err)
	}
	if gqt != "INT4" || ggs != 32 {
		t.Errorf("globals = (%q, %d), want (INT4, 32)", gqt, ggs)
	}
	info := infos["foo.weight"]
	if info == nil || info.QuantType != "INT4" || info.GroupSize != 32 {
		t.Errorf("infos[foo.weight] = %+v, want {INT4, 32}", info)
	}
}

func TestReadBlobTensorQuantInfo_ErrorPaths(t *testing.T) {
	t.Run("missing_file", func(t *testing.T) {
		if _, _, _, err := readBlobTensorQuantInfo("/nonexistent/path"); err == nil {
			t.Error("want error")
		}
	})
	t.Run("truncated_header_size", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "truncated")
		if err := os.WriteFile(p, []byte{0x01, 0x02}, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := readBlobTensorQuantInfo(p); err == nil {
			t.Error("want error")
		}
	})
	t.Run("oversize_header", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "oversize")
		f, _ := os.Create(p)
		_ = binary.Write(f, binary.LittleEndian, uint64(200*1024*1024))
		f.Close()
		if _, _, _, err := readBlobTensorQuantInfo(p); err == nil {
			t.Error("want error")
		}
	})
	t.Run("malformed_json", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "malformed")
		f, _ := os.Create(p)
		_ = binary.Write(f, binary.LittleEndian, uint64(10))
		f.WriteString("not-json!!")
		f.Close()
		if _, _, _, err := readBlobTensorQuantInfo(p); err == nil {
			t.Error("want error")
		}
	})
}

func TestMainTensorNames_SkipsSingularAuxSuffixes(t *testing.T) {
	hdr := map[string]json.RawMessage{
		"foo.weight":       json.RawMessage(`{}`),
		"foo.weight.scale": json.RawMessage(`{}`),
		"foo.weight.bias":  json.RawMessage(`{}`),
		"bar.weight":       json.RawMessage(`{}`),
		"__metadata__":     json.RawMessage(`{}`),
	}
	got := mainTensorNames(hdr)
	want := []string{"bar.weight", "foo.weight"}
	sort.Strings(got)
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestInferQuantTypeFromShapes_SingularAuxName(t *testing.T) {
	hdr := map[string]json.RawMessage{
		"foo.weight":       json.RawMessage(`{"dtype":"U32","shape":[4,8]}`),
		"foo.weight.scale": json.RawMessage(`{"dtype":"U8","shape":[4,2]}`),
	}
	qt, gs := inferQuantTypeFromShapes(hdr, "foo.weight", "")
	if qt != "INT4" || gs != 32 {
		t.Errorf("got (%q, %d), want (INT4, 32)", qt, gs)
	}
}

func TestParseGlobalQuantMetadata(t *testing.T) {
	t.Run("no_metadata_key", func(t *testing.T) {
		qt, gs := parseGlobalQuantMetadata(map[string]json.RawMessage{"foo": json.RawMessage(`{}`)})
		if qt != "" || gs != 0 {
			t.Errorf("got (%q, %d), want empty", qt, gs)
		}
	})
	t.Run("empty_metadata", func(t *testing.T) {
		qt, gs := parseGlobalQuantMetadata(map[string]json.RawMessage{"__metadata__": json.RawMessage(`{}`)})
		if qt != "" || gs != 0 {
			t.Errorf("got (%q, %d), want empty", qt, gs)
		}
	})
	t.Run("populated", func(t *testing.T) {
		qt, gs := parseGlobalQuantMetadata(map[string]json.RawMessage{"__metadata__": json.RawMessage(`{"quant_type":"INT8","group_size":"64"}`)})
		if qt != "INT8" || gs != 64 {
			t.Errorf("got (%q, %d), want (INT8, 64)", qt, gs)
		}
	})
	t.Run("non_numeric_group_size", func(t *testing.T) {
		qt, gs := parseGlobalQuantMetadata(map[string]json.RawMessage{"__metadata__": json.RawMessage(`{"quant_type":"INT4","group_size":"bogus"}`)})
		if qt != "INT4" || gs != 0 {
			t.Errorf("got (%q, %d), want (INT4, 0)", qt, gs)
		}
	})
}
