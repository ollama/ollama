package tokenizer

import "testing"

// A minimal BPE tokenizer whose tokenizer.json post-processor prepends <bos>,
// matching the Gemma family (where add_bos_token is absent from
// tokenizer_config.json but the post-processor specifies BOS).
const bosPostProcessorJSON = `{
	"model": {"type":"BPE","vocab":{"a":0,"b":1,"<bos>":2},"merges":[]},
	"added_tokens":[{"id":2,"content":"<bos>","special":true}],
	"post_processor":{
		"type":"TemplateProcessing",
		"single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
		"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}
	}
}`

func TestPostProcessorEnablesAddBOS(t *testing.T) {
	tok, err := LoadFromBytes([]byte(bosPostProcessorJSON))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !tok.AddBOS() {
		t.Fatal("AddBOS() = false, want true (post-processor prepends <bos>)")
	}
	if tok.BOS() != 2 {
		t.Fatalf("BOS() = %d, want 2 (resolved from post-processor special_tokens)", tok.BOS())
	}

	// BOS prepended when requested.
	if ids := tok.Encode("ab", true); len(ids) == 0 || ids[0] != 2 {
		t.Fatalf("Encode(addBOS=true) = %v, want a leading 2", ids)
	}
	// No BOS when not requested.
	if ids := tok.Encode("ab", false); len(ids) > 0 && ids[0] == 2 {
		t.Fatalf("Encode(addBOS=false) = %v, want no leading BOS", ids)
	}
	// Dedup: text already starting with <bos> must not get a second BOS — this
	// is what keeps the renderer path (which emits a literal <bos>) from
	// double-prepending once AddBOS() is true.
	ids := tok.Encode("<bos>ab", true)
	if len(ids) < 1 || ids[0] != 2 {
		t.Fatalf("Encode(\"<bos>ab\") = %v, want leading 2", ids)
	}
	if len(ids) >= 2 && ids[1] == 2 {
		t.Fatalf("Encode(\"<bos>ab\") = %v, double BOS was not deduped", ids)
	}
}

// An explicit add_bos_token in tokenizer_config.json takes precedence over the
// post-processor default.
func TestAddBOSTokenConfigOverridesPostProcessor(t *testing.T) {
	cfg := &TokenizerConfig{TokenizerConfigJSON: []byte(`{"add_bos_token": false}`)}
	tok, err := LoadFromBytesWithConfig([]byte(bosPostProcessorJSON), cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.AddBOS() {
		t.Fatal("AddBOS() = true, want false (explicit add_bos_token=false should win)")
	}
}

// A post-processor whose leading special token is NOT the configured bos_token
// must not be treated as BOS: encoding should not prepend it, and BOS should
// remain the configured token.
const nonBOSPostProcessorJSON = `{
	"model": {"type":"BPE","vocab":{"a":0,"b":1,"<bos>":2,"<other>":3},"merges":[]},
	"added_tokens":[
		{"id":2,"content":"<bos>","special":true},
		{"id":3,"content":"<other>","special":true}
	],
	"post_processor":{
		"type":"TemplateProcessing",
		"single":[{"SpecialToken":{"id":"<other>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
		"special_tokens":{"<other>":{"id":"<other>","ids":[3],"tokens":["<other>"]}}
	}
}`

// The real Gemma case: companion config names <bos> as bos_token and the
// post-processor's leading special token is also <bos> -> AddBOS is enabled.
func TestPostProcessorMatchesConfiguredBOS(t *testing.T) {
	cfg := &TokenizerConfig{TokenizerConfigJSON: []byte(`{"bos_token": "<bos>"}`)}
	tok, err := LoadFromBytesWithConfig([]byte(bosPostProcessorJSON), cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !tok.AddBOS() {
		t.Fatal("AddBOS() = false, want true (post-processor prepends the configured BOS)")
	}
	if tok.BOS() != 2 {
		t.Fatalf("BOS() = %d, want 2", tok.BOS())
	}
}

func TestPostProcessorLeadingNonBOSNotPrepended(t *testing.T) {
	cfg := &TokenizerConfig{TokenizerConfigJSON: []byte(`{"bos_token": "<bos>"}`)}
	tok, err := LoadFromBytesWithConfig([]byte(nonBOSPostProcessorJSON), cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.AddBOS() {
		t.Fatal("AddBOS() = true, want false (the post-processor's leading token is not the configured BOS)")
	}
	if tok.BOS() != 2 {
		t.Fatalf("BOS() = %d, want 2 (the configured bos_token, not the post-processor's leading token)", tok.BOS())
	}
	if ids := tok.Encode("ab", true); len(ids) > 0 && ids[0] == 3 {
		t.Fatalf("Encode prepended the non-BOS leading token: %v", ids)
	}
}
