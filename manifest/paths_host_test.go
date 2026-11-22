package manifest

import "testing"

// encodeV2ManifestHost must be reversible: escaping "%" first keeps a host
// that already contains an encoded sequence (e.g. "reg%3A5000") from
// colliding with the encoding of a different host ("reg:5000"). See #19834.
func TestV2ManifestHostRoundTrip(t *testing.T) {
	for _, host := range []string{
		"registry.ollama.ai",
		"reg:5000",
		"reg%3A5000",
		"a%b:1",
		"%%",
	} {
		encoded := encodeV2ManifestHost(host)
		if got := decodeV2ManifestHost(encoded); got != host {
			t.Errorf("round trip %q -> %q -> %q; want %q", host, encoded, got, host)
		}
	}

	if encodeV2ManifestHost("reg:5000") == encodeV2ManifestHost("reg%3A5000") {
		t.Error("two distinct hosts must not share the same encoded folder name")
	}

	if got := decodeV2ManifestHost("reg%3a5000"); got != "reg:5000" {
		t.Errorf("decodeV2ManifestHost should tolerate lowercase escapes: got %q", got)
	}
}
