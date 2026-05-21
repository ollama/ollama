package manifest

import "testing"

func TestTouchLayerDigestWithInvalidDigest(t *testing.T) {
	if err := touchLayerDigest("not-a-digest"); err == nil {
		t.Fatal("expected invalid digest error")
	}
}
