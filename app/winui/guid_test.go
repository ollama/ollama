package winui

import "testing"

func TestParameterizedIID(t *testing.T) {
	for _, tt := range []struct {
		signature string
		want      string
	}{
		// IReference<Int32>
		{"pinterface({61c17706-2d65-11e0-9ae8-d48564015472};i4)", "548cefbd-bc8a-5fa0-8df2-957440fc8bf4"},
		// IVector<String>
		{"pinterface({913337e9-11a1-4345-a3a2-4e7f956e222d};string)", "98b9acc1-4b56-532e-ac73-03d5291cca90"},
		// IAsyncOperation<Boolean>
		{"pinterface({9fc2b0bb-e446-44e2-aa61-9cab8f636af2};b1)", "cdb5efb3-5788-509d-9be1-71ccb8a3362a"},
	} {
		if got := parameterizedIID(tt.signature).String(); got != tt.want {
			t.Errorf("parameterizedIID(%q) = %s, want %s", tt.signature, got, tt.want)
		}
	}
}

func TestGUIDRoundTrip(t *testing.T) {
	const s = "4e0d09f5-4358-512c-a987-503b52848e95"
	if got := guid(s).String(); got != s {
		t.Fatalf("guid(%q).String() = %s", s, got)
	}
}
