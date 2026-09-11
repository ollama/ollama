package discover

import "testing"

func TestSplitSyclVisibleDeviceList(t *testing.T) {
	type testcase struct {
		name string
		inp  string
		exp  []string
	}
	for _, tc := range []testcase{
		{name: "empty", inp: "", exp: nil},
		{name: "blank", inp: "   ", exp: nil},
		{name: "backend ids", inp: "level_zero:0;level_zero:1", exp: []string{"0", "1"}},
		{name: "bare ids", inp: "0;1", exp: []string{"0", "1"}},
		{name: "mixed spacing", inp: " level_zero:0 ; 1 ", exp: []string{"0", "1"}},
		{name: "trailing separator", inp: "level_zero:0;", exp: []string{"0"}},
		{name: "invalid id", inp: "level_zero:x", exp: nil},
		{name: "negative id", inp: "level_zero:-1", exp: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := splitSyclVisibleDeviceList(tc.inp)
			if len(got) != len(tc.exp) {
				t.Fatalf("expected %v, got %v", tc.exp, got)
			}
			for i := range got {
				if got[i] != tc.exp[i] {
					t.Fatalf("expected %v, got %v", tc.exp, got)
				}
			}
		})
	}
}

func TestInferLibrarySYCL(t *testing.T) {
	type testcase struct {
		name        string
		deviceName  string
		description string
		exp         string
	}
	for _, tc := range []testcase{
		{
			name:        "arc gpu",
			deviceName:  "SYCL0",
			description: "Intel(R) Arc(TM) B580 Graphics",
			exp:         "SYCL",
		},
		{
			name:        "case insensitive",
			deviceName:  "sycl0",
			description: "intel arc",
			exp:         "SYCL",
		},
		{
			name:        "vulkan still detected",
			deviceName:  "Vulkan0",
			description: "Intel(R) Arc(TM) B580 Graphics",
			exp:         "Vulkan",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := inferLibrary(tc.deviceName, tc.description); got != tc.exp {
				t.Fatalf("expected %q, got %q", tc.exp, got)
			}
		})
	}
}
