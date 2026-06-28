package discover

import "testing"

func TestJetpackRunner(t *testing.T) {
	tests := []struct {
		name         string
		override     string
		tegraRelease string
		want         string
	}{
		{name: "no jetson", want: ""},

		// JETSON_JETPACK override.
		{name: "override jetpack 5", override: "5.1.2", want: "jetpack5"},
		{name: "override jetpack 6", override: "6", want: "jetpack6"},
		{name: "override jetpack 7 uses sbsa cuda", override: "7.2", want: ""},
		{name: "override future jetpack uses sbsa cuda", override: "8.0", want: ""},
		{name: "override jetpack 4 has no bundled runner", override: "4.6.1", want: ""},
		{name: "override invalid falls back to standard build", override: "not-a-version", want: ""},
		{
			name:         "override takes precedence over release file",
			override:     "6.1",
			tegraRelease: "# R38 (release), REVISION: 2.0",
			want:         "jetpack6",
		},

		// /etc/nv_tegra_release detection.
		{name: "l4t r35 is jetpack 5", tegraRelease: "# R35 (release), REVISION: 4.1, GCID: 0", want: "jetpack5"},
		{name: "l4t r36 is jetpack 6", tegraRelease: "# R36 (release), REVISION: 4.0, GCID: 0", want: "jetpack6"},
		{name: "l4t r38 (jetpack 7 / thor) uses sbsa cuda", tegraRelease: "# R38 (release), REVISION: 2.0, GCID: 0", want: ""},
		{name: "l4t r39 (jetpack 7 / orin) uses sbsa cuda", tegraRelease: "# R39 (release), REVISION: 2.0, GCID: 45755727", want: ""},
		{name: "future l4t uses sbsa cuda", tegraRelease: "# R40 (release), REVISION: 0.0", want: ""},
		{name: "older l4t is unrecognized", tegraRelease: "# R32 (release), REVISION: 7.1", want: ""},
		{name: "malformed release file", tegraRelease: "garbage without a version", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := jetpackRunner(tt.override, []byte(tt.tegraRelease)); got != tt.want {
				t.Errorf("jetpackRunner(%q, %q) = %q, want %q", tt.override, tt.tegraRelease, got, tt.want)
			}
		})
	}
}
