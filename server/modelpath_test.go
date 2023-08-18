package server

import "testing"

func TestParseModelPath(t *testing.T) {
	type input struct {
		name          string
		allowInsecure bool
	}

	tests := []struct {
		name    string
		args    input
		want    ModelPath
		wantErr error
	}{
		{
			"full path https",
			input{"https://example.com/ns/repo:tag", false},
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
			nil,
		},
		{
			"full path http without insecure",
			input{"http://example.com/ns/repo:tag", false},
			ModelPath{},
			ErrInsecureProtocol,
		},
		{
			"full path http with insecure",
			input{"http://example.com/ns/repo:tag", true},
			ModelPath{
				ProtocolScheme: "http",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
			nil,
		},
		{
			"full path invalid protocol",
			input{"file://example.com/ns/repo:tag", false},
			ModelPath{},
			ErrInvalidProtocol,
		},
		{
			"no protocol",
			input{"example.com/ns/repo:tag", false},
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
			nil,
		},
		{
			"no registry",
			input{"ns/repo:tag", false},
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
			nil,
		},
		{
			"no namespace",
			input{"repo:tag", false},
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            "tag",
			},
			nil,
		},
		{
			"no tag",
			input{"repo", false},
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            DefaultTag,
			},
			nil,
		},
		{
			"invalid image format",
			input{"example.com/a/b/c", false},
			ModelPath{},
			ErrInvalidImageFormat,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseModelPath(tc.args.name, tc.args.allowInsecure)

			if err != tc.wantErr {
				t.Errorf("got: %q want: %q", err, tc.wantErr)
			}

			if got != tc.want {
				t.Errorf("got: %q want: %q", got, tc.want)
			}
		})
	}
}
