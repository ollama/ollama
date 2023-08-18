package server

import "testing"

func TestParseModelPath(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  ModelPath
	}{
		{
			"full path https",
			"https://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"full path non-http",
			"file://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no protocol",
			"example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no registry",
			"ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       DefaultRegistry,
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no namespace",
			"repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no tag",
			"repo",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            DefaultTag,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseModelPath(tc.input)
			if got != tc.want {
				t.Errorf("got: %q want: %q", got, tc.want)
			}
		})
	}
}
