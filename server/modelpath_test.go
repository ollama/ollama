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
			"https://ollama.ai/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "ollama.ai",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"full path non-http",
			"file://ollama.ai/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "ollama.ai",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no protocol",
			"ollama.ai/ns/repo:tag",
			ModelPath{
				ProtocolScheme: DefaultProtocolScheme,
				Registry:       "ollama.ai",
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
			want := tc.want

			if got != want {
				t.Errorf("got: %q want: %q", got, tc.want)
			}
		})
	}
}
