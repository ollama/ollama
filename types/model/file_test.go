package model

import (
	"io"
	"strings"
	"testing"
)

// setup is a helper function to set up the test environment.
func setup(t *testing.T, models map[Name]map[string]io.Reader) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	for m, s := range models {
		f, err := Create(m)
		if err != nil {
			t.Fatal(err)
		}

		for n, r := range s {
			w, err := f.Create(n)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := io.Copy(w, r); err != nil {
				t.Fatal(err)
			}
		}

		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestOpen(t *testing.T) {
	setup(t, map[Name]map[string]io.Reader{
		ParseName("namespace/model"): {
			"./.": strings.NewReader(`{"key":"value"}`),
		},
		ParseName("namespace/model:8b"): {
			"./.": strings.NewReader(`{"foo":"bar"}`),
		},
		ParseName("another/model"): {
			"./.": strings.NewReader(`{"another":"config"}`),
		},
	})

	f, err := Open(ParseName("namespace/model"))
	if err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{"./."} {
		r, err := f.Open(name)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := io.ReadAll(r); err != nil {
			t.Fatal(err)
		}

		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	t.Run("does not exist", func(t *testing.T) {
		if _, err := Open(ParseName("namespace/unknown")); err == nil {
			t.Error("expected error for unknown model")
		}
	})

	t.Run("write", func(t *testing.T) {
		f, err := Open(ParseName("namespace/model"))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		if _, err := f.Create("new-blob"); err == nil {
			t.Error("expected error creating blob in read-only mode")
		}
	})
}
