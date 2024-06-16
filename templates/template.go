package templates

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"math"
	"sync"

	"github.com/agnivade/levenshtein"
)

//go:embed index.json
var indexBytes []byte

//go:embed *.gotmpl
var templatesFS embed.FS

var templatesOnce = sync.OnceValues(func() ([]*Template, error) {
	var templates []*Template
	if err := json.Unmarshal(indexBytes, &templates); err != nil {
		return nil, err
	}

	for _, t := range templates {
		bts, err := templatesFS.ReadFile(t.Name + ".gotmpl")
		if err != nil {
			return nil, err
		}

		// normalize line endings
		t.Bytes = bytes.ReplaceAll(bts, []byte("\r\n"), []byte("\n"))
	}

	return templates, nil
})

type Template struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Bytes []byte
}

func (t Template) Reader() io.Reader {
	return bytes.NewReader(t.Bytes)
}

func NamedTemplate(s string) (*Template, error) {
	templates, err := templatesOnce()
	if err != nil {
		return nil, err
	}

	var template *Template
	score := math.MaxInt
	for _, t := range templates {
		if s := levenshtein.ComputeDistance(s, t.Template); s < score {
			score = s
			template = t
		}
	}

	if score < 100 {
		return template, nil
	}

	return nil, errors.New("no matching template found")
}
