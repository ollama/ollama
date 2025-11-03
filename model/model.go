package model

import (
	"errors"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"strings"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model/input"
)

var (
	ErrNoVisionModel        = errors.New("this model is missing data required for image input")
	ErrUnsupportedModel     = errors.New("model not supported")
	ErrUnsupportedTokenizer = errors.New("tokenizer not supported")
)

// Model implements a specific model architecture, defining the forward pass and any model-specific configuration
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
}

// MultimodalProcessor must be implemented by multimodal models.
type MultimodalProcessor interface {
	// EncodeMultimodal processes a single input (such as an image) and
	// generates an output (typically an embedding) that can be used by the model.
	//
	// The return value is one or more tensors, each with optional model-specific
	// opaque metadata. Typically, the tensors might be views into an embedding
	// with each view representing a chunk of data that can be processed independently
	// in different batches.
	//
	// The result may be cached by the runner.
	EncodeMultimodal(ml.Context, []byte) ([]input.Multimodal, error)

	// PostTokenize is called after tokenization to allow the model to edit the
	// input stream to correctly arrange multimodal elements.
	//
	// The input is a slice of tokens with the results of EncodeMultimodal interleaved
	// in the order that the user provided them. Each element of the slice will be
	// either a single token or single multimodal object.
	//
	// The model must ensure that inputs are stored according to how they will be
	// processed and stored in the cache. For example, Llava-style models should insert
	// placeholder tokens equal to the feature size of the corresponding image with
	// the image itself attached to and split across these tokens. When Forward is called
	// a partial subset of these tokens may be submitted according to the batch size.
	//
	// This function is also responsible for updating MultimodalHash for any Multimodal
	// that is modified to ensure that there is a unique hash value that accurately
	// represents the contents.
	PostTokenize([]*input.Input) ([]*input.Input, error)
}

// Base implements the common fields and methods for all models
type Base struct {
	b ml.Backend
	config
}

type config struct {
	Cache kvcache.Cache
}

// Backend returns the underlying backend that will run the model
func (m *Base) Backend() ml.Backend {
	return m.b
}

func (m *Base) Config() config {
	return m.config
}

var models = make(map[string]func(fs.Config) (Model, error))

// Register registers a model constructor for the given architecture
func Register(name string, f func(fs.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

// New initializes a new model instance with the provided configuration based on the metadata in the model file
func New(modelPath string, params ml.BackendParams) (Model, error) {
	b, err := ml.NewBackend(modelPath, params)
	if err != nil {
		return nil, err
	}

	m, err := modelForArch(b.Config())
	if err != nil {
		return nil, err
	}

	base := Base{b: b, config: m.Config()}
	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v.Elem()))
	return m, nil
}

func NewTextProcessor(s string) (TextProcessor, error) {
	r, err := os.Open(s)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	m, err := modelForArch(meta.KV())
	if err != nil {
		return nil, err
	}

	tp, ok := m.(TextProcessor)
	if !ok {
		return nil, ErrUnsupportedTokenizer
	}
	return tp, nil
}

func modelForArch(c fs.Config) (Model, error) {
	arch := c.Architecture()
	if pooling.Type(c.Uint("pooling_type")) != pooling.TypeNone {
		arch = arch + "_embed"
	}

	f, ok := models[arch]
	if !ok {
		return nil, ErrUnsupportedModel
	}

	return f(c)
}

func populateFields(base Base, v reflect.Value, tags ...Tag) reflect.Value {
	t := v.Type()

	if t.Kind() == reflect.Struct {
		allNil := true
		for i := range t.NumField() {
			tt := t.Field(i).Type
			vv := v.Field(i)
			if !vv.CanSet() {
				continue
			}

			// make a copy
			tagsCopy := tags
			if tag := t.Field(i).Tag.Get("gguf"); tag != "" {
				tagsCopy = append(tagsCopy, parseTag(tag))
			}

			if tt == reflect.TypeOf((*Base)(nil)).Elem() {
				vv.Set(reflect.ValueOf(base))
			} else if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
				var fn func([]Tag, string, string) [][]string
				fn = func(tags []Tag, prefix, suffix string) (fullNames [][]string) {
					if len(tags) > 0 {
						var names []string
						if tags[0].name != "" {
							for _, n := range append([]string{tags[0].name}, tags[0].alternatives...) {
								names = append(names, prefix+n+suffix)
							}
						}
						childNames := fn(tags[1:], tags[0].prefix, tags[0].suffix)
						if len(names) == 0 {
							// current tag has no name, use child names only
							fullNames = append(fullNames, childNames...)
						} else if len(childNames) == 0 {
							// current tag has names but no children, create branches for each name
							for _, name := range names {
								fullNames = append(fullNames, []string{name})
							}
						} else {
							// merge each name with each child
							for _, name := range names {
								for _, childName := range childNames {
									fullNames = append(fullNames, append([]string{name}, childName...))
								}
							}
						}
					}

					return fullNames
				}

				names := fn(tagsCopy, "", "")
				for _, name := range names {
					if tensor := base.Backend().Get(strings.Join(name, ".")); tensor != nil {
						logutil.Trace("found tensor", "", tensor)
						vv.Set(reflect.ValueOf(tensor))
						break
					}
				}
			} else if tt.Kind() == reflect.Pointer || tt.Kind() == reflect.Interface {
				setPointer(base, vv, tagsCopy)
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vvv := vv.Index(i)
					if vvv.Kind() == reflect.Pointer || vvv.Kind() == reflect.Interface {
						setPointer(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)}))
					} else {
						vvv.Set(populateFields(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)})...))
					}
				}
			}

			if !canNil(tt) || !vv.IsNil() {
				allNil = false
			}
		}

		if allNil {
			return reflect.Zero(t)
		}
	}

	return v
}

func setPointer(base Base, v reflect.Value, tags []Tag) {
	vv := v
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}

		vv = vv.Elem()
	}

	vv = reflect.Indirect(vv)
	if v.IsNil() {
		vv = reflect.New(v.Type().Elem()).Elem()
	}

	if f := populateFields(base, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}

type Tag struct {
	name,
	// prefix and suffix are applied to child tags
	prefix,
	suffix string
	alternatives []string
}

func parseTag(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok && tag.name == "" {
				// elevate alternative to primary if no primary given
				tag.name = value
				slog.Warn("gguf tag has alt: but no primary name", "tag", s)
			} else if ok {
				tag.alternatives = append(tag.alternatives, value)
			}
			if value, ok := strings.CutPrefix(part, "pre:"); ok {
				tag.prefix = value
			}
			if value, ok := strings.CutPrefix(part, "suf:"); ok {
				tag.suffix = value
			}
		}
	}

	return
}

func canNil(t reflect.Type) bool {
	return t.Kind() == reflect.Chan ||
		t.Kind() == reflect.Func ||
		t.Kind() == reflect.Interface ||
		t.Kind() == reflect.Map ||
		t.Kind() == reflect.Pointer ||
		t.Kind() == reflect.Slice
}

func Forward(ctx ml.Context, m Model, batch input.Batch) (ml.Tensor, error) {
	if len(batch.Positions) != len(batch.Sequences) {
		return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(batch.Positions), len(batch.Sequences))
	}

	if len(batch.Positions) < 1 {
		return nil, errors.New("batch size cannot be less than 1")
	}

	cache := m.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			return nil, err
		}
	}

	t, err := m.Forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)

	return t, nil
}
