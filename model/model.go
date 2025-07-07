package model

import (
	"context"
	"errors"
	"fmt"
	"image"
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
	"github.com/ollama/ollama/model/input"
)

var ErrNoVisionModel = errors.New("this model is missing data required for image input")

// Model implements a specific model architecture, defining the forward pass and any model-specific configuration
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)
	Backend() ml.Backend
	Config() config

	// ModelName returns the name of the model
	ModelName() string

	// SetBiasAdapters applies bias adapters from options
	SetBiasAdapters(options map[string]interface{}) error

	// Family returns the model family
	Family() string

	// Name returns the specific model name
	Name() string
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

// MultimodalProcessor represents a processor for multimodal inputs
type MultimodalProcessor interface {
	// ProcessImage processes an image for model consumption
	// Returns: normalized image data, image dimensions, and any error
	ProcessImage(image []byte) ([]float32, image.Point, error)

	// EncodeMultimodal encodes image data for model consumption
	// Returns: encoded multimodal data and any error
	EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error)

	// EncodeMultimodal encodes multimodal input for model consumption
	// Deprecated: Use EncodeMultimodal(ctx, data) instead
	EncodeMultimodal(input any) ([]int, error)

	// PostTokenize performs post-processing on tokenized inputs
	// Returns: processed inputs and any error
	PostTokenize(inputs []input.Input) ([]input.Input, error)

	// PostTokenize performs post-processing on tokenized inputs
	// Deprecated: Use PostTokenize([]input.Input) instead
	PostTokenize(tokens []int) ([]int, error)
}

// ModelName returns the default model name
func (m *Base) ModelName() string {
	return "base"
}

// SetBiasAdapters is a default implementation that does nothing
func (m *Base) SetBiasAdapters(options map[string]interface{}) error {
	// Default implementation does nothing
	return nil
}

// Family returns the default family name
func (m *Base) Family() string {
	return "base"
}

// Name returns the default name
func (m *Base) Name() string {
	return "base"
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

	arch := b.Config().Architecture()
	f, ok := models[arch]
	if !ok {
		return nil, fmt.Errorf("unsupported model architecture %q", arch)
	}

	m, err := f(b.Config())
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
	return getTextProcessor(meta.KV())
}

func getTextProcessor(kv fsggml.KV) (TextProcessor, error) {
	arch := kv.Architecture()
	f, ok := models[arch]
	if !ok {
		return nil, fmt.Errorf("unsupported model architecture %q", arch)
	}
	m, err := f(kv)
	if err != nil {
		return nil, err
	}
	tp, ok := m.(TextProcessor)
	if !ok {
		return nil, fmt.Errorf("%v is not a TextProcessor", m)
	}
	return tp, nil
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
				tagsCopy = append(tagsCopy, ParseTags(tag))
			}

			if tt == reflect.TypeOf((*Base)(nil)).Elem() {
				vv.Set(reflect.ValueOf(base))
			} else if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
				var fn func([]Tag) [][]string
				fn = func(tags []Tag) (values [][]string) {
					if len(tags) < 1 {
						return nil
					}

					values = [][]string{{tags[0].Name}}
					for _, alt := range tags[0].Alternate {
						values = append(values, []string{alt})
					}

					for i, value := range values {
						for _, rest := range fn(tags[1:]) {
							value = append(value, rest...)
						}

						values[i] = value
					}

					return values
				}

				names := fn(tagsCopy)
				for _, name := range names {
					if tensor := base.Backend().Get(strings.Join(name, ".")); tensor != nil {
						slog.Log(context.TODO(), logutil.LevelTrace, "found tensor", "", tensor)
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
						setPointer(base, vvv, append(tagsCopy, Tag{Name: strconv.Itoa(i)}))
					} else {
						vvv.Set(populateFields(base, vvv, append(tagsCopy, Tag{Name: strconv.Itoa(i)})...))
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

	vv = vv.Elem()
	if v.IsNil() {
		vv = reflect.New(v.Type().Elem()).Elem()
	}

	if f := populateFields(base, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}

type Tag struct {
	Name      string
	Alternate []string
}

func ParseTags(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.Name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok {
				tag.Alternate = append(tag.Alternate, value)
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

func Forward(ctx ml.Context, m Model, inputs []int32, batch input.Batch) (ml.Tensor, error) {
	if len(batch.Positions) != len(batch.Sequences) {
		return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(batch.Positions), len(batch.Sequences))
	}

	if len(batch.Positions) < 1 {
		return nil, errors.New("batch size cannot be less than 1")
	}

	batch.Inputs = ctx.Input().FromIntSlice(inputs, len(inputs))

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

	ctx.Forward(t).Compute(t)

	return t, nil
}
