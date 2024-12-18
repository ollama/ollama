package model

import (
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

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
)

type Options struct {
	inputs    []int32
	positions []int32
	outputs   []int32

	sequences []int

	Images []image.Image

	cache.Cache
}

func (opts Options) Inputs() []int32 {
	return opts.inputs
}

func (opts Options) Positions() []int32 {
	return opts.positions
}

func (opts Options) Outputs() []int32 {
	return opts.outputs
}

type OptionsFunc func(Model, *Options)

func WithInputIDs(ids []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.inputs = ids
	}
}

func WithPositions(pos []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.positions = pos
	}
}

func WithOutputs(outputs []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.outputs = outputs
	}
}

func WithSequences(seqs []int) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.sequences = seqs
	}
}

func WithImage(img image.Image) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Images = append(opts.Images, img)
	}
}

func WithCache(c cache.Cache) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Cache = c
	}
}

type Base struct {
	b ml.Backend
}

func (m *Base) Backend() ml.Backend {
	return m.b
}

type Model interface {
	Forward(ml.Context, Options) (ml.Tensor, error)

	Backend() ml.Backend
}

var models = make(map[string]func(ml.Config) (Model, error))

func Register(name string, f func(ml.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

func New(s string) (Model, error) {
	r, err := os.Open(s)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	b, err := ml.NewBackend(r)
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

	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(b, v))
	return m, nil
}

func populateFields(b ml.Backend, v reflect.Value, tags ...Tag) reflect.Value {
	var iface bool
	if v.Kind() == reflect.Interface {
		iface = true
		v = v.Elem()
	}

	t := v.Type()
	if t.Kind() == reflect.Pointer {
		t, v = t.Elem(), v.Elem()
	}

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
				vv.Set(reflect.ValueOf(Base{b: b}))
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
					if tensor := b.Get(strings.Join(name, ".")); tensor != nil {
						slog.Debug("found tensor", "", tensor)
						vv.Set(reflect.ValueOf(tensor))
						break
					}
				}
			} else if tt.Kind() == reflect.Pointer {
				vvv := vv.Elem()
				if vv.IsNil() {
					vvv = reflect.New(tt.Elem())
				}

				if f := populateFields(b, vvv, tagsCopy...); f.CanAddr() {
					vv.Set(f.Addr())
				}
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vv.Index(i).Set(populateFields(b, vv.Index(i), append(tagsCopy, Tag{Name: strconv.Itoa(i)})...))
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

	if iface {
		return v.Addr()
	}

	return v
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

func Forward(ctx ml.Context, m Model, optsFuncs ...OptionsFunc) (ml.Tensor, error) {
	var opts Options
	for _, optsFunc := range optsFuncs {
		optsFunc(m, &opts)
	}

	err := opts.Cache.StartForward(ctx, opts.sequences)
	if err != nil {
		return nil, err
	}

	t, err := m.Forward(ctx, opts)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)
	return ctx.Compute(t), nil
}
