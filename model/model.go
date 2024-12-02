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

type Cache struct {
	cache.Cache
	cache.Options
}

func (c Cache) Sub(i int) Cache {
	if c.Cache != nil {
		return Cache{
			Cache:   c.Cache.Sub(i),
			Options: c.Options,
		}
	}

	return c
}

func (c Cache) Put(ctx ml.Context, key, value ml.Tensor, opts cache.Options) (ml.Tensor, ml.Tensor) {
	if c.Cache != nil {
		return c.Cache.Put(ctx, key, value, opts)
	}

	return key, value
}

type Options struct {
	inputs []int32

	Offset int

	Images []image.Image

	Cache
}

func (opts Options) Inputs() []int32 {
	return opts.inputs[opts.Offset:]
}

func (opts Options) Positions() []int32 {
	positions := make([]int32, len(opts.inputs)-opts.Offset)
	for i := range positions {
		positions[i] = int32(opts.Offset + i)
	}

	return positions
}

type OptionsFunc func(Model, *Options)

func WithInputIDs(ids []int32) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.inputs = ids
	}
}

func WithOffset(offset int) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Offset = offset
		opts.Cache.Position = offset
	}
}

func WithImage(img image.Image) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Images = append(opts.Images, img)
	}
}

func WithCache(c cache.Cache) OptionsFunc {
	return func(m Model, opts *Options) {
		opts.Cache = Cache{
			Cache: c,
			Options: cache.Options{
				Position: opts.Offset,
			},
		}
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

func Forward(m Model, optsFuncs ...OptionsFunc) (ml.Tensor, error) {
	var opts Options
	for _, optsFunc := range optsFuncs {
		optsFunc(m, &opts)
	}

	ctx := m.Backend().NewContext()
	t, err := m.Forward(ctx, opts)
	if err != nil {
		return nil, err
	}
	defer ctx.Close()

	return ctx.Compute(t), nil
}
