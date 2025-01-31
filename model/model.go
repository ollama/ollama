package model

import (
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

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
)

type Options struct {
	Inputs    []int32
	Positions []int32
	Sequences []int
	Outputs   []int32

	Images []image.Image
}

type config struct {
	Cache kvcache.Cache
}

type Base struct {
	b ml.Backend
	config
}

func (m *Base) Backend() ml.Backend {
	return m.b
}

func (m *Base) Config() config {
	return m.config
}

type Model interface {
	Forward(ml.Context, Options) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
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

	base := Base{b: b, config: m.Config()}

	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v))
	return m, nil
}

func populateFields(base Base, v reflect.Value, tags ...Tag) reflect.Value {
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

				if f := populateFields(base, vvv, tagsCopy...); f.CanAddr() {
					vv.Set(f.Addr())
				}
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vv.Index(i).Set(populateFields(base, vv.Index(i), append(tagsCopy, Tag{Name: strconv.Itoa(i)})...))
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

func Forward(ctx ml.Context, m Model, opts Options) (ml.Tensor, error) {
	if len(opts.Positions) != len(opts.Sequences) {
		return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(opts.Positions), len(opts.Sequences))
	}

	if len(opts.Positions) < 1 {
		return nil, errors.New("batch size cannot be less than 1")
	}

	cache := m.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, opts.Positions, opts.Sequences)
		if err != nil {
			return nil, err
		}
	}

	t, err := m.Forward(ctx, opts)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)
	return ctx.Compute(t), nil
}

func ArangeF32(start, end, step float32) []float32 {
	if step == 0 || start >= end {
		return nil
	}
	var res []float32
	for i := float32(start); i < end; i += step {
		res = append(res, i)
	}
	return res
}
