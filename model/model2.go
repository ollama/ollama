package model

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
)

type Model2 struct {
	ml.Backend2
	Model
}

func New2(cfg *fs.Model, b ml.Backend2) (*Model2, error) {
	fn, ok := models[cfg.KV.Architecture()]
	if !ok {
		return nil, fmt.Errorf("unsupported model architecture %q", cfg.KV.Architecture())
	}

	m, err := fn(cfg.KV)
	if err != nil {
		return nil, err
	}

	// TODO: load tensors from the model into the backend
	v := reflect.ValueOf(m)
	v.Elem().Set(temp(b, cfg.Tensors, v.Elem()))

	if r, ok := b.Scheduler().(ml.Reserver); ok {
		// TODO: build a graph of the model and reserve the necessary resources
		r.Reserve()
	}

	return &Model2{b, m}, nil
}

func temp(b ml.Backend2, tensors map[string]fs.TensorReader, v reflect.Value, tags ...Tag) reflect.Value {
	t := v.Type()
	if t.Kind() != reflect.Struct {
		return v
	}

	allNil := true
	for i := range t.NumField() {
		tt := t.Field(i).Type
		vv := v.Field(i)
		if !vv.CanSet() {
			continue
		}

		tagsCopy := tags
		if s := t.Field(i).Tag.Get("gguf"); s != "" {
			tag := ParseTags(s)
			if tag.Root {
				tagsCopy = []Tag{tag}
			} else {
				tagsCopy = append(tagsCopy, ParseTags(s))
			}
		}

		switch {
		case tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem():
			var permute func([]Tag) [][]string
			permute = func(tags []Tag) (values [][]string) {
				if len(tags) < 1 {
					return nil
				}

				values = [][]string{{tags[0].Name}}
				for _, alt := range tags[0].Alternate {
					values = append(values, []string{alt})
				}

				for i, value := range values {
					for _, rest := range permute(tags[1:]) {
						value = append(value, rest...)
					}

					values[i] = value
				}

				return values
			}

			names := permute(tagsCopy)
			for _, name := range names {
				if tensor, ok := tensors[strings.Join(name, ".")]; ok {
					vv.Set(reflect.ValueOf(b.Get(tensor, tags[0].Device)))
					break
				}
			}
		case tt.Kind() == reflect.Pointer || tt.Kind() == reflect.Interface:
			setPointer2(b, tensors, vv, tagsCopy)
		case tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array:
			for i := vv.Len() - 1; i >= 0; i-- {
				vvv := vv.Index(i)
				if vvv.Kind() == reflect.Pointer || vvv.Kind() == reflect.Interface {
					setPointer2(b, tensors, vvv, append(tagsCopy, Tag{Name: strconv.Itoa(i)}))
				} else {
					vvv.Set(temp(b, tensors, vvv, append(tagsCopy, Tag{Name: strconv.Itoa(i)})...))
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

	return v
}

func setPointer2(b ml.Backend2, tensors map[string]fs.TensorReader, v reflect.Value, tags []Tag) {
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

	if f := temp(b, tensors, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}
