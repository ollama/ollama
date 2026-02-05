package base

import (
	"encoding/json"
	"errors"
	"log/slog"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Model interface {
	// Forward performs a forward pass through the model.
	Forward(inputs *mlx.Tensor, cache []cache.Cache) *mlx.Tensor

	// NumLayers returns the number of layers in the model.
	// This is used to initialize caches.
	// TODO: consider moving cache initialization into the model itself.
	NumLayers() int
}

type TextGeneration interface {
	Model
	Unembed(*mlx.Tensor) *mlx.Tensor
}

func Weights(m Model) (map[string]*mlx.Tensor, []func(*model.Root) error) {
	mapping := make(map[string]*mlx.Tensor)
	var afterLoadFuncs []func(*model.Root) error
	var fn func(v reflect.Value, tags []string)
	fn = func(v reflect.Value, tags []string) {
		t := v.Type()

		if method := v.Addr().MethodByName("AfterLoad"); method.IsValid() {
			var afterLoadFunc func(*model.Root) error
			reflect.ValueOf(&afterLoadFunc).Elem().Set(method)
			afterLoadFuncs = append(afterLoadFuncs, afterLoadFunc)
		}

		for _, field := range reflect.VisibleFields(t) {
			if field.IsExported() {
				tt, vv := field.Type, v.FieldByIndex(field.Index)

				// create local copy so tags are not modified between fields
				tags := tags
				if tag := field.Tag.Get("weight"); tag != "" {
					// TODO: use model.Tag
					tags = append(tags, tag)
				}

				if tt == reflect.TypeOf((*mlx.Tensor)(nil)).Elem() {
					name := strings.Join(tags, ".")
					mapping[name] =  vv.Addr().Interface().(*mlx.Tensor)
					continue
				}

				switch tt.Kind() {
				case reflect.Interface:
					vv = vv.Elem()
					fallthrough
				case reflect.Pointer:
					vv = vv.Elem()
					fallthrough
				case reflect.Struct:
					fn(vv, tags)
				case reflect.Slice, reflect.Array:
					for i := range vv.Len() {
						fn(vv.Index(i), append(tags, strconv.Itoa(i)))
					}
				}
			}
		}
	}
	fn(reflect.ValueOf(m).Elem(), []string{})
	return mapping, afterLoadFuncs
}

var m = make(map[string]func(*model.Root) (Model, error))

func Register(name string, f func(*model.Root) (Model, error)) {
	if _, exists := m[name]; exists {
		panic("model already registered: " + name)
	}

	m[name] = f
}

func New(root *model.Root) (Model, error) {
	c, err := root.Open("config.json")
	if err != nil {
		return nil, err
	}
	defer c.Close()

	var config struct {
		Architectures []string `json:"architectures"`
	}

	if err := json.NewDecoder(c).Decode(&config); err != nil {
		return nil, err
	}

	slog.Info("Model architecture", "arch", config.Architectures[0])
	if f, exists := m[config.Architectures[0]]; exists {
		return f(root)
	}

	return nil, errors.New("unknown architecture")
}
