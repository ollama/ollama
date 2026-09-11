//go:build mlx

package mlx

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../../build/_deps/mlx-c-src
#cgo LDFLAGS: -L${SRCDIR}/../../../../build/lib/ollama/ -lmlxc -lmlx
#cgo LDFLAGS: -framework Accelerate
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../../build/lib/ollama/
#include <stdlib.h>
#include "mlx/c/mlx.h"
static inline size_t stride(const mlx_array a, int i) {return mlx_array_strides(a)[i];}

extern void goStackTrace();
static void error_handler(const char *msg, void* data) {
	fprintf(stderr, "MLX error: %s\n", msg);
	goStackTrace();
	exit(-1); // TODO adjust so this can become a return code on the current thread instead of exit
}
static void set_error_handler() {mlx_set_error_handler(&error_handler, NULL, NULL);}
static void* mlx_array_data_float16_asvoid(const mlx_array a) {return (void*)mlx_array_data_float16(a);}
typedef const char cchar_t;
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/x/ml"
	"github.com/x448/float16"
)

func init() {
	ml.RegisterBackend("mlx", New)
	C.set_error_handler()
}

//export goStackTrace
func goStackTrace() {
	debug.PrintStack()
}

type SafetensorsIndexMetadata struct {
	TotalSize uint64 `json:"total_size"`
}
type SafetensorsIndex struct {
	Metadata  SafetensorsIndexMetadata `json:"metadata"`
	WeightMap map[string]string        `json:"weight_map"`
}

type Backend struct {
	meta    fs.Config
	tensors map[string]*Array
}

func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	// TODO assumes modelPath is actually a directory for now...
	kv, tokenizer, err := convert.LoadModelMetadata(os.DirFS(modelPath))
	if err != nil {
		return nil, fmt.Errorf("unable to load model: %w", err)
	}

	b := &Backend{
		meta: kv.KV(tokenizer),
	}

	err = b.LoadSafeTensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("safetensors load failed: %w", err)
	}
	return b, nil
}

func (b *Backend) LoadSafeTensors(dir string) error {
	if _, err := os.Stat(dir); err != nil {
		return fmt.Errorf("failed to stat dir: %w", err)
	}
	// other variations to try?
	stFilename := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(stFilename); err != nil {
		return fmt.Errorf("failed to stat %s: %w", stFilename, err)
	}

	fp, err := os.Open(stFilename)
	if err != nil {
		return fmt.Errorf("failed to open safetensor index: %s: %w", stFilename, err)
	}
	decoder := json.NewDecoder(fp)
	var index SafetensorsIndex
	if err := decoder.Decode(&index); err != nil {
		return fmt.Errorf("decode error: %s: %w", stFilename, err)
	}
	slog.Info("XXX parsed metadata", "size", index.Metadata.TotalSize, "weights", len(index.WeightMap))
	filenames := map[string]struct{}{}
	for _, filename := range index.WeightMap {
		filenames[filename] = struct{}{}
	}
	stream := C.mlx_default_cpu_stream_new()

	b.tensors = map[string]*Array{}

	for filename := range filenames {
		filepath := filepath.Join(dir, filename)
		if _, err := os.Stat(filepath); err != nil {
			return fmt.Errorf("failed to stat %s: %w", filepath, err)
		}
		slog.Info("Loading tensors from", "filename", filename)
		cFilename := C.CString(filepath)
		defer C.free(unsafe.Pointer(cFilename))
		data := C.mlx_map_string_to_array_new() // TODO is this needed or just var it?
		metadata := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_array_free(data)
		defer C.mlx_map_string_to_string_free(metadata)

		if C.mlx_load_safetensors(&data, &metadata, cFilename, stream) != 0 {
			// TODO with the current error handling, this will never happen
			return fmt.Errorf("load failed")
		}

		it := C.mlx_map_string_to_array_iterator_new(data)
		// 	defer C.mlx_array_free(shaped)
		// TODO confusing how memory management works with this...
		for {
			var key *C.cchar_t
			var value C.mlx_array
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}
			k := C.GoString((*C.char)(key))
			b.tensors[k] = &Array{
				name: k,
				a:    value,
			}
			// slog.Info("XXX read", "tensor", b.tensors[k], "type", b.tensors[k].TypeString())
		}
	}

	return nil
}

func (b *Backend) Get(name string) ml.Tensor {
	var t ml.Tensor
	var ok bool
	if t, ok = b.tensors[name]; !ok {
		// slog.Warn("unable to locate", "tensor", name)
		return nil
	}
	// slog.Info("Fetching", "tensor", name, "type", b.tensors[name].TypeString())
	return t
}

func (b *Backend) NewContext() ml.Context {
	// slog.Info("MLX.NewContext")
	return &Context{
		stream: C.mlx_default_gpu_stream_new(),
	}
}

func (b *Backend) Config() fs.Config {
	return b.meta
}

type Context struct {
	stream C.mlx_stream

	mu     sync.Mutex
	arrays []C.mlx_array // TODO should we do some bookkeeping to ensure none of these Arrays are still lingering?
}

func (c *Context) Close() {
	// C.mlx_synchronize(c.stream) // ???
	C.mlx_stream_free(c.stream)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, a := range c.arrays {
		slog.Info("XXX freeing", "array", a)
		C.mlx_array_free(a)
	}
}

func (c *Context) Compute(tensors ...ml.Tensor) {
	// TODO - for the zero tensor case this feels like it might not be correct...
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range tensors {
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
		t.(*Array).sync = sync
	}
	C.mlx_async_eval(vec)
}

func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	for _, t := range tensors {
		t.(*Array).sync = sync
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
	}
	C.mlx_async_eval(vec)
	return c
}

func (c *Context) Input() ml.Context {
	return c
}

// func (c *Context) Output() ml.Context {
// 	return c
// }

func (c *Context) Layer(_ int) ml.Context {
	return c
}

func (c *Context) RandomNormal(shape []int, dtype ml.DType, loc, scale float32, key ml.Tensor) ml.Tensor {
	var r C.mlx_array
	var k C.mlx_array
	if key != nil {
		k = key.(*Array).a
	}
	sh := make([]C.int, len(shape))
	for i := range shape {
		sh[i] = C.int(shape[i])
	}
	C.mlx_random_normal(
		&r,
		&sh[0],
		C.size_t(len(shape)),
		C.mlx_dtype(dtype),
		C.float(loc),
		C.float(scale),
		k,
		c.stream,
	)
	return newArray(c, r)
}

func (c *Context) CompareWith(filepath string, tensors map[string]ml.Tensor, abortOnError bool) (err error) {
	minCosine := float32(0.96) // TODO too low...
	fileTensors := map[string]*Array{}
	defer func() {
		if err != nil {
			for k, v := range tensors {
				fmt.Fprintln(os.Stderr, "input tensor "+k+"\n"+v.ToString())
				if fv, ok := fileTensors[k]; ok {
					fmt.Fprintln(os.Stderr, " file tensor "+k+"\n"+fv.ToString())
				} else {
					fmt.Fprintln(os.Stderr, " file tensor "+k+" missing!\n")
				}
			}
		}
		if abortOnError {
			if err != nil {
				panic(fmt.Sprintf("%s", err))
			}
		}
	}()
	if _, err = os.Stat(filepath); err != nil {
		filepath += ".safetensors"
		if _, err = os.Stat(filepath); err != nil {
			err = fmt.Errorf("failed to stat %s: %w", filepath, err)
			return
		}
		err = nil
	}
	// slog.Info("Loading tensors from", "filename", filepath)
	cFilename := C.CString(filepath)
	defer C.free(unsafe.Pointer(cFilename))
	data := C.mlx_map_string_to_array_new() // TODO is this needed or just var it?
	metadata := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_array_free(data)
	defer C.mlx_map_string_to_string_free(metadata)

	stream := C.mlx_default_cpu_stream_new()

	if C.mlx_load_safetensors(&data, &metadata, cFilename, stream) != 0 {
		// TODO with the current error handling, this will never happen
		err = fmt.Errorf("load failed")
		return
	}

	it := C.mlx_map_string_to_array_iterator_new(data)
	allTensors := []ml.Tensor{}
	for _, t := range tensors {
		allTensors = append(allTensors, t)
	}

	for {
		var key *C.cchar_t
		var value C.mlx_array
		defer C.mlx_array_free(value)
		if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
			break
		}
		k := C.GoString((*C.char)(key))
		var r C.mlx_array
		defer C.mlx_array_free(r)
		C.mlx_astype(
			&r,
			value,
			C.MLX_FLOAT32,
			stream,
		)

		fileTensors[k] = &Array{
			name: k,
			a:    r,
		}
		// slog.Info("XXX read", "tensor", t, "type", t.TypeString())
		allTensors = append(allTensors, fileTensors[k])
	}
	c.Forward(allTensors...)
	for k, t := range tensors {
		a, ok := fileTensors[k]
		if !ok {
			err = fmt.Errorf("tensor named %s not found in file", k)
			return
		}
		if !reflect.DeepEqual(a.Shape(), t.Shape()) {
			err = fmt.Errorf("mismatched shapes:  file: %v vs. input %v", a.Shape(), t.Shape())
			return
		}
		// slog.Info("XXX shapes match", "shape", t.Shape())
		// TODO handle int types...
		tDType := t.DType()
		if tDType != ml.DTypeFloat16 && tDType != ml.DTypeFloat32 {
			var r C.mlx_array
			defer C.mlx_array_free(r)
			C.mlx_astype(
				&r,
				t.(*Array).a,
				C.MLX_FLOAT32,
				stream,
			)
			t = &Array{
				a: r,
			}
			c.Forward(t)
		}

		af := a.Floats()
		tf := t.Floats()
		cos := cosineSimilarity(af, tf)
		diff := a.Sub(c, t)
		min := diff.Min(c, nil, true)
		max := diff.Max(c, nil, true)
		c.Forward(min, max)
		minf := min.Floats()
		maxf := max.Floats()
		if cos < minCosine {
			err = fmt.Errorf("%s shapes match, but not similar enough:  %v  min_difference=%v max_difference=%v", k, cos, minf, maxf)
			return
		}

		slog.Info("XXX tensors are similar", k, cos, "shape", t.Shape(), "min_difference", minf, "max_difference", maxf)
	}
	err = nil

	return
}

func dotProduct[V float32 | float64](v1, v2 []V) V {
	var result V = 0
	if len(v1) != len(v2) {
		return result
	}

	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}
	return result
}

func magnitude[V float32 | float64](v []V) V {
	var result V = 0
	for _, val := range v {
		result += val * val
	}
	return V(math.Sqrt(float64(result)))
}

func cosineSimilarity[V float32 | float64](v1, v2 []V) V {
	mag1 := magnitude(v1)
	mag2 := magnitude(v2)

	if mag1 == 0 || mag2 == 0 {
		return 0
	}

	return dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))
}

func euclideanDistance[V float32 | float64](v1, v2 []V) V {
	if len(v1) != len(v2) {
		return V(math.Inf(1))
	}

	var sum V = 0
	for i := 0; i < len(v1); i++ {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}

	return V(math.Sqrt(float64(sum)))
}

func manhattanDistance[V float32 | float64](v1, v2 []V) V {
	if len(v1) != len(v2) {
		return V(math.Inf(1))
	}

	var sum V = 0
	for i := 0; i < len(v1); i++ {
		sum += V(math.Abs(float64(v1[i] - v2[i])))
	}

	return sum
}

type Array struct {
	name string
	a    C.mlx_array
	c    *Context

	sync func()
}

func newArray(ctx *Context, a C.mlx_array) *Array {
	// TODO measure impact and if this slows things down, make it conditional on some debugging flag at load time
	var name string
	_, f, l, ok := runtime.Caller(2)
	if ok {
		name = fmt.Sprintf("%s:%d", f, l)
	}

	t := &Array{
		name: name,
		a:    a,
		c:    ctx,
	}
	// DEBUG memory allocation problems...
	// slog.Info("XXX Allocated", "array", t, "a", a)
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	ctx.arrays = append(ctx.arrays, a)
	return t
}

// FromFloats implements ml.Context.
func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	u16s := make([]float16.Float16, len(s))
	for i := range u16s {
		u16s[i] = float16.Fromfloat32(s[i])
	}
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&u16s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_FLOAT16,
		),
	)
}

func (a *Array) Floats() []float32 {
	if a.sync != nil {
		a.sync()
	}
	l := (int)(C.mlx_array_size(a.a))

	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BFLOAT16:
		panic("bfloat16 not yet implemented")
	case C.MLX_FLOAT16:
		data := C.mlx_array_data_float16_asvoid(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		u16s := unsafe.Slice((*uint16)(data), l)
		f32s := make([]float32, len(u16s))
		for i := range u16s {
			f32s[i] = float16.Frombits(u16s[i]).Float32()
		}
		return f32s
	case C.MLX_FLOAT32:
		data := C.mlx_array_data_float32(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		f32s := unsafe.Slice((*float32)(data), l)
		return f32s
	default:
		panic(fmt.Sprintf("unsupported dtype for Floats: %d", C.mlx_array_dtype(a.a)))
	}
}

// FromInts implements ml.Context.
func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	)
}

func (a *Array) Ints() []int32 {
	if a.sync != nil {
		a.sync()
	}
	l := (int)(C.mlx_array_size(a.a))

	switch C.mlx_array_dtype(a.a) {
	case C.MLX_INT32:
		data := C.mlx_array_data_int32(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		i32s := unsafe.Slice((*int32)(data), l)
		return i32s

		// TODO other types via conversion?
	default:
		panic(fmt.Sprintf("unsupported dtype for Ints: %d", C.mlx_array_dtype(a.a)))
	}
}

func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	sh := make([]C.int, len(shape))
	for i, s := range shape {
		sh[i] = (C.int)(s)
	}

	var r C.mlx_array
	C.mlx_zeros(
		&r,
		&sh[0],
		(C.size_t)(len(sh)),
		C.mlx_dtype(dtype),
		c.stream,
	)
	return newArray(c, r)
}

func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	// TODO more efficient impl?
	return c.Zeros(dtype, shape...)
}

func (a *Array) DType() ml.DType {
	return (ml.DType)(C.mlx_array_dtype(a.a))
}

func (a *Array) Dim(n int) int {
	return int(C.mlx_array_dim(a.a, C.int(n)))
}

func (a *Array) Stride(n int) int {
	return (int)(C.stride(a.a, (C.int)(n)))
}

func (c *Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	var r C.mlx_array
	C.mlx_arange(
		&r,
		C.double(start),
		C.double(stop),
		C.double(step),
		(C.mlx_dtype)(dtype),
		c.stream,
	)

	return newArray(c, r)
}

// Scale implements ml.Tensor.
func (a *Array) Scale(ctx ml.Context, s float64) ml.Tensor {
	scale := C.mlx_array_new_float(C.float(s))
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		scale,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) Softmax(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	C.mlx_softmax(
		&r,
		a.a,
		false, // TODO - precise?
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) SliceUpdate(ctx ml.Context, update ml.Tensor, start, stop, strides []int) ml.Tensor {
	cStart := make([]C.int, len(start))
	for i := range start {
		cStart[i] = C.int(start[i])
	}
	cStop := make([]C.int, len(stop))
	for i := range stop {
		cStop[i] = C.int(stop[i])
	}
	cStrides := make([]C.int, len(strides))
	for i := range strides {
		cStrides[i] = C.int(strides[i])
	}
	var r C.mlx_array
	C.mlx_slice_update(
		&r,
		a.a,
		update.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")
}

func (a *Array) SliceUpdateDynamic(ctx ml.Context, update, start ml.Tensor, axes []int) ml.Tensor {
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}

	var r C.mlx_array
	C.mlx_slice_update_dynamic(
		&r,
		a.a,
		update.(*Array).a,
		start.(*Array).a,
		(*C.int)(unsafe.Pointer(&cAxes[0])),
		C.size_t(len(cAxes)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")

}

func (a *Array) PutAlongAxis(ctx ml.Context, indicies, values ml.Tensor, axis int) ml.Tensor {
	var r C.mlx_array
	C.mlx_put_along_axis(
		&r,
		a.a,
		indicies.(*Array).a,
		values.(*Array).a,
		C.int(axis),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays = append(a.c.arrays[:i], a.c.arrays[i+1:]...)
			return a
		}
	}
	panic("unable to locate array in context")
}

func (a *Array) Scatter(ctx ml.Context, indicies []ml.Tensor, updates ml.Tensor, axes []int) ml.Tensor {

	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
	}
	indiciesVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(indiciesVec)
	for _, ind := range indicies {
		C.mlx_vector_array_append_value(indiciesVec, ind.(*Array).a)
	}

	var r C.mlx_array
	C.mlx_scatter(
		&r,
		a.a,
		indiciesVec,
		updates.(*Array).a,
		cAxes0,
		C.size_t(len(cAxes)),
		ctx.(*Context).stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	a.c.mu.Lock()
	defer a.c.mu.Unlock()
	for i := range a.c.arrays {
		if a.c.arrays[i] == a.a {
			C.mlx_array_free(a.a)
			a.a = r
			a.c.arrays[i] = r
			return a
		}
	}
	panic("unable to locate array in context")

}

func (a *Array) Copy(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	C.mlx_copy(
		&a2.(*Array).a,
		a.a,
		ctx.(*Context).stream,
	)
	// TODO - view?
	return newArray(ctx.(*Context), a2.(*Array).a)
}

func (a *Array) Add(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_add(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) Sub(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_subtract(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) Max(ctx ml.Context, axes []int, keepDims bool) ml.Tensor {
	var r C.mlx_array
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
		C.mlx_max_axes(
			&r,
			a.a,
			cAxes0,
			C.size_t(len(cAxes)),
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	} else {
		C.mlx_max(
			&r,
			a.a,
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)

	}

	return newArray(ctx.(*Context), r)
}

func (a *Array) Min(ctx ml.Context, axes []int, keepDims bool) ml.Tensor {
	var r C.mlx_array
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}
	var cAxes0 *C.int
	if len(cAxes) > 0 {
		cAxes0 = (*C.int)(unsafe.Pointer(&cAxes[0]))
		C.mlx_min_axes(
			&r,
			a.a,
			cAxes0,
			C.size_t(len(cAxes)),
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	} else {
		C.mlx_min(
			&r,
			a.a,
			C._Bool(keepDims),
			ctx.(*Context).stream,
		)
	}

	return newArray(ctx.(*Context), r)
}

func (a *Array) Matmul(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_matmul(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	// slog.Info("MLX.RMSNorm", "a", a, "w", w)
	var r C.mlx_array
	C.mlx_fast_rms_norm(
		&r,
		a.a,
		w.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_layer_norm(
		&r,
		a.a,
		w.(*Array).a,
		b.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}

func (t Array) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

// RoPE implements Rotary Positional Encoding
//
// dims (int) – The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.
// traditional (bool) – If set to True choose the traditional implementation which rotates consecutive dimensions.
// scale (float) – The scale used to scale the positions.
// offset (int) – The position offset to start at.  TODO MLX-C does not yet expose Offset as an Array
// WithBase (float, optional) – The base used to compute angular frequency for each dimension in the positional encodings. Exactly one of base and freqs must be None.
// WithFreqs (array, optional) – Optional frequencies to use with RoPE. If set, the base parameter must be None. Default: None.
func (a *Array) RoPE(ctx ml.Context, dims int, traditional bool, scale float32, offset int, options ...func(*ml.RoPEOptions)) ml.Tensor {
	opts := ml.RoPEOptions{}

	// Apply any provided options
	for _, option := range options {
		option(&opts)
	}
	var r C.mlx_array
	var base C.mlx_optional_float
	var freqs C.mlx_array

	if opts.Base != nil {
		base.value = C.float(*opts.Base)
		base.has_value = true
	}
	if opts.Freqs != nil {
		freqs = opts.Freqs.(*Array).a
	}
	C.mlx_fast_rope(
		&r,
		a.a,
		C.int(dims),
		C._Bool(traditional),
		base,
		C.float(scale),
		C.int(offset),
		freqs,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// A fast implementation of multi-head attention: O = softmax(Q @ K.T, dim=-1) @ V.
//
// Supports:
// - Multi-Head Attention
// - Grouped Query Attention
// - Multi-Query Attention
//
// Note:
// - The softmax operation is performed in float32 regardless of the input precision.
// - For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q.
//
// In the following the dimensions are given by:
// - B: The batch size.
// - N_q: The number of query heads.
// - N_kv: The number of key and value heads.
// - T_q: The number of queries per example.
// - T_kv: The number of keys and values per example.
// - D: The per-head dimension.
//
// Parameters:
// - [subject array] queries (array) – Queries with shape [B, N_q, T_q, D].
// - keys (array) – with shape [B, N_kv, T_kv, D].
// - values (array) – with shape [B, N_kv, T_kv, D].
// - scale (float) – Scale for queries (typically 1.0 / sqrt(q.shape(-1)).
// - mask (str or array, optional) – The mask to apply to the query-key scores.
//   The mask can be an array or a string indicating the mask type. The only supported string type is "causal".
//   If the mask is an array it can be a boolean or additive mask. The mask can have at most 4 dimensions and
//   must be broadcast-compatible with the shape [B, N, T_q, T_kv]. If an additive mask is given its type must
//   promote to the promoted type of q, k, and v.
// - sinks (array, optional) – An optional array of attention sinks. Default: None.

func (queries *Array) ScaledDotProductAttention(ctx ml.Context, keys, values ml.Tensor, scale float64, maskMode string, mask ml.Tensor, sinks ml.Tensor) ml.Tensor {
	var r C.mlx_array
	var s C.mlx_array
	if sinks != nil {
		s = sinks.(*Array).a
	}
	maskModeC := C.CString(maskMode)
	defer C.free(unsafe.Pointer(maskModeC))
	var maskArr C.mlx_array
	if mask != nil {
		maskArr = mask.(*Array).a
	}

	C.mlx_fast_scaled_dot_product_attention(
		&r,
		queries.a,
		keys.(*Array).a,
		values.(*Array).a,
		C.float(scale),
		maskModeC,
		maskArr,
		s,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) TakeAxes(ctx ml.Context, indicies ml.Tensor, axes int) ml.Tensor {
	var r C.mlx_array

	C.mlx_take_axis(&r, a.a, indicies.(*Array).a, C.int(axes), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)

}

// TODO not sure if we'll want this variation taking raw ints instead of a tensor...
// func (a *Array) TakeAxes(ctx ml.Context, axes int, indicies ...int) ml.Tensor {
// 	var i C.mlx_array
// 	var r C.mlx_array

// 	if indicies != nil {
// 		shape := []C.int{C.int(len(indicies))}
// 		cindicies := make([]int32, len(indicies))
// 		for i, v := range indicies {
// 			cindicies[i] = int32(v)
// 		}
// 		i = C.mlx_array_new_data(
// 			unsafe.Pointer(&cindicies[0]),
// 			&shape[0],
// 			C.int(len(shape)),
// 			C.MLX_INT32,
// 		)
// 	}
// 	C.mlx_take_axis(&r, a.a, i, C.int(axes), ctx.(*Context).stream)
// 	return newArray(ctx.(*Context), r)

// }

func (a *Array) GELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	// TODO precise vs fast, and compile
	// x * mx.sigmoid(1.702 * x)
	u16s := []float16.Float16{float16.Fromfloat32(1.702)}
	cshape := []C.int{1}
	f := C.mlx_array_new_data(unsafe.Pointer(&u16s[0]), &cshape[0], 1, C.MLX_FLOAT16)
	defer C.mlx_array_free(f)
	var r1, r2, r3 C.mlx_array
	C.mlx_multiply(&r1, a.a, f, ctx.(*Context).stream)
	defer C.mlx_array_free(r1)
	C.mlx_sigmoid(&r2, r1, ctx.(*Context).stream)
	defer C.mlx_array_free(r2)
	C.mlx_multiply(&r3, a.a, r2, ctx.(*Context).stream)

	if len(up) > 0 {
		var r4 C.mlx_array
		defer C.mlx_array_free(r3)
		C.mlx_multiply(&r4, r3, up[0].(*Array).a, ctx.(*Context).stream)
		return newArray(ctx.(*Context), r4)
	}

	return newArray(ctx.(*Context), r3)
}

// Create a view into the array with the given shape and strides.
//
// The resulting array will always be as if the provided array was row
// contiguous regardless of the provided arrays storage order and current
// strides.
//
// Note that this function should be used with caution as it changes the shape
// and strides of the array directly. This can lead to the resulting array
// pointing to invalid memory locations which can result into crashes.
//
// Parameters:
//   - shape (list(int), optional) – The shape of the resulting array. If None it defaults to a.shape().
//   - strides (list(int), optional) – The strides of the resulting array. If None it defaults to the
//     reverse exclusive cumulative product of a.shape().
//   - offset (int) – Skip that many elements from the beginning of the input array.
func (a *Array) AsStrided(ctx ml.Context, shape, strides []int, offset int) ml.Tensor {
	var r C.mlx_array
	sh := make([]C.int, len(shape))
	st := make([]C.int64_t, len(strides))
	var sh0 *C.int
	var st0 *C.int64_t
	for i, s := range shape {
		sh[i] = C.int(s)
	}
	for i, s := range strides {
		st[i] = C.int64_t(s)
	}
	if len(sh) > 0 {
		sh0 = (*C.int)(unsafe.Pointer(&sh[0]))
	}
	if len(st) > 0 {
		st0 = (*C.int64_t)(unsafe.Pointer(&st[0]))
	}

	C.mlx_as_strided(
		&r,
		a.a,
		sh0,
		C.size_t(len(sh)),
		st0,
		C.size_t(len(st)),
		C.size_t(offset),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)

}

func (a *Array) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	var r C.mlx_array
	C.mlx_reshape(&r, a.a, &cshape[0], C.size_t(len(cshape)), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

func (a *Array) Transpose(ctx ml.Context, shape ...int) ml.Tensor {
	ndim := min(C.mlx_array_ndim(a.a), C.size_t(len(shape)))
	var r C.mlx_array
	sh := make([]C.int, ndim)
	for i := range ndim {
		sh[i] = (C.int)(shape[i])
		if int(sh[i]) >= int(ndim) {
			slog.Error("Permute error", "tensor", a, "shape", shape)
			panic("invalid pemute call")
		}
	}
	if len(sh) > 0 {
		C.mlx_transpose_axes(
			&r,
			a.a,
			&sh[0],
			ndim,
			ctx.(*Context).stream,
		)
	} else {
		C.mlx_transpose(
			&r,
			a.a,
			ctx.(*Context).stream,
		)
	}
	return newArray(ctx.(*Context), r)
}

func (a *Array) Contiguous(ctx ml.Context, allowColMajor bool) ml.Tensor {
	var r C.mlx_array
	C.mlx_contiguous(
		&r,
		a.a,
		(C._Bool)(allowColMajor),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Conv2D implements ml.Tensor.
// GGML API
// 	input: [N, IC, IH, IW]
// 	weight: [OC，IC, KH, KW]
// 	result: [N, OC, OH, OW]
//
// MLX:
//  input: (N, KH, KW, C_in)
//  weight: (C_out, IH, IW, C_in)
//  result: XXX

func (input *Array) Conv2D(ctx ml.Context, weight ml.Tensor, stride0, stride1, padding0, padding1, dilation0, dilation1, groups int) ml.Tensor {
	var r C.mlx_array
	C.mlx_conv2d(
		&r,
		input.a,
		weight.(*Array).a,
		C.int(stride0),
		C.int(stride1),
		C.int(padding0),
		C.int(padding1),
		C.int(dilation0),
		C.int(dilation1),
		C.int(groups),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (input *Array) Conv3D(ctx ml.Context, weight ml.Tensor, stride0, stride1, stride2, padding0, padding1, padding2, dilation0, dilation1, dilation2, groups int) ml.Tensor {
	var r C.mlx_array
	C.mlx_conv3d(
		&r,
		input.a,
		weight.(*Array).a,
		C.int(stride0),
		C.int(stride1),
		C.int(stride2),
		C.int(padding0),
		C.int(padding1),
		C.int(padding2),
		C.int(dilation0),
		C.int(dilation1),
		C.int(dilation2),
		C.int(groups),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

func (a *Array) ToString() string {
	str := C.mlx_string_new()
	C.mlx_array_tostring(&str, a.a)
	s := C.mlx_string_data(str)
	defer C.mlx_string_free(str)
	return C.GoString(s)
}

func (a *Array) LogValue() slog.Value {

	dims := int(C.mlx_array_ndim(a.a))
	strides := make([]int, dims)
	for i := range strides {
		strides[i] = int(C.stride(a.a, (C.int)(i)))
	}

	return slog.GroupValue(
		slog.String("name", a.name),
		slog.String("type", a.TypeString()),
		slog.Any("shape", a.Shape()),
		slog.Any("strides", strides),
		// slog.String("values", C.GoString(s)),
	)
}

func (a *Array) Shape() []int {
	shape := make([]int, C.mlx_array_ndim(a.a))
	for i := range shape {
		shape[i] = int(C.mlx_array_dim(a.a, C.int(i)))
	}

	return shape
}

func (a *Array) TypeString() string {
	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BOOL:
		return "bool"
	case C.MLX_UINT8:
		return "uint8"
	case C.MLX_UINT16:
		return "uint16"
	case C.MLX_UINT32:
		return "uint32"
	case C.MLX_UINT64:
		return "uint64"
	case C.MLX_INT8:
		return "int8"
	case C.MLX_INT16:
		return "int16"
	case C.MLX_INT32:
		return "int32"
	case C.MLX_INT64:
		return "int64"
	case C.MLX_FLOAT16:
		return "float16"
	case C.MLX_FLOAT32:
		return "float32"
	case C.MLX_BFLOAT16:
		return "bfloat16"
	case C.MLX_COMPLEX64:
		return "complex64"
	default:
		return "unknown"
	}
}
