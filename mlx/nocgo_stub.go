//go:build !cgo

// Package mlx wraps the MLX C API. All real entry points live in cgo
// files; this file provides an equivalent exported surface for builds
// without cgo (e.g. `go build` on Windows with no C toolchain) so the
// tree still compiles. Every stub panics at run time: without cgo there
// is no MLX engine to execute against.
package mlx

import (
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"math"
)

var _ = iter.Seq[struct{}](nil)
var _ = math.MaxInt32
var _ = fmt.Stringer(nil)
var _ = errors.New
var _ = slog.Level(0)

type Array struct{}
type Device struct{}
type Embedding struct{}
type LayerNorm struct{}
type Linear struct{}
type RMSNorm struct{}
type SafetensorsFile struct{}
type Scope struct{}
type Stream struct{}
type Byte int
type DType int
type GibiByte int
type KibiByte int
type MebiByte int
type TebiByte int
type Memory struct{}

type CompileFunc func(inputs ...*Array) []*Array
type CompileOption func(*compileConfig)
type scalarTypes interface {
	~bool | ~int | ~float32 | ~float64 | ~complex64
}
type arrayTypes interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 | ~complex64
}
type compileConfig struct {
	shapeless bool
}
type slice struct {
	args []int
}

func Slice(args ...int) slice { return slice{args: args} }

const End = math.MaxInt32
const Nvfp4MaxProduct = 448 * 6
const (
	DTypeBool DType = iota
	DTypeUint8
	DTypeUint16
	DTypeUint32
	DTypeUint64
	DTypeInt8
	DTypeInt16
	DTypeInt32
	DTypeInt64
	DTypeFloat16
	DTypeFloat32
	DTypeFloat64
	DTypeBFloat16
	DTypeComplex64
)

var GELU func(*Array) *Array
var GELUApprox func(*Array) *Array
var GeGLU func(*Array, *Array) *Array
var LogitSoftcap func(*Array, *Array) *Array
var ReLUSquared func(*Array) *Array
var SiLU func(*Array) *Array
var SoftplusF32 func(*Array) *Array
var SwiGLU func(*Array, *Array) *Array

func ActiveMemory() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func AsyncEval(outputs ...*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func CUDAIsAvailable() bool { return false }
func CacheMemory() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func CheckInit() error {
	return errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func ClearCache() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Compile1(name string, fn func(*Array) *Array, opts ...CompileOption) func(*Array) *Array {
	return func(a *Array) *Array {
		panic("github.com/ollama/ollama/mlx requires cgo")
	}
}
func Compile2(name string, fn func(*Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array) *Array {
	return func(a, b *Array) *Array {
		panic("github.com/ollama/ollama/mlx requires cgo")
	}
}
func Compile3(name string, fn func(*Array, *Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array, *Array) *Array {
	return func(a, b, c *Array) *Array {
		panic("github.com/ollama/ollama/mlx requires cgo")
	}
}
func DisableCompile() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func EnableCompile() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Eval(outputs ...*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func GPUIsAvailable() bool { return false }
func GatedDelta(packed, ba, dtBias, aExp, state, mask *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Load(path string) iter.Seq2[string, *Array] {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func LoadedLibraryPath() (string, error) {
	return "", errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func Mamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias, mask *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func MaxRecommendedWorkingSetSize() (int, error) {
	return 0, errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func MetalIsAvailable() bool { return false }
func PeakMemory() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func PrettyBytes(n int) fmt.Stringer {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ResetPeakMemory() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SaveSafetensors(path string, arrays map[string]*Array) error {
	return errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func SaveSafetensorsWithMetadata(path string, arrays map[string]*Array, metadata map[string]string) error {
	return errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func Scoped(fn func()) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SetDefaultDeviceGPU() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SetWiredLimit(limit int) (int, error) {
	return 0, errors.New("github.com/ollama/ollama/mlx requires cgo")
}
func Version() string { return "" }
func Add(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func AddMM(c, a, b *Array, alpha, beta float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func AddScalar(a *Array, s float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Arange(start, stop, step float64, dtype DType) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Argpartition(a *Array, kth int, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Argsort(a *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Bernoulli(p *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func BernoulliWithKey(p *Array, key *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func BroadcastTo(a *Array, shape ...int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Clamp(a *Array, minVal, maxVal float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Clip(a, aMin, aMax *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Collect(v any) []*Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Concatenate(arrays []*Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Contiguous(a *Array, allowColMajor bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Conv1d(x, weight *Array, bias *Array, stride, padding, dilation, groups int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Conv2d(x, weight *Array, strideH, strideW, padH, padW, dilationH, dilationW, groups int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Cos(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func DepthwiseConv1d(x, weight *Array, bias *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func DepthwiseConvSiLU(x, w, bias *Array, outLen int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string, globalScale *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Div(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func DivScalar(a *Array, s float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Erf(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Exp(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ExpandDims(a *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func FastScaledDotProductAttention(q, k, v *Array, scale float32, mode string, mask *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Flatten(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func FloorDivideScalar(a *Array, s int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func FromFP8(x *Array, dtype DType) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func FromValue[T scalarTypes](t T) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func FromValues[S ~[]E, E arrayTypes](s S, shape ...int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func GLU(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func GatherMM(a, b *Array, lhsIndices, rhsIndices *Array, sortedIndices bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func GatherQMM(x, w, scales *Array, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, mode string, globalScale *Array, sortedIndices bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func LayerNormFn(x, weight, bias *Array, eps float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Log(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Logaddexp(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Matmul(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Maximum(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Mean(a *Array, axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Minimum(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Mul(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func MulScalar(a *Array, s float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Neg(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func New(name string) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func NewArrayInt32(data []int32, shape []int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func NewScalarArray(value float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Pad(a *Array, axes []int, lowPad, highPad []int, padValue *Array, mode string) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func PadConstant(a *Array, axes []int, lowPad, highPad []int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Quantize(w *Array, groupSize, bits int, mode string) (weights, scales, biases *Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string, globalScale *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func RMSNormFn(x, weight *Array, eps float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func RSqrt(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func RandomKey(seed uint64) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ReLU(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Reshape(a *Array, shape ...int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func RoPEWithBase(x *Array, dims int, traditional bool, base, scale float32, offsets *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func RoPEWithFreqs(x *Array, dims int, traditional bool, base, scale float32, offsets *Array, freqs *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ScopedArrays(fn func() []*Array) []*Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ScopedAsyncEval(fn func() []*Array) []*Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ScopedEval(fn func() []*Array) []*Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Sigmoid(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SigmoidRouter(gates, bias *Array) (origScores, negScores *Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Sin(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SliceStartStop(a *Array, start, stop []int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SoftmaxAxis(a *Array, axis int, precise bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Softplus(a *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Squeeze(a *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Stack(arrays []*Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Sub(a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Sum(a *Array, axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func SwiGLUScaled(gate, gateScale, up, upScale *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Take(a *Array, indices *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Tile(a *Array, reps []int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ToFP8(x *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Transpose(a *Array, axes ...int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Tri(n, m int32, k int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Where(condition, a, b *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Zeros(dtype DType, shape ...int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func ZerosF32(shape []int32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func Compile(name string, fn CompileFunc, opts ...CompileOption) CompileFunc {
	return func(inputs ...*Array) []*Array {
		panic("github.com/ollama/ollama/mlx requires cgo")
	}
}
func Shapeless() CompileOption {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func DefaultDevice() Device {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func LoadSafetensorsNative(path string) (*SafetensorsFile, error) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func NewScope() *Scope {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func DefaultStream() Stream {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Abs() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Add(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Addmm(a, b *Array, alpha, beta float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Argmax(axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) ArgpartitionAxis(kth int, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) ArgsortAxis(axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) AsStrided(shape []int, strides []int, offset int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) AsType(dtype DType) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) BitwiseAnd(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) BitwiseXor(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Categorical(axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) CategoricalWithKey(axis int, key *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Clone() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Concatenate(axis int, others ...*Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Cumsum(axis int, reverse, inclusive bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) DType() DType {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Dim(dim int) int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Dims() []int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Divide(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Equal(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) ExpandDims(axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Flatten(startAxis, endAxis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Float() float32 {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Floats() []float32 {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) FloorDivide(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) GatherMM(other, lhs, rhs *Array, sorted bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Greater(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Int() int32 {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Ints() []int32 {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Less(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) LessEqual(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) LogValue() slog.Value {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) LogsumexpAxis(axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Matmul(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) MaxAxis(axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Multiply(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Negative() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) NumBytes() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) NumDims() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Power(exponent *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) PutAlongAxis(indices, values *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Remainder(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Reshape(axes ...int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) RightShift(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Save(name string) error {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) ScatterAddAxis(indices, values *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Set(other *Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Sigmoid() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Sign() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Size() int {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Slice(slices ...slice) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) SliceUpdate(other *Array, slices ...slice) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Sqrt() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Squeeze(axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) StackAxis(axis int, others ...*Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Subtract(other *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) SumAxis(axis int, keepDims bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) TakeAlongAxis(indices *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) TakeAxis(indices *Array, axis int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Tanh() *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *Array) Transpose(axes ...int) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (b *Byte) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *DType) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (t *DType) UnmarshalJSON(b []byte) error {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (d *Device) LogValue() slog.Value {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (e *Embedding) AsLinear() Linear {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (e *Embedding) Forward(indices *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (b *GibiByte) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (b *KibiByte) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (r *LayerNorm) Forward(x *Array, eps float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (m *Linear) Forward(x *Array) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (m *Linear) Gather(x, lhs, rhs *Array, sorted bool) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (b *MebiByte) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (r *RMSNorm) Forward(x *Array, eps float32) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *SafetensorsFile) Free() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *SafetensorsFile) Get(name string) *Array {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *SafetensorsFile) GetMetadata(key string) string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *Scope) Attach(arrays ...*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *Scope) Close() {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *Scope) Detach(arrays ...*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *Scope) Discard(arrays ...*Array) {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (s *Stream) LogValue() slog.Value {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
func (b *TebiByte) String() string {
	panic("github.com/ollama/ollama/mlx requires cgo")
}
