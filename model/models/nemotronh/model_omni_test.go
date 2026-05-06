package nemotronh

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	backendggml "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model/input"
)

type fakeTensor struct {
	*backendggml.Tensor
	dims []int
}

func (t *fakeTensor) Dim(i int) int {
	return t.dims[i]
}

func setupTestContext(t *testing.T) ml.Context {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := fsggml.WriteGGUF(f, fsggml.KV{"general.architecture": "test"}, nil); err != nil {
		t.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}

	ctx := b.NewContext().Input()
	t.Cleanup(func() {
		ctx.Close()
		b.Close()
	})

	return ctx
}

func TestPostTokenizeImageSpans(t *testing.T) {
	m := &OmniModel{
		imageTokenID:    18,
		imageStartToken: 19,
		imageEndToken:   20,
	}

	makeChunk := func() input.Multimodal {
		return input.Multimodal{Tensor: &fakeTensor{dims: []int{2688, 256, 1, 1}}}
	}

	in := []*input.Input{
		{Token: 7},
		{
			Multimodal:     []input.Multimodal{makeChunk(), makeChunk()},
			MultimodalHash: 99,
		},
		{Token: 8},
	}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	if len(out) != 516 {
		t.Fatalf("len(out) = %d, want 516", len(out))
	}

	if out[0].Token != 7 {
		t.Fatalf("out[0].Token = %d, want 7", out[0].Token)
	}
	if out[1].Token != 19 {
		t.Fatalf("out[1].Token = %d, want 19", out[1].Token)
	}
	if out[1].SameBatch != 513 {
		t.Fatalf("out[1].SameBatch = %d, want 513", out[1].SameBatch)
	}
	if out[2].Token != 18 || len(out[2].Multimodal) != 1 || out[2].MultimodalHash != 99 || out[2].SameBatch != 0 {
		t.Fatalf("unexpected first image token: %+v", *out[2])
	}
	if out[258].Token != 18 || len(out[258].Multimodal) != 1 || out[258].MultimodalHash != 99 || out[258].SameBatch != 0 {
		t.Fatalf("unexpected second image token: %+v", *out[258])
	}
	if out[514].Token != 20 {
		t.Fatalf("out[514].Token = %d, want 20", out[514].Token)
	}
	if out[515].Token != 8 {
		t.Fatalf("out[515].Token = %d, want 8", out[515].Token)
	}
}

func TestProjectorPixelShuffleMatchesReferenceV2Order(t *testing.T) {
	ctx := setupTestContext(t)

	hidden := 2
	width := 4
	height := 2
	values := make([]float32, 0, hidden*width*height)
	for y := range height {
		for x := range width {
			for c := range hidden {
				values = append(values, float32(100*y+10*x+c))
			}
		}
	}

	got := pixelShuffleVisionOutputs(ctx, ctx.FromFloats(values, hidden, width*height), visionPatchGrid{
		Width:  width,
		Height: height,
	}, 2)
	ctx.Forward(got).Compute(got)

	want := []float32{
		0, 1, 10, 11, 100, 101, 110, 111,
		20, 21, 30, 31, 120, 121, 130, 131,
	}
	if got.Shape()[0] != 8 || got.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [8 2 1]", got.Shape())
	}
	gotValues := got.BackendGet()
	if len(gotValues) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(gotValues), len(want))
	}
	for i := range want {
		if gotValues[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, gotValues[i], want[i])
		}
	}
}

func TestPostTokenizeAudioSpans(t *testing.T) {
	m := &OmniModel{
		audioTokenID: 27,
	}

	in := []*input.Input{
		{Token: 7},
		{
			Multimodal: []input.Multimodal{{
				Tensor: &fakeTensor{dims: []int{2688, 13, 1, 1}},
				Data:   audioTag{},
			}},
			MultimodalHash: 99,
		},
		{Token: 8},
	}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	if len(out) != 15 {
		t.Fatalf("len(out) = %d, want 15", len(out))
	}
	if out[0].Token != 7 || out[14].Token != 8 {
		t.Fatalf("unexpected surrounding tokens: first=%d last=%d", out[0].Token, out[14].Token)
	}
	for i := 1; i <= 13; i++ {
		if out[i].Token != 27 {
			t.Fatalf("out[%d].Token = %d, want 27", i, out[i].Token)
		}
	}
	if len(out[1].Multimodal) != 1 || out[1].MultimodalHash != 99 {
		t.Fatalf("first audio token did not carry multimodal payload: %+v", *out[1])
	}
	if out[1].SameBatch != 12 {
		t.Fatalf("first audio token SameBatch = %d, want 12", out[1].SameBatch)
	}
	if len(out[2].Multimodal) != 0 {
		t.Fatalf("only the first audio token should carry multimodal payload: %+v", *out[2])
	}
}

func TestParakeetAudioPreprocessShapes(t *testing.T) {
	data := sineWAV(t, 16000, 440, 1.0)
	samples, err := decodeWAV(data, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(samples), 16000; got != want {
		t.Fatalf("sample count = %d, want %d", got, want)
	}

	mel, frames, validFrames, err := computeParakeetMelSpectrogram(samples, nil, defaultAudioOptions())
	if err != nil {
		t.Fatal(err)
	}
	if frames != 101 {
		t.Fatalf("frames = %d, want 101", frames)
	}
	if validFrames != 100 {
		t.Fatalf("validFrames = %d, want 100", validFrames)
	}
	if len(mel) != 101*128 {
		t.Fatalf("len(mel) = %d, want %d", len(mel), 101*128)
	}
	lastFrame := mel[100*128 : 101*128]
	if !slices.Equal(lastFrame, make([]float32, 128)) {
		t.Fatal("expected masked final frame to be zero")
	}
}

func TestParakeetAudioPreprocessMatchesIntegrationWAVReference(t *testing.T) {
	data := integrationAudioWAV(t)
	samples, err := decodeWAV(data, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(samples), 42083; got != want {
		t.Fatalf("sample count = %d, want %d", got, want)
	}

	mel, frames, validFrames, err := computeParakeetMelSpectrogram(samples, nil, defaultAudioOptions())
	if err != nil {
		t.Fatal(err)
	}
	if frames != 264 {
		t.Fatalf("frames = %d, want 264", frames)
	}
	if validFrames != 263 {
		t.Fatalf("validFrames = %d, want 263", validFrames)
	}
	if len(mel) != 264*128 {
		t.Fatalf("len(mel) = %d, want %d", len(mel), 264*128)
	}
	lastFrame := mel[263*128 : 264*128]
	if !slices.Equal(lastFrame, make([]float32, 128)) {
		t.Fatal("expected masked final frame to be zero")
	}

	// Reference values come from the ParakeetExtractor path used by vLLM:
	// pre-emphasis, torch.stft(center=True, pad_mode="constant"), Slaney mel
	// filters, log guard 2^-24, and per-mel normalization over valid frames.
	checks := map[[2]int]float32{
		{0, 0}:     -1.0855197,
		{0, 50}:    -0.93212974,
		{1, 10}:    -0.9735168,
		{2, 100}:   -0.6533053,
		{50, 0}:    2.2483668,
		{50, 127}:  -0.3828735,
		{100, 50}:  2.9742377,
		{262, 0}:   -0.9521758,
		{262, 127}: -0.4602786,
		{263, 50}:  0,
	}
	for pos, want := range checks {
		got := mel[pos[0]*128+pos[1]]
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Errorf("mel[%d,%d] = %v, want %v", pos[0], pos[1], got, want)
		}
	}
}

func integrationAudioWAV(t *testing.T) []byte {
	t.Helper()

	path := filepath.Join("..", "..", "..", "integration", "audio_test_data_test.go")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	const marker = "const audioEncodingPrompt = `"
	s := string(b)
	start := strings.Index(s, marker)
	if start < 0 {
		t.Fatal("audioEncodingPrompt marker not found")
	}
	start += len(marker)
	end := strings.Index(s[start:], "`")
	if end < 0 {
		t.Fatal("audioEncodingPrompt terminator not found")
	}

	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s[start : start+end]))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRelativeShiftParakeetMatchesReference(t *testing.T) {
	ctx := setupTestContext(t)

	seqLen := 3
	positionLen := 2*seqLen - 1
	values := make([]float32, seqLen*positionLen)
	for q := range seqLen {
		for p := range positionLen {
			values[q*positionLen+p] = float32(q*10 + p)
		}
	}

	x := ctx.FromFloats(values, positionLen, seqLen, 1)
	got := relativeShiftParakeet(ctx, x, seqLen, 1)
	ctx.Forward(got).Compute(got)

	want := []float32{
		2, 3, 4,
		11, 12, 13,
		20, 21, 22,
	}
	if !slices.Equal(got.BackendGet(), want) {
		t.Fatalf("relative shift mismatch:\n got %v\nwant %v", got.BackendGet(), want)
	}
}

func TestAudioDepthwiseConv2DMatchesReference(t *testing.T) {
	ctx := setupTestContext(t)

	freq, frames, channels := 4, 5, 2
	xValues := make([]float32, freq*frames*channels)
	for i := range xValues {
		xValues[i] = float32(i)/10 - 1
	}
	kernelValues := make([]float32, 3*3*channels)
	for i := range kernelValues {
		kernelValues[i] = float32(i)/7 - 1
	}

	x := ctx.FromFloats(xValues, freq, frames, channels, 1)
	kernel := ctx.FromFloats(kernelValues, 3, 3, 1, channels)
	bias := ctx.FromFloats([]float32{0.25, -0.5}, channels)
	got := audioDepthwiseConv2D(ctx, x, kernel, 2, 2, 1, 1, 1, 1).Add(ctx, bias.Reshape(ctx, 1, 1, -1))
	ctx.Forward(got).Compute(got)

	want := []float32{
		0.86428565, 1.3357141,
		1.2785715, 1.3642857,
		-0.5928571, -1.7499999,
		5.4000001, 8.8142853,
		10.514286, 16.042856,
		6.6857138, 9.8428574,
	}
	assertCloseSlice(t, got.BackendGet(), want, 1e-5)
}

func TestFlattenAudioSubsamplingOutputMatchesReference(t *testing.T) {
	ctx := setupTestContext(t)

	const (
		freq     = 2
		frames   = 3
		channels = 2
	)
	values := make([]float32, freq*frames*channels)
	for c := range channels {
		for t := range frames {
			for f := range freq {
				values[f+freq*(t+frames*c)] = float32(100*c + 10*t + f)
			}
		}
	}

	got := flattenAudioSubsamplingOutput(ctx, ctx.FromFloats(values, freq, frames, channels, 1))
	ctx.Forward(got).Compute(got)

	want := []float32{
		0, 1, 100, 101,
		10, 11, 110, 111,
		20, 21, 120, 121,
	}
	assertCloseSlice(t, got.BackendGet(), want, 0)
}

func TestAudioDepthwiseConv1DMatchesReference(t *testing.T) {
	ctx := setupTestContext(t)

	xValues := make([]float32, 2*5)
	for i := range xValues {
		xValues[i] = float32(i)/5 - 0.7
	}
	kernelValues := make([]float32, 3*2)
	for i := range kernelValues {
		kernelValues[i] = float32(i)/3 - 0.5
	}

	x := ctx.FromFloats(xValues, 2, 5)
	kernel := ctx.FromFloats(kernelValues, 3, 2)
	got := audioDepthwiseConv1DSame(ctx, x, kernel, 1)
	ctx.Forward(got).Compute(got)

	want := []float32{
		0.066666655, -0.5333333,
		0.41666666, 0.016666688,
		0.21666668, 1.0166667,
		0.01666667, 2.0166664,
		-0.40000004, 1.2666667,
	}
	assertCloseSlice(t, got.BackendGet(), want, 1e-5)
}

func TestAudioSelfAttentionMatchesReference(t *testing.T) {
	ctx := setupTestContext(t)

	const (
		hiddenSize = 4
		numHeads   = 2
		headDim    = 2
		seqLen     = 3
	)
	xValues := make([]float32, hiddenSize*seqLen)
	for i := range xValues {
		xValues[i] = float32(i)/10 - 0.5
	}

	identity := make([]float32, hiddenSize*hiddenSize)
	for i := range hiddenSize {
		identity[i*hiddenSize+i] = 1
	}
	linear := func() *nn.Linear {
		return &nn.Linear{Weight: ctx.FromFloats(identity, hiddenSize, hiddenSize)}
	}

	attn := &AudioSelfAttention{
		Query:       linear(),
		Key:         linear(),
		Value:       linear(),
		Output:      linear(),
		RelativeKey: linear(),
		BiasU:       ctx.FromFloats([]float32{0.1, -0.2, 0.3, -0.4}, headDim, numHeads),
		BiasV:       ctx.FromFloats([]float32{-0.05, 0.07, 0.11, -0.13}, headDim, numHeads),
	}
	got := attn.Forward(ctx, ctx.FromFloats(xValues, hiddenSize, seqLen), seqLen, &AudioOptions{
		hiddenSize: hiddenSize,
		numHeads:   numHeads,
		headDim:    headDim,
	})
	ctx.Forward(got).Compute(got)

	want := []float32{
		-0.08471569, 0.015284289, 0.05532019, 0.1553202,
		-0.09135241, 0.008647568, 0.11468154, 0.21468155,
		-0.019152153, 0.08084783, 0.1733382, 0.2733382,
	}
	assertCloseSlice(t, got.BackendGet(), want, 1e-5)
}

func assertCloseSlice(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("got[%d] = %v, want %v\nall got: %v", i, got[i], want[i], got)
		}
	}
}

func TestPackPatchesCHW(t *testing.T) {
	values := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
		100, 101, 102, 103,
		104, 105, 106, 107,
		108, 109, 110, 111,
		112, 113, 114, 115,
	}

	got := packVisionPatchesCHW(values, 4, 4, 2, 2)
	want := []float32{
		0, 1, 4, 5, 100, 101, 104, 105,
		2, 3, 6, 7, 102, 103, 106, 107,
		8, 9, 12, 13, 108, 109, 112, 113,
		10, 11, 14, 15, 110, 111, 114, 115,
	}

	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestResizePositionEmbeddingMatchesReferenceInterpolation(t *testing.T) {
	values := []float32{
		0, 10,
		20, 30,
	}
	got := resizePositionEmbedding(values, 1, 2, 2, 3, 3)
	want := []float32{
		0, 5, 10,
		10, 15, 20,
		20, 25, 30,
	}

	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDynamicImageProcessorMatchesReferencePatchBudget(t *testing.T) {
	p := ImageProcessor{
		imageSize:      512,
		patchSize:      16,
		numChannels:    3,
		minNumPatches:  1024,
		maxNumPatches:  13312,
		projectorScale: 2,
		imageMean:      [3]float32{0.48145466, 0.4578275, 0.40821073},
		imageStd:       [3]float32{0.26862954, 0.26130258, 0.27577711},
	}

	img := image.NewRGBA(image.Rect(0, 0, 400, 250))
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	for y := range height {
		for x := range width {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 128, A: 255})
		}
	}

	tiles, err := p.ProcessImage(img)
	if err != nil {
		t.Fatalf("ProcessImage() error = %v", err)
	}
	if got, want := len(tiles), 1; got != want {
		t.Fatalf("len(tiles) = %d, want %d", got, want)
	}
	if got, want := tiles[0].size, (image.Point{X: 672, Y: 416}); got != want {
		t.Fatalf("tile size = %v, want %v", got, want)
	}
	if got, want := len(tiles[0].data), 3*672*416; got != want {
		t.Fatalf("tile data len = %d, want %d", got, want)
	}
}

func sineWAV(t *testing.T, sampleRate int, frequency float64, seconds float64) []byte {
	t.Helper()

	samples := int(float64(sampleRate) * seconds)
	var pcm bytes.Buffer
	for i := range samples {
		v := int16(math.Sin(2*math.Pi*frequency*float64(i)/float64(sampleRate)) * 32767)
		if err := binary.Write(&pcm, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}

	var out bytes.Buffer
	out.WriteString("RIFF")
	if err := binary.Write(&out, binary.LittleEndian, uint32(36+pcm.Len())); err != nil {
		t.Fatal(err)
	}
	out.WriteString("WAVE")
	out.WriteString("fmt ")
	if err := binary.Write(&out, binary.LittleEndian, uint32(16)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint16(1)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint16(1)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint32(sampleRate)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint32(sampleRate*2)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint16(2)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&out, binary.LittleEndian, uint16(16)); err != nil {
		t.Fatal(err)
	}
	out.WriteString("data")
	if err := binary.Write(&out, binary.LittleEndian, uint32(pcm.Len())); err != nil {
		t.Fatal(err)
	}
	out.Write(pcm.Bytes())
	return out.Bytes()
}
