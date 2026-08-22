package nemotron_h

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestPrepareMediaMarksImageExpansionCausal(t *testing.T) {
	var imageData bytes.Buffer
	if err := png.Encode(&imageData, image.NewNRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}

	m := &Model{
		VisionEncoder: &RadioVisionEncoder{},
		Projector:     &VisionProjector{},
		VisionConfig: &VisionConfig{
			PatchSize:        1,
			DownsampleFactor: 1,
			MinNumPatches:    1,
			MaxNumPatches:    1,
			MaxModelLen:      16,
			Std:              [3]float32{1, 1, 1},
		},
		imageStartTokenID: 10,
		imageTokenID:      11,
		imageEndTokenID:   12,
	}

	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{1}},
		{Kind: "image", Data: imageData.Bytes()},
		{Tokens: []int32{2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	if got, want := prepared.Items[0].Range, [2]int{1, 4}; got != want {
		t.Fatalf("range = %v, want %v", got, want)
	}
	if !prepared.Items[0].Causal {
		t.Fatal("image expansion must be causal in the text stack")
	}
}

func TestPrepareMediaReportsDisabledVision(t *testing.T) {
	m := &Model{visionErr: errors.New("unsupported RADIO version \"radio-v5\"")}

	prepared, err := m.PrepareMedia([]base.Segment{{Tokens: []int32{1, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := prepared.Tokens, []int32{1, 2}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("text tokens = %v, want %v", got, want)
	}

	_, err = m.PrepareMedia([]base.Segment{{Kind: "image", Data: []byte{1}}})
	if err == nil {
		t.Fatal("PrepareMedia unexpectedly accepted image input")
	}
	if got, want := err.Error(), "nemotron_h vision is unavailable: unsupported RADIO version \"radio-v5\""; got != want {
		t.Fatalf("PrepareMedia error = %q, want %q", got, want)
	}
}

// scatterMediaFixture builds a model and batch whose expansion covers
// positions 1..4: one item spliced at position 0, so imageStart sits at 0 and
// four feature rows follow. PatchSize and DownsampleFactor of 1 make
// visionTokenCount simply height*width.
func scatterMediaFixture() (*Model, *batch.Batch, *mlx.Array) {
	m := &Model{
		Config:       &Config{HiddenSize: 2},
		VisionConfig: &VisionConfig{PatchSize: 1, DownsampleFactor: 1},
	}

	features := mlx.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 4, 2)

	b := &batch.Batch{
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{6},
		Media: []batch.MediaItem{{
			Seq:      0,
			Pos:      0,
			Features: features,
			Opaque:   preparedImage{height: 1, width: 4},
		}},
	}

	return m, b, mlx.Zeros(mlx.DTypeFloat32, 1, 6, 2)
}

// The target forward's column 0 is the sequence position in SeqOffsets, so
// the feature rows land on the expansion's own positions 1..4.
func TestScatterMediaTargetForward(t *testing.T) {
	mlxtest.Setup(t)

	m, b, h := scatterMediaFixture()
	got := m.scatterMedia(h, b, 0)
	mlx.Eval(got)

	assertAllClose(t, "target-forward scatter", got.Floats(), []float32{
		0, 0, // position 0: imageStart, untouched
		1, 2, // positions 1..4: feature rows
		3, 4,
		5, 6,
		7, 8,
		0, 0, // position 5: past the expansion
	}, 1e-5)
}

// The MTP draft's slot S embeds the look-ahead token S+1, so its column 0
// holds the token one past its offset and every feature row shifts down by
// one column. Getting this wrong embeds raw placeholder tokens into the
// draft, which degrades acceptance silently rather than failing outright.
func TestScatterMediaDraftForwardShiftsByOne(t *testing.T) {
	mlxtest.Setup(t)

	m, b, h := scatterMediaFixture()
	got := m.scatterMedia(h, b, 1)
	mlx.Eval(got)

	assertAllClose(t, "draft-forward scatter", got.Floats(), []float32{
		1, 2, // shifted one column earlier than the target forward
		3, 4,
		5, 6,
		7, 8,
		0, 0,
		0, 0,
	}, 1e-5)
}

// A forward whose query range misses the expansion writes nothing, so a
// decode step past the image leaves the hidden rows alone.
func TestScatterMediaSkipsNonOverlappingQuery(t *testing.T) {
	mlxtest.Setup(t)

	m, b, h := scatterMediaFixture()
	b.SeqOffsets = []int32{8}
	b.SeqQueryLens = []int32{1}

	got := m.scatterMedia(h, b, 0)
	mlx.Eval(got)

	assertAllClose(t, "non-overlapping scatter", got.Floats(), []float32{
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
	}, 1e-5)
}
