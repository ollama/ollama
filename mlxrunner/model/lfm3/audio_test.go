package lfm3

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model/lfm3/lfm3audio"
)

func TestSpeechModelArraysAreCollectable(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		encoder := mlx.FromValue(float32(1))
		suppress := mlx.FromValue(float32(2))
		vl := &VLModel{Model: &Model{
			Audio:    &audioInput{Enc: &lfm3audio.Encoder{Layers: []*lfm3audio.Layer{{Conv: &lfm3audio.ShortConv{Weight: encoder}}}}},
			Suppress: suppress,
		}}
		got := mlx.Collect(vl)
		if len(got) != 2 || !slices.Contains(got, encoder) || !slices.Contains(got, suppress) {
			t.Fatalf("mlx.Collect = %v, want the encoder weight and suppress bias", got)
		}
	})
}
