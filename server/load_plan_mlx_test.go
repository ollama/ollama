package server

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestMLXLoadPlanEvictionRequiresGPU(t *testing.T) {
	plan := mlxLoadPlan{
		modelContext: 32768,
		memory: loadMemoryAssessment{
			predictedLoad: 10,
			available:     0,
		},
	}

	require.Equal(t, 32768, plan.trainContext())

	require.Equal(t, loadedRunnerFitSkipped, plan.assessLoadedRunnerFit(true, 1))

	plan.hasGPU = true
	require.Equal(t, loadedRunnerNeedsEviction, plan.assessLoadedRunnerFit(true, 1))

	plan.image = true
	require.Equal(t, loadedRunnerFitSkipped, plan.assessLoadedRunnerFit(true, 1))
}

func TestMLXLoadPlanImageSkipsMemoryPreflight(t *testing.T) {
	req := &LlmRequest{
		model: &Model{
			ShortName: "image-model",
			Config: model.ConfigV2{
				Capabilities: []string{"image"},
			},
		},
		opts: api.Options{Runner: api.Runner{NumCtx: 8192}},
	}

	plan, err := newMLXLoadPlan(req, loadProposal{})
	require.NoError(t, err)
	require.True(t, plan.image)
	require.Equal(t, "image-model", plan.modelName)
	require.Equal(t, 8192, plan.softContextLength)
	require.Equal(t, loadedRunnerFitSkipped, plan.assessLoadedRunnerFit(true, 1))
}
