package server

import (
	"testing"

	"github.com/stretchr/testify/require"
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
}
