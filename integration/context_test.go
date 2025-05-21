//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestLongInputContext(t *testing.T) {
	// Setting NUM_PARALLEL to 1 ensures the allocated context is exactly what
	// we asked for and there is nothing extra that we could spill over into
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")

	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "llama2",
		Prompt: "Oh, don’t speak to me of Austria. Perhaps I don’t understand things, but Austria never has wished, and does not wish, for war. She is betraying us! Russia alone must save Europe. Our gracious sovereign recognizes his high vocation and will be true to it. That is the one thing I have faith in! Our good and wonderful sovereign has to perform the noblest role on earth, and he is so virtuous and noble that God will not forsake him. He will fulfill his vocation and crush the hydra of revolution, which has become more terrible than ever in the person of this murderer and villain! We alone must avenge the blood of the just one.... Whom, I ask you, can we rely on?... England with her commercial spirit will not and cannot understand the Emperor Alexander’s loftiness of soul. She has refused to evacuate Malta. She wanted to find, and still seeks, some secret motive in our actions. What answer did Novosíltsev get? None. The English have not understood and cannot understand the self-abnegation of our Emperor who wants nothing for himself, but only desires the good of mankind. And what have they promised? Nothing! And what little they have promised they will not perform! Prussia has always declared that Buonaparte is invincible, and that all Europe is powerless before him.... And I don’t believe a word that Hardenburg says, or Haugwitz either. This famous Prussian neutrality is just a trap. I have faith only in God and the lofty destiny of our adored monarch. He will save Europe! What country is this referring to?",
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
			"num_ctx":     128,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("PullIfMissing failed: %v", err)
	}
	DoGenerate(ctx, t, client, req, []string{"russia", "germany", "france", "england", "austria", "prussia"}, 120*time.Second, 10*time.Second)
}

func TestContextExhaustion(t *testing.T) {
	// Setting NUM_PARALLEL to 1 ensures the allocated context is exactly what
	// we asked for and there is nothing extra that we could spill over into
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")

	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "llama2",
		Prompt: "Write me a story with a ton of emojis?",
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
			"num_ctx":     128,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("PullIfMissing failed: %v", err)
	}
	DoGenerate(ctx, t, client, req, []string{"once", "upon", "lived"}, 120*time.Second, 10*time.Second)
}
