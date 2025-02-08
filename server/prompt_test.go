package server

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func TestChatPrompt(t *testing.T) {
	type expect struct {
		prompt        string
		images        [][]byte
		aspectRatioID int
		error         error
	}

	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	visionModel := Model{Template: tmpl, ProjectorPaths: []string{"vision"}}
	mllamaModel := Model{Template: tmpl, ProjectorPaths: []string{"vision"}, Config: ConfigV2{ModelFamilies: []string{"mllama"}}}

	createImg := func(width, height int) ([]byte, error) {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		var buf bytes.Buffer

		if err := png.Encode(&buf, img); err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	}

	imgBuf, err := createImg(5, 5)
	if err != nil {
		t.Fatal(err)
	}

	imgBuf2, err := createImg(6, 6)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name  string
		model Model
		limit int
		msgs  []api.Message
		expect
	}{
		{
			name:  "messages",
			model: visionModel,
			limit: 64,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:  "truncate messages",
			model: visionModel,
			limit: 1,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:  "truncate messages with image",
			model: visionModel,
			limit: 64,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("something")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
				},
			},
		},
		{
			name:  "truncate messages with images",
			model: visionModel,
			limit: 64,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:  "messages with images",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]You're a test, Harry! I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:  "message with image tag",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry! [img]", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "You're a test, Harry! [img-0] I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:  "messages with interleaved images",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry!\n\n[img-0]\n\n[img-1] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:  "truncate message with interleaved images",
			model: visionModel,
			limit: 1024,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "[img-0] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:  "message with system prompt",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You are the Test Who Lived. You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:  "out of order system",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? You are the Test Who Lived. A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:  "multiple images same prompt",
			model: visionModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "Compare these two pictures of hotdogs", Images: []api.ImageData{[]byte("one hotdog"), []byte("two hotdogs")}},
			},
			expect: expect{
				prompt: "[img-0][img-1]Compare these two pictures of hotdogs ",
				images: [][]byte{[]byte("one hotdog"), []byte("two hotdogs")},
			},
		},
		{
			name:  "messages with mllama (no images)",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:  "messages with mllama single prompt",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "How many hotdogs are in this image?", Images: []api.ImageData{imgBuf}},
			},
			expect: expect{
				prompt:        "[img-0]<|image|>How many hotdogs are in this image? ",
				images:        [][]byte{imgBuf},
				aspectRatioID: 1,
			},
		},
		{
			name:  "messages with mllama",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{imgBuf}},
			},
			expect: expect{
				prompt:        "You're a test, Harry! I-I'm a what? [img-0]<|image|>A test. And a thumping good one at that, I'd wager. ",
				images:        [][]byte{imgBuf},
				aspectRatioID: 1,
			},
		},
		{
			name:  "multiple messages with mllama",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{imgBuf}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{imgBuf2}},
			},
			expect: expect{
				prompt:        "[img-0]<|image|>You're a test, Harry! I-I'm a what? [img-1]<|image|>A test. And a thumping good one at that, I'd wager. ",
				images:        [][]byte{imgBuf, imgBuf2},
				aspectRatioID: 1,
			},
		},
		{
			name:  "earlier image with mllama",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "How many hotdogs are in this image?", Images: []api.ImageData{imgBuf}},
				{Role: "assistant", Content: "There are four hotdogs."},
				{Role: "user", Content: "Which ones have mustard?"},
			},
			expect: expect{
				prompt:        "[img-0]<|image|>How many hotdogs are in this image? There are four hotdogs. Which ones have mustard? ",
				images:        [][]byte{imgBuf},
				aspectRatioID: 1,
			},
		},
		{
			name:  "too many images with mllama",
			model: mllamaModel,
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{imgBuf, imgBuf}},
			},
			expect: expect{
				error: errTooManyImages,
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			model := tt.model
			opts := api.Options{Runner: api.Runner{NumCtx: tt.limit}}
			prompt, images, err := chatPrompt(context.TODO(), &model, mockRunner{}.Tokenize, &opts, tt.msgs, nil)
			if tt.error == nil && err != nil {
				t.Fatal(err)
			} else if tt.error != nil && err != tt.error {
				t.Fatalf("expected err '%q', got '%q'", tt.error, err)
			}

			if diff := cmp.Diff(prompt, tt.prompt); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}

			if len(images) != len(tt.images) {
				t.Fatalf("expected %d images, got %d", len(tt.images), len(images))
			}

			for i := range images {
				if images[i].ID != i {
					t.Errorf("expected ID %d, got %d", i, images[i].ID)
				}

				if len(model.Config.ModelFamilies) == 0 {
					if !bytes.Equal(images[i].Data, tt.images[i]) {
						t.Errorf("expected %q, got %q", tt.images[i], images[i].Data)
					}
				} else {
					if images[i].AspectRatioID != tt.aspectRatioID {
						t.Errorf("expected aspect ratio %d, got %d", tt.aspectRatioID, images[i].AspectRatioID)
					}
				}
			}
		})
	}
}
