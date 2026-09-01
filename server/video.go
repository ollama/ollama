package server

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/video"
)

// maxVideoFrames bounds what a single video contributes to the context. Each
// frame costs roughly as much as an image, so an unbounded 1fps sampling would
// push out the rest of the conversation on longer clips.
const maxVideoFrames = 16

// expandVideos replaces every video in images with the frames and the audio track
// decoded from it.
//
// Videos are recognized by their content, not by a separate request field: the
// media pipeline already sorts images from audio by sniffing bytes (see
// llm.DetectMediaKind), and audio is passed through `images` today. Expanding
// here, before anything else looks at the request, keeps the rest of the pipeline
// unchanged — prompt tagging, the context estimate and the single-image check for
// mllama all count the extracted frames as what they are.
func expandVideos(images []api.ImageData) ([]api.ImageData, error) {
	found := false
	for _, data := range images {
		if video.IsVideo(data) {
			found = true
			break
		}
	}
	if !found {
		return images, nil
	}

	out := make([]api.ImageData, 0, len(images)+maxVideoFrames)
	for _, data := range images {
		if !video.IsVideo(data) {
			out = append(out, data)
			continue
		}

		res, err := video.ExtractBytes(data, video.Options{
			MaxFrames:    maxVideoFrames,
			ExtractAudio: true,
		})
		if err != nil {
			return nil, fmt.Errorf("video: %w", err)
		}

		for _, frame := range res.Frames {
			out = append(out, api.ImageData(frame))
		}
		if len(res.Audio) > 0 {
			out = append(out, api.ImageData(res.Audio))
		}
	}

	return out, nil
}

// expandVideoMessages applies expandVideos to every message in a chat request.
func expandVideoMessages(msgs []api.Message) error {
	for i := range msgs {
		if len(msgs[i].Images) == 0 {
			continue
		}
		expanded, err := expandVideos(msgs[i].Images)
		if err != nil {
			return err
		}
		msgs[i].Images = expanded
	}
	return nil
}
