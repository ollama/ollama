package qwen3_5

import (
	"errors"

	"github.com/ollama/ollama/mlxrunner/model/videoclip"
)

// qwenVideo is the family's shipped video_preprocessor_config.json under
// the reference processor's sampling defaults: 2 fps, 4 to 768 frames.
var qwenVideo = videoclip.Config{FPS: 2, MinFrames: 4, MaxFrames: 768, MinPixels: 4096, MaxPixels: 25165824}

func (m *Model) openVideo(data []byte) (*videoclip.Clip, error) {
	// The timestamps leading each frame pair are tokenized here.
	if m.MM.VideoTokenID == 0 || m.Vision.TemporalPatchSize != 2 || m.tok == nil {
		return nil, errors.New("this model does not support video input")
	}
	return videoclip.Open(data, qwenVideo, m.Vision.PatchSize, m.Vision.SpatialMergeSize)
}
