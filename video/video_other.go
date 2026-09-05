//go:build !darwin

package video

// Video decoding is currently implemented only on macOS, where AVFoundation is
// part of the system and nothing has to be installed by the user. The ffmpeg
// based Linux and Windows implementations from 5852ceda are kept in git history
// and can be restored once a decoder can be shipped with Ollama itself.
func extract(path string, opts Options) (*Result, error) {
	return nil, ErrUnsupportedPlatform
}
