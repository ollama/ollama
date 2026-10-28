//go:build !darwin

package video

// TODO: decode on Windows (Media Foundation) and Linux (ffmpeg) once a
// runner there consumes video.
func openNative(_, _ string) (File, error) {
	return nil, errUnsupported
}
