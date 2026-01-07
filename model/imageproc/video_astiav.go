//go:build ffmpeg && cgo

package imageproc

// #cgo pkg-config: libavformat libavcodec libavutil libswscale
// #cgo LDFLAGS: -lm -lpthread
import "C"

import (
	"fmt"
	"image"
	"log/slog"
	"os"

	"github.com/asticode/go-astiav"
)

// extractVideoFramesImpl tries embedded FFmpeg first, then falls back to system ffmpeg.
func extractVideoFramesImpl(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	// Try embedded FFmpeg first
	slog.Debug("Attempting video extraction with embedded FFmpeg", "size", len(videoData))
	frames, err := extractVideoFramesEmbedded(videoData, config)
	if err == nil {
		slog.Debug("Embedded FFmpeg extraction succeeded", "num_frames", len(frames))
		return frames, nil
	}

	slog.Debug("Embedded FFmpeg extraction failed, falling back to system ffmpeg", "error", err)
	// Fallback to system ffmpeg
	return extractVideoFramesSystem(videoData, config)
}

// extractVideoFramesEmbedded implements video frame extraction using embedded FFmpeg libs via go-astiav.
func extractVideoFramesEmbedded(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	// Allocate input format context
	inputCtx := astiav.AllocFormatContext()
	if inputCtx == nil {
		return nil, fmt.Errorf("failed to allocate format context")
	}
	defer inputCtx.Free()

	// For now, write video data to a temp file to work around IOContext limitations
	// TODO: Implement custom IOContext callbacks for reading from bytes
	tempFile := fmt.Sprintf("/tmp/ollama-video-%d.tmp", len(videoData))
	if err := os.WriteFile(tempFile, videoData, 0600); err != nil {
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	defer os.Remove(tempFile)

	// Open input file
	if err := inputCtx.OpenInput(tempFile, nil, nil); err != nil {
		return nil, fmt.Errorf("failed to open input: %w", err)
	}
	defer inputCtx.CloseInput()

	// Find stream information
	if err := inputCtx.FindStreamInfo(nil); err != nil {
		return nil, fmt.Errorf("failed to find stream info: %w", err)
	}

	// Find the first video stream
	var videoStream *astiav.Stream
	var codec *astiav.Codec
	for _, stream := range inputCtx.Streams() {
		if stream.CodecParameters().MediaType() == astiav.MediaTypeVideo {
			videoStream = stream
			codec = astiav.FindDecoder(stream.CodecParameters().CodecID())
			break
		}
	}

	if videoStream == nil {
		return nil, fmt.Errorf("no video stream found")
	}
	if codec == nil {
		return nil, fmt.Errorf("unsupported video codec")
	}

	// Allocate codec context
	codecCtx := astiav.AllocCodecContext(codec)
	if codecCtx == nil {
		return nil, fmt.Errorf("failed to allocate codec context")
	}
	defer codecCtx.Free()

	// Copy codec parameters from stream to codec context
	if err := codecCtx.FromCodecParameters(videoStream.CodecParameters()); err != nil {
		return nil, fmt.Errorf("failed to copy codec parameters: %w", err)
	}

	// Open codec
	if err := codecCtx.Open(codec, nil); err != nil {
		return nil, fmt.Errorf("failed to open codec: %w", err)
	}

	// Calculate frame sampling interval based on desired FPS
	videoFPS := float64(videoStream.AvgFrameRate().Num()) / float64(videoStream.AvgFrameRate().Den())
	if videoFPS == 0 {
		videoFPS = 30.0 // Default fallback
	}
	frameInterval := int(videoFPS / config.FPS)
	if frameInterval < 1 {
		frameInterval = 1
	}

	// Allocate packet and frame
	packet := astiav.AllocPacket()
	if packet == nil {
		return nil, fmt.Errorf("failed to allocate packet")
	}
	defer packet.Free()

	frame := astiav.AllocFrame()
	if frame == nil {
		return nil, fmt.Errorf("failed to allocate frame")
	}
	defer frame.Free()

	// Decode frames
	var frames []image.Image
	frameCount := 0

	for {
		// Read packet from input
		if err := inputCtx.ReadFrame(packet); err != nil {
			if err == astiav.ErrEof {
				// End of file reached
				break
			}
			return nil, fmt.Errorf("error reading frame: %w", err)
		}

		// Skip non-video packets
		if packet.StreamIndex() != videoStream.Index() {
			packet.Unref()
			continue
		}

		// Send packet to decoder
		if err := codecCtx.SendPacket(packet); err != nil {
			packet.Unref()
			// Ignore errors sending packet, try to receive frames
			continue
		}

		// Receive decoded frames from codec
		for {
			if err := codecCtx.ReceiveFrame(frame); err != nil {
				if err == astiav.ErrEof || err == astiav.ErrEagain {
					break
				}
				// Continue on other errors
				break
			}

			// Sample frames at the specified interval
			if frameCount%frameInterval == 0 {
				img, err := convertFrameToImage(frame, codecCtx.Width(), codecCtx.Height(), codecCtx.PixelFormat())
				if err != nil {
					return nil, fmt.Errorf("failed to convert frame: %w", err)
				}

				frames = append(frames, img)

				// Check if we've reached max frames limit
				if config.MaxFrames > 0 && len(frames) >= config.MaxFrames {
					return frames, nil
				}
			}

			frameCount++
			frame.Unref()
		}

		packet.Unref()
	}

	// Flush decoder to get remaining frames
	if err := codecCtx.SendPacket(nil); err == nil {
		for {
			if err := codecCtx.ReceiveFrame(frame); err != nil {
				break
			}

			if frameCount%frameInterval == 0 {
				img, err := convertFrameToImage(frame, codecCtx.Width(), codecCtx.Height(), codecCtx.PixelFormat())
				if err == nil {
					frames = append(frames, img)
					if config.MaxFrames > 0 && len(frames) >= config.MaxFrames {
						break
					}
				}
			}

			frameCount++
			frame.Unref()
		}
	}

	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames extracted from video")
	}

	return frames, nil
}

// convertFrameToImage converts an AVFrame to a Go image.Image
func convertFrameToImage(frame *astiav.Frame, width, height int, srcPixFmt astiav.PixelFormat) (image.Image, error) {
	// Create software scale context for pixel format conversion
	swsCtx, err := astiav.CreateSoftwareScaleContext(
		width, height, srcPixFmt,
		width, height, astiav.PixelFormatRgba,
		astiav.NewSoftwareScaleContextFlags(astiav.SoftwareScaleContextFlagBilinear),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create swscale context: %w", err)
	}
	defer swsCtx.Free()

	// Allocate destination frame
	dstFrame := astiav.AllocFrame()
	if dstFrame == nil {
		return nil, fmt.Errorf("failed to allocate destination frame")
	}
	defer dstFrame.Free()

	dstFrame.SetWidth(width)
	dstFrame.SetHeight(height)
	dstFrame.SetPixelFormat(astiav.PixelFormatRgba)

	// Allocate buffer for destination frame
	if err := dstFrame.AllocBuffer(0); err != nil {
		return nil, fmt.Errorf("failed to allocate frame buffer: %w", err)
	}

	// Scale and convert pixel format
	if err := swsCtx.ScaleFrame(frame, dstFrame); err != nil {
		return nil, fmt.Errorf("failed to scale frame: %w", err)
	}

	// Convert raw frame data to Go image
	img, err := dstFrame.Data().GuessImageFormat()
	if err != nil {
		return nil, fmt.Errorf("failed to guess image format: %w", err)
	}

	if err := dstFrame.Data().ToImage(img); err != nil {
		return nil, fmt.Errorf("failed to convert frame to image: %w", err)
	}

	return img, nil
}

// checkEmbeddedFFmpeg returns true when built with embedded FFmpeg support
func checkEmbeddedFFmpeg() bool {
	return true
}
