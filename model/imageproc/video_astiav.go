//go:build ffmpeg && cgo
// +build ffmpeg,cgo

package imageproc

// #cgo pkg-config: libavformat libavcodec libavutil libswscale
// #cgo LDFLAGS: -lm -lpthread
import "C"

import (
	"bytes"
	"fmt"
	"image"

	"github.com/asticode/go-astiav"
)

// extractVideoFramesImpl implements video frame extraction using embedded FFmpeg libs via go-astiav.
// This implementation is used when building with -tags ffmpeg,cgo and FFmpeg libraries are available.
func extractVideoFramesImpl(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	// Allocate input format context
	inputCtx := astiav.AllocFormatContext()
	if inputCtx == nil {
		return nil, fmt.Errorf("failed to allocate format context")
	}
	defer inputCtx.Free()

	// Create IO context from memory buffer
	// This allows us to decode video directly from bytes without writing to disk
	ioCtx := astiav.NewIOContext(bytes.NewReader(videoData), nil)
	inputCtx.SetPb(ioCtx)

	// Open input (probe format and read header)
	if err := inputCtx.OpenInput("", nil, nil); err != nil {
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
			codec = astiav.FindDecoder(stream.CodecParameters().CodecId())
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
	// Create destination image
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// Create swscale context for pixel format conversion
	swsCtx := astiav.AllocSwsContext(
		width, height, srcPixFmt,
		width, height, astiav.PixelFormatRgba,
		astiav.SwsScaleFlagBilinear,
		nil, nil, nil,
	)
	if swsCtx == nil {
		return nil, fmt.Errorf("failed to create swscale context")
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
	if err := swsCtx.Scale(
		frame.Data(), frame.Linesize(),
		0, height,
		dstFrame.Data(), dstFrame.Linesize(),
	); err != nil {
		return nil, fmt.Errorf("failed to scale frame: %w", err)
	}

	// Copy pixel data from AVFrame to Go image
	// dstFrame.Data()[0] contains RGBA data
	pixelDataSize := width * height * 4 // RGBA = 4 bytes per pixel
	pixelData := dstFrame.Data()[0]

	if len(pixelData) < pixelDataSize {
		return nil, fmt.Errorf("pixel data size mismatch: expected %d, got %d", pixelDataSize, len(pixelData))
	}

	copy(img.Pix, pixelData[:pixelDataSize])

	return img, nil
}

// checkEmbeddedFFmpeg returns true when built with embedded FFmpeg support
func checkEmbeddedFFmpeg() bool {
	return true
}
