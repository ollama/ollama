package video

import (
	"bytes"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Windows implementation: shell out to ffmpeg, same as Linux.
// Media Foundation via syscall is planned but ffmpeg is simpler for v1
// and commonly available via winget/chocolatey/scoop.
//
// TODO: implement Media Foundation via golang.org/x/sys/windows for
// zero-dependency video extraction on Windows.

func extract(path string, opts Options) (*Result, error) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		return nil, fmt.Errorf("video support requires ffmpeg; install it with: winget install ffmpeg (or scoop install ffmpeg)")
	}

	duration, err := probeDuration(path)
	if err != nil {
		return nil, fmt.Errorf("failed to probe video: %w", err)
	}

	frameCount := opts.MaxFrames
	if duration < float64(frameCount) {
		frameCount = int(duration)
	}
	if frameCount < 1 {
		frameCount = 1
	}

	fps := float64(frameCount) / duration

	result := &Result{}

	args := []string{
		"-i", path,
		"-vf", fmt.Sprintf("fps=%.4f", fps),
		"-frames:v", strconv.Itoa(frameCount),
		"-f", "image2pipe",
		"-c:v", "mjpeg",
		"-q:v", "5",
		"pipe:1",
	}

	cmd := exec.Command(ffmpeg, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg frame extraction failed: %s", stderr.String())
	}

	result.Frames = splitJPEGs(stdout.Bytes())

	if opts.ExtractAudio {
		audio, err := extractAudio(ffmpeg, path)
		if err == nil && len(audio) > 44 {
			result.Audio = audio
		}
	}

	return result, nil
}

func probeDuration(path string) (float64, error) {
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		ffmpeg, _ := exec.LookPath("ffmpeg")
		cmd := exec.Command(ffmpeg, "-i", path)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		cmd.Run()
		return parseDurationFromFFmpeg(stderr.String())
	}

	cmd := exec.Command(ffprobe,
		"-v", "quiet",
		"-show_entries", "format=duration",
		"-of", "csv=p=0",
		path,
	)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return 0, err
	}

	return strconv.ParseFloat(strings.TrimSpace(stdout.String()), 64)
}

func parseDurationFromFFmpeg(output string) (float64, error) {
	idx := strings.Index(output, "Duration: ")
	if idx < 0 {
		return 0, fmt.Errorf("could not find duration in ffmpeg output")
	}
	durStr := output[idx+10:]
	commaIdx := strings.Index(durStr, ",")
	if commaIdx > 0 {
		durStr = durStr[:commaIdx]
	}
	durStr = strings.TrimSpace(durStr)

	parts := strings.Split(durStr, ":")
	if len(parts) != 3 {
		return 0, fmt.Errorf("unexpected duration format: %s", durStr)
	}
	hours, _ := strconv.ParseFloat(parts[0], 64)
	mins, _ := strconv.ParseFloat(parts[1], 64)
	secs, _ := strconv.ParseFloat(parts[2], 64)
	return hours*3600 + mins*60 + secs, nil
}

func extractAudio(ffmpeg, path string) ([]byte, error) {
	args := []string{
		"-i", path,
		"-ar", "16000",
		"-ac", "1",
		"-f", "wav",
		"pipe:1",
	}

	cmd := exec.Command(ffmpeg, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg audio extraction failed: %s", stderr.String())
	}

	return stdout.Bytes(), nil
}

func splitJPEGs(data []byte) [][]byte {
	var frames [][]byte
	start := -1

	for i := 0; i < len(data)-1; i++ {
		if data[i] == 0xFF && data[i+1] == 0xD8 {
			start = i
		} else if data[i] == 0xFF && data[i+1] == 0xD9 && start >= 0 {
			frames = append(frames, data[start:i+2])
			start = -1
		}
	}

	return frames
}
