package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"golang.org/x/term"

	"github.com/ollama/ollama/envconfig"
)

// runSpeech reads text aloud, or reads each line typed at the prompt aloud.
func runSpeech(ctx context.Context, model, text string, interactive bool) error {
	if text != "" || !interactive {
		if text == "" {
			return fmt.Errorf("speech models require input text. Usage: ollama run %s \"your text here\"", model)
		}
		return speak(ctx, model, text)
	}
	scanner := bufio.NewScanner(os.Stdin)
	for fmt.Print(">>> "); scanner.Scan(); fmt.Print(">>> ") {
		text := strings.TrimSpace(scanner.Text())
		if text == "/bye" {
			return nil
		}
		if text == "" {
			continue
		}
		if err := speak(ctx, model, text); err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
		}
	}
	fmt.Println()
	return scanner.Err()
}

// speak plays the audio, or writes it as WAV when stdout is not a terminal,
// for example `ollama run model "text" > speech.wav` on a machine without
// speakers.
func speak(ctx context.Context, model, text string) error {
	wav, err := synthesize(ctx, model, text)
	if err != nil {
		return err
	}
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		_, err := os.Stdout.Write(wav)
		return err
	}
	return play(model, wav)
}

func synthesize(ctx context.Context, model, text string) ([]byte, error) {
	body, err := json.Marshal(map[string]string{"model": model, "input": text, "response_format": "wav"})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, envconfig.Host().JoinPath("v1/audio/speech").String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		var e struct{ Error struct{ Message string } }
		if json.Unmarshal(data, &e) == nil && e.Error.Message != "" {
			return nil, errors.New(e.Error.Message)
		}
		return nil, fmt.Errorf("speech request failed: %s", res.Status)
	}
	if len(data) <= 44 {
		return nil, errors.New("speech response contained no audio")
	}
	// The server streams WAV of unknown length; record the length for players.
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	binary.LittleEndian.PutUint32(data[40:], uint32(len(data)-44))
	return data, nil
}

// play uses the system's built-in WAV player.
func play(model string, wav []byte) error {
	f, err := os.CreateTemp("", "ollama-*.wav")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(wav); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	path := f.Name()
	players := [][]string{{"paplay", path}, {"pw-play", path}, {"aplay", "-q", path}, {"ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path}}
	switch runtime.GOOS {
	case "darwin":
		players = [][]string{{"afplay", path}}
	case "windows":
		players = [][]string{{"powershell", "-NoProfile", "-Command", "(New-Object Media.SoundPlayer '" + strings.ReplaceAll(path, "'", "''") + "').PlaySync()"}}
	}
	hint := fmt.Sprintf(`to save the audio instead, redirect it to a file: ollama run %s "text" > speech.wav`, model)
	for _, p := range players {
		if _, err := exec.LookPath(p[0]); err != nil {
			continue
		}
		c := exec.Command(p[0], p[1:]...)
		c.Stderr = os.Stderr
		if err := c.Run(); err != nil {
			return fmt.Errorf("%s failed: %w; %s", p[0], err, hint)
		}
		return nil
	}
	return errors.New("no audio player found; " + hint)
}
