package cmd

import (
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
	"os/signal"
	"runtime"
	"strings"
	"unicode/utf8"

	"golang.org/x/term"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/readline"
)

// runSpeech reads text aloud: piped input as it arrives followed by any
// text arguments, the text arguments alone, or each line typed at the prompt.
func runSpeech(ctx context.Context, model, voice, text string, input io.Reader, interactive bool) error {
	if input != nil {
		return speak(ctx, model, voice, io.MultiReader(input, strings.NewReader(" "+text)))
	}
	if text != "" || !interactive {
		return speak(ctx, model, voice, strings.NewReader(text))
	}
	if _, err := player(model, ""); err != nil {
		return err
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:      ">>> ",
		Placeholder: "Enter text to speak (/bye to exit)",
	})
	if err != nil {
		return err
	}
	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}
	for {
		// Clear anything typed while audio played; readline redraws it.
		fmt.Print(readline.CursorBOL + readline.ClearToEOL)
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}
			continue
		case errors.Is(err, readline.ErrEditPrompt):
			continue
		case err != nil:
			return err
		}
		text := strings.TrimSpace(line)
		if text == "/bye" {
			return nil
		}
		if text == "" {
			continue
		}
		// Ctrl+C stops speaking and returns to the prompt.
		speakCtx, stop := signal.NotifyContext(ctx, os.Interrupt)
		if err := speak(speakCtx, model, voice, strings.NewReader(text)); err != nil && speakCtx.Err() == nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
		}
		stop()
	}
}

// speak plays text from r a few sentences at a time as it arrives, so a chat
// model piped into a speech model is heard while it is still writing. When
// stdout is not a terminal it writes one WAV instead, for example
// `ollama run model "text" > speech.wav` on a machine without speakers.
func speak(ctx context.Context, model, voice string, r io.Reader) error {
	toTerminal := term.IsTerminal(int(os.Stdout.Fd()))
	if toTerminal {
		if _, err := player(model, ""); err != nil {
			return err
		}
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Generate the next piece while the current one plays.
	type result struct {
		wav []byte
		err error
	}
	audio := make(chan result, 1)
	go func() {
		defer close(audio)
		for piece := range speechPieces(ctx, r) {
			wav, err := synthesize(ctx, model, voice, piece)
			select {
			case audio <- result{wav, err}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()

	var out []byte
	spoken := false
	for res := range audio {
		if res.err != nil {
			return res.err
		}
		spoken = true
		switch {
		case toTerminal:
			if err := play(ctx, model, res.wav); err != nil {
				return err
			}
		case out == nil:
			out = res.wav
		default:
			out = append(out, res.wav[44:]...)
		}
	}
	if !spoken {
		return fmt.Errorf("speech models require input text. Usage: ollama run %s \"your text here\"", model)
	}
	if toTerminal {
		return nil
	}
	setWAVSizes(out)
	_, err := os.Stdout.Write(out)
	return err
}

// minSpeechPiece is how much text speechPieces gathers before it sends a
// sentence on, so speech starts quickly without many tiny requests.
const minSpeechPiece = 100

// speechPieces sends text from r on as soon as it ends at a sentence or
// paragraph break, sized for /v1/audio/speech, and the rest when r ends.
func speechPieces(ctx context.Context, r io.Reader) <-chan string {
	pieces := make(chan string)
	go func() {
		defer close(pieces)
		send := func(text string) bool {
			for _, piece := range speechChunks(text, maxSpeechInput) {
				select {
				case pieces <- piece:
				case <-ctx.Done():
					return false
				}
			}
			return true
		}
		var pending string
		buf := make([]byte, 4096)
		for {
			n, err := r.Read(buf)
			pending += string(buf[:n])
			if cut := speechCut(pending); cut > 0 {
				if !send(pending[:cut]) {
					return
				}
				pending = pending[cut:]
			}
			if err != nil {
				send(pending)
				return
			}
		}
	}()
	return pieces
}

// speechCut returns where text can be cut so the part before it is spoken
// now: after the last paragraph break or sentence end, once a paragraph or at
// least minSpeechPiece characters have arrived. It returns 0 to wait.
func speechCut(text string) int {
	para := strings.LastIndex(text, "\n\n")
	cut := max(para, lastSentenceEnd(text))
	if para > 0 || (cut > 0 && utf8.RuneCountInString(text[:cut]) >= minSpeechPiece) {
		return cut
	}
	return 0
}

// maxSpeechInput is the most characters /v1/audio/speech accepts at once.
const maxSpeechInput = 4096

// speechChunks splits text into one piece per paragraph, cutting paragraphs
// longer than limit characters at the last sentence end or space that fits.
func speechChunks(text string, limit int) []string {
	var chunks []string
	for para := range strings.SplitSeq(strings.ReplaceAll(text, "\r\n", "\n"), "\n\n") {
		for p := strings.TrimSpace(para); p != ""; p = strings.TrimSpace(p) {
			end := len(p)
			if utf8.RuneCountInString(p) > limit {
				end = 0
				for range limit {
					_, size := utf8.DecodeRuneInString(p[end:])
					end += size
				}
				if i := lastSentenceEnd(p[:end]); i > 0 {
					end = i
				} else if i := strings.LastIndexAny(p[:end], " \t\n"); i > 0 {
					end = i
				}
			}
			chunks = append(chunks, strings.TrimSpace(p[:end]))
			p = p[end:]
		}
	}
	return chunks
}

// lastSentenceEnd returns the index just past the last '.', '!' or '?' in s
// that is followed by whitespace, or -1.
func lastSentenceEnd(s string) int {
	for i := len(s) - 2; i > 0; i-- {
		if strings.IndexByte(".!?", s[i]) >= 0 && (s[i+1] == ' ' || s[i+1] == '\n') {
			return i + 1
		}
	}
	return -1
}

func synthesize(ctx context.Context, model, voice, text string) ([]byte, error) {
	body, err := json.Marshal(map[string]string{"model": model, "voice": voice, "input": text, "response_format": "wav"})
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
	setWAVSizes(data)
	return data, nil
}

// setWAVSizes records the length of a WAV the server streamed as unknown,
// for players that need it.
func setWAVSizes(wav []byte) {
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	binary.LittleEndian.PutUint32(wav[40:], uint32(len(wav)-44))
}

// play uses the system's built-in WAV player.
func play(ctx context.Context, model string, wav []byte) error {
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

	p, err := player(model, f.Name())
	if err != nil {
		return err
	}
	c := exec.CommandContext(ctx, p[0], p[1:]...)
	c.Stderr = os.Stderr
	if err := c.Run(); err != nil {
		return fmt.Errorf("%s failed: %w. %s", p[0], err, saveHint(model))
	}
	return nil
}

// player returns the command that plays the WAV file at path.
func player(model, path string) ([]string, error) {
	players := [][]string{{"paplay", path}, {"pw-play", path}, {"aplay", "-q", path}, {"ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path}}
	switch runtime.GOOS {
	case "darwin":
		players = [][]string{{"afplay", path}}
	case "windows":
		players = [][]string{{"powershell", "-NoProfile", "-Command", "(New-Object Media.SoundPlayer '" + strings.ReplaceAll(path, "'", "''") + "').PlaySync()"}}
	}
	for _, p := range players {
		if _, err := exec.LookPath(p[0]); err == nil {
			return p, nil
		}
	}
	return nil, errors.New("no audio player found. " + saveHint(model))
}

func saveHint(model string) string {
	return fmt.Sprintf(`To save the audio instead, run: ollama run %s "text" > speech.wav`, model)
}
