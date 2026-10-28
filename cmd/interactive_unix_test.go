//go:build linux || darwin

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/containerd/console"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

func TestInteractiveLoadSystem(t *testing.T) {
	for _, target := range []string{"new-model", "missing-model", "unloadable-model"} {
		t.Run(target, func(t *testing.T) {
			chats := make(chan api.ChatRequest, 2)
			creates := make(chan api.CreateRequest, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/api/show":
					var req api.ShowRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if req.Model == "missing-model" {
						w.WriteHeader(http.StatusNotFound)
						fmt.Fprintln(w, `{"error":"model not found"}`)
						return
					}
					json.NewEncoder(w).Encode(api.ShowResponse{System: "new model default"})
				case "/api/generate":
					if target == "unloadable-model" {
						w.WriteHeader(http.StatusNotFound)
						fmt.Fprintln(w, `{"error":"model not found"}`)
						return
					}
					fmt.Fprintln(w, `{"done":true}`)
				case "/api/chat":
					var req api.ChatRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					chats <- req
					fmt.Fprintln(w, `{"done":true,"message":{"role":"assistant","content":"hello"}}`)
				case "/api/create":
					var saved api.CreateRequest
					if err := json.NewDecoder(r.Body).Decode(&saved); err != nil {
						t.Error(err)
					}
					creates <- saved
					fmt.Fprintln(w, `{"status":"success"}`)
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			t.Setenv("OLLAMA_NOHISTORY", "1")
			t.Setenv("HOME", t.TempDir())

			master, name, err := console.NewPty()
			if err != nil {
				t.Fatal(err)
			}
			defer master.Close()
			slave, err := os.OpenFile(name, os.O_RDWR, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer slave.Close()
			reader, writer, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			defer writer.Close()
			stdin, stdout := os.Stdin, os.Stdout
			os.Stdin, os.Stdout = slave, writer
			defer func() { os.Stdin, os.Stdout = stdin, stdout }()

			output := make(chan string)
			stopOutput := make(chan struct{})
			defer close(stopOutput)
			go func() {
				defer close(output)
				buf := make([]byte, 4096)
				for {
					n, err := reader.Read(buf)
					if n > 0 {
						select {
						case output <- string(buf[:n]):
						case <-stopOutput:
							return
						}
					}
					if err != nil {
						return
					}
				}
			}()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := &cobra.Command{}
			cmd.SetContext(ctx)
			cmd.Flags().Bool("verbose", false, "")
			done := make(chan error, 1)
			finished := make(chan struct{})
			go func() {
				defer close(finished)
				done <- generateInteractive(cmd, runOptions{Model: "old-model"})
			}()
			defer func() {
				cancel()
				master.Close()
				// Unblock the session and drain its final output even if an assertion fails.
				for {
					select {
					case <-finished:
						return
					case <-output:
					}
				}
			}()

			waitPrompt := func() string {
				t.Helper()
				var text strings.Builder
				for !strings.Contains(text.String(), ">>> ") {
					select {
					case chunk := <-output:
						text.WriteString(chunk)
					case err := <-done:
						t.Fatalf("interactive session exited early: %v", err)
					case <-ctx.Done():
						t.Fatal("timed out waiting for prompt")
					}
				}
				return text.String()
			}
			waitPrompt()
			command := func(line string) string {
				t.Helper()
				if _, err := io.WriteString(master, line+"\r"); err != nil {
					t.Fatal(err)
				}
				return waitPrompt()
			}
			command("/set system old session override")
			command("/load " + target)
			shown := command("/show system")
			command("hi")
			command("/clear")
			command("hi again")
			command("/save saved-model")
			io.WriteString(master, "/bye\r")
			// Drain output until the session exits so writes cannot block.
		loop:
			for {
				select {
				case <-output:
				case err := <-done:
					if err != nil {
						t.Fatal(err)
					}
					break loop
				case <-ctx.Done():
					t.Fatal("timed out exiting interactive session")
				}
			}

			wantSystem, wantShown := "", "new model default"
			if target != "new-model" {
				wantSystem, wantShown = "old session override", "old session override"
			}
			if !strings.Contains(shown, wantShown) {
				t.Errorf("/show system output %q does not contain %q", shown, wantShown)
			}
			if len(creates) != 1 {
				t.Fatalf("got %d create requests, want 1", len(creates))
			}
			saved := <-creates
			if saved.System != wantSystem {
				t.Errorf("saved system = %q, want %q", saved.System, wantSystem)
			}
			if len(chats) != 2 {
				t.Fatalf("got %d chat requests, want 2", len(chats))
			}
			for i := range 2 {
				chat := <-chats
				var system string
				for _, message := range chat.Messages {
					if message.Role == "system" {
						system = message.Content
					}
				}
				if system != wantSystem {
					t.Errorf("chat %d system = %q, want %q", i, system, wantSystem)
				}
			}
		})
	}
}
