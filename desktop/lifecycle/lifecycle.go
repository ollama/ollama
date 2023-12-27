package lifecycle

import (
	"context"
	"log"
	"runtime"

	"github.com/jmorganca/ollama/desktop/tray"
)

func GetOllamaName() string {
	if runtime.GOOS == "windows" {
		return "ollama.exe"
	} else {
		return "ollama"
	}
}

func Run() {
	InitLogging("app.log")

	ctx, cancel := context.WithCancel(context.Background())

	t, err := tray.NewTray(DoUpgrade)
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}

	var done <-chan int
	if IsServerRunning(ctx) {
		log.Printf("XXX detected server already running")
		// TODO - should we fail fast, try to kill it, or just ignore?
	} else {
		log.Printf("XXX spawning server")
		done, err = SpawnServer(ctx, CLIName)
		if err != nil {
			log.Printf("Failed to spawn server %s", err)
		}
	}

	StartBackgroundUpdaterChecker(ctx, t.SetUpdateAvailable)

	t.Run()
	cancel()
	log.Printf("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	log.Printf("Done.")
}
