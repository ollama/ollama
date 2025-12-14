// +build ignore

package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/whisper"
)

func main() {
	modelPath := `D:\Labs\ollama\models\ggml-tiny.bin`
	audioPath := `D:\Labs\ollama\models\jfk.wav`

	fmt.Println("=== Test Whisper.cpp dans Ollama ===")
	fmt.Println()

	// Vérifier les fichiers
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Erreur: modèle non trouvé: %s\n", modelPath)
		os.Exit(1)
	}
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		fmt.Printf("Erreur: audio non trouvé: %s\n", audioPath)
		os.Exit(1)
	}

	fmt.Printf("Modèle: %s\n", modelPath)
	fmt.Printf("Audio: %s\n", audioPath)
	fmt.Println()

	// Charger le modèle
	fmt.Println("Chargement du modèle...")
	params := whisper.DefaultContextParams()
	params.UseGPU = false // CPU seulement pour le test
	ctx, err := whisper.NewContext(modelPath, params)
	if err != nil {
		fmt.Printf("Erreur lors du chargement: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Free()

	fmt.Printf("Modèle chargé! Multilingue: %v\n", ctx.IsMultilingual())
	fmt.Println()

	// Charger l'audio WAV
	fmt.Println("Chargement de l'audio...")
	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		fmt.Printf("Erreur lecture audio: %v\n", err)
		os.Exit(1)
	}

	// Convertir WAV en PCM float32 (16kHz mono)
	samples, err := wavToPCM(audioData)
	if err != nil {
		fmt.Printf("Erreur conversion audio: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Audio chargé: %d échantillons (%.2f secondes)\n", len(samples), float64(len(samples))/16000.0)
	fmt.Println()

	// Transcrire
	fmt.Println("Transcription en cours...")
	transcribeParams := whisper.DefaultTranscribeParams()
	transcribeParams.Language = "en"
	transcribeParams.PrintProgress = true

	segments, err := ctx.Transcribe(samples, transcribeParams)
	if err != nil {
		fmt.Printf("Erreur transcription: %v\n", err)
		os.Exit(1)
	}

	// Afficher les résultats
	fmt.Println()
	fmt.Println("=== RÉSULTAT ===")
	fmt.Printf("Nombre de segments: %d\n\n", len(segments))

	for i, seg := range segments {
		fmt.Printf("[%d] [%s --> %s] %s\n",
			i,
			formatDuration(seg.Start),
			formatDuration(seg.End),
			seg.Text)
	}

	fmt.Println()
	fmt.Println("=== TEST RÉUSSI ! ===")
}

func formatDuration(d interface{}) string {
	// time.Duration
	switch v := d.(type) {
	case int64:
		ms := v
		s := ms / 1000
		m := s / 60
		s = s % 60
		ms = ms % 1000
		return fmt.Sprintf("%02d:%02d.%03d", m, s, ms)
	default:
		return fmt.Sprintf("%v", d)
	}
}

func wavToPCM(data []byte) ([]float32, error) {
	// Simple WAV parser pour 16-bit PCM
	if len(data) < 44 {
		return nil, fmt.Errorf("fichier WAV trop petit")
	}

	// Vérifier le header RIFF
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("pas un fichier WAV valide")
	}

	// Trouver le chunk "data"
	offset := 12
	var dataStart, dataSize int
	for offset < len(data)-8 {
		chunkID := string(data[offset : offset+4])
		chunkSize := int(data[offset+4]) | int(data[offset+5])<<8 | int(data[offset+6])<<16 | int(data[offset+7])<<24

		if chunkID == "data" {
			dataStart = offset + 8
			dataSize = chunkSize
			break
		}
		offset += 8 + chunkSize
		if chunkSize%2 != 0 {
			offset++ // padding
		}
	}

	if dataStart == 0 {
		return nil, fmt.Errorf("chunk data non trouvé")
	}

	// Convertir 16-bit PCM en float32
	numSamples := dataSize / 2
	samples := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		idx := dataStart + i*2
		if idx+1 >= len(data) {
			break
		}
		sample := int16(data[idx]) | int16(data[idx+1])<<8
		samples[i] = float32(sample) / 32768.0
	}

	return samples, nil
}
