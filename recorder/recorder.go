package recorder

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"golang.org/x/sys/unix"
	"golang.org/x/term"

	"github.com/gordonklaus/portaudio"
)

const (
	sampleRate    = 16000
	numChannels   = 1
	bitsPerSample = 16
)

func RecordAudio(f *os.File) error {
	fmt.Print("Recording. Press any key to stop.\n\n")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)

	portaudio.Initialize()
	defer portaudio.Terminate()

	in := make([]int16, 64)
	stream, err := portaudio.OpenDefaultStream(numChannels, 0, sampleRate, len(in), in)
	if err != nil {
		return err
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		return err
	}

	// Write WAV header with placeholder sizes
	writeWavHeader(f, sampleRate, numChannels, bitsPerSample)

	var totalSamples uint32

	// Set up terminal input reading
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return err
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	// Create a channel to handle the stop signal
	stop := make(chan struct{})

	go func() {
		_, err := unix.Read(int(os.Stdin.Fd()), make([]byte, 1))
		if err != nil {
			fmt.Println("Error reading from stdin:", err)
			return
		}
		// Send signal to stop recording
		stop <- struct{}{}
	}()

loop:
	for {
		err = stream.Read()
		if err != nil {
			return err
		}

		err = binary.Write(f, binary.LittleEndian, in)
		if err != nil {
			return err
		}
		totalSamples += uint32(len(in))

		select {
		case <-stop:
			break loop
		case <-sig:
			break loop
		default:
		}
	}

	err = stream.Stop()
	if err != nil {
		return err
	}

	// Update WAV header with actual sizes
	updateWavHeader(f, totalSamples, numChannels, bitsPerSample)

	return nil
}

func writeWavHeader(f *os.File, sampleRate int, numChannels int, bitsPerSample int) {
	subchunk1Size := 16
	audioFormat := 1
	byteRate := sampleRate * numChannels * (bitsPerSample / 8)
	blockAlign := numChannels * (bitsPerSample / 8)

	// Write the RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(0)) // Placeholder for file size
	f.Write([]byte("WAVE"))

	// Write the fmt subchunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(subchunk1Size))
	binary.Write(f, binary.LittleEndian, uint16(audioFormat))
	binary.Write(f, binary.LittleEndian, uint16(numChannels))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate))
	binary.Write(f, binary.LittleEndian, uint32(byteRate))
	binary.Write(f, binary.LittleEndian, uint16(blockAlign))
	binary.Write(f, binary.LittleEndian, uint16(bitsPerSample))

	// Write the data subchunk header
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(0)) // Placeholder for data size
}

func updateWavHeader(f *os.File, totalSamples uint32, numChannels int, bitsPerSample int) {
	fileSize := 36 + (totalSamples * uint32(numChannels) * uint32(bitsPerSample/8))
	dataSize := totalSamples * uint32(numChannels) * uint32(bitsPerSample/8)

	// Seek to the start of the file and write updated sizes
	f.Seek(4, 0)
	binary.Write(f, binary.LittleEndian, uint32(fileSize))

	f.Seek(40, 0)
	binary.Write(f, binary.LittleEndian, uint32(dataSize))
}
