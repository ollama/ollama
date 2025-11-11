import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Label } from "@/components/ui/fieldset";
import {
  MicrophoneIcon,
  SpeakerWaveIcon,
  StopIcon,
} from "@heroicons/react/20/solid";
import { transcribeAudio, synthesizeSpeech } from "@/api";

interface VoiceControlsProps {
  onTranscriptionComplete?: (text: string) => void;
  apiKey: string;
}

export default function VoiceControls({
  onTranscriptionComplete,
  apiKey,
}: VoiceControlsProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [textToSpeak, setTextToSpeak] = useState("");
  const [selectedVoice, setSelectedVoice] = useState("alloy");
  const [audioURL, setAudioURL] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const audioFile = new File([audioBlob], "recording.webm", {
          type: "audio/webm",
        });

        // Transcribe the audio
        setIsTranscribing(true);
        try {
          const text = await transcribeAudio(audioFile, apiKey);
          if (onTranscriptionComplete) {
            onTranscriptionComplete(text);
          }
        } catch (error) {
          console.error("Transcription error:", error);
          alert(
            "Failed to transcribe audio: " +
              (error instanceof Error ? error.message : "Unknown error"),
          );
        } finally {
          setIsTranscribing(false);
        }

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Failed to access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSynthesizeSpeech = async () => {
    if (!textToSpeak.trim()) {
      alert("Please enter text to synthesize");
      return;
    }

    setIsSynthesizing(true);
    try {
      const audioBlob = await synthesizeSpeech(textToSpeak, apiKey, selectedVoice);
      const url = URL.createObjectURL(audioBlob);
      setAudioURL(url);

      // Auto-play the audio
      const audio = new Audio(url);
      audio.play();
    } catch (error) {
      console.error("Speech synthesis error:", error);
      alert(
        "Failed to synthesize speech: " +
          (error instanceof Error ? error.message : "Unknown error"),
      );
    } finally {
      setIsSynthesizing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">Voice Input</h3>
        <p className="text-sm text-gray-400 mb-4">
          Record audio and transcribe it to text using Whisper AI
        </p>

        <div className="flex gap-2 items-center">
          {!isRecording ? (
            <Button
              onClick={startRecording}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700"
            >
              <MicrophoneIcon className="h-5 w-5" />
              Start Recording
            </Button>
          ) : (
            <Button
              onClick={stopRecording}
              className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 animate-pulse"
            >
              <StopIcon className="h-5 w-5" />
              Stop Recording
            </Button>
          )}

          {isTranscribing && (
            <span className="text-sm text-gray-400">Transcribing...</span>
          )}
        </div>
      </div>

      <div className="border-t border-gray-700 pt-6">
        <h3 className="text-lg font-semibold text-white mb-2">
          Text-to-Speech
        </h3>
        <p className="text-sm text-gray-400 mb-4">
          Convert text to natural-sounding speech
        </p>

        <div className="space-y-4">
          <Field>
            <Label>Text to Speak</Label>
            <textarea
              className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white min-h-[100px]"
              placeholder="Enter text to convert to speech..."
              value={textToSpeak}
              onChange={(e) => setTextToSpeak(e.target.value)}
            />
          </Field>

          <Field>
            <Label>Voice</Label>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white"
            >
              <option value="alloy">Alloy (Neutral)</option>
              <option value="echo">Echo (Male)</option>
              <option value="fable">Fable (British Male)</option>
              <option value="onyx">Onyx (Deep Male)</option>
              <option value="nova">Nova (Female)</option>
              <option value="shimmer">Shimmer (Soft Female)</option>
            </select>
          </Field>

          <Button
            onClick={handleSynthesizeSpeech}
            disabled={isSynthesizing}
            className="flex items-center gap-2"
          >
            <SpeakerWaveIcon className="h-5 w-5" />
            {isSynthesizing ? "Synthesizing..." : "Generate Speech"}
          </Button>

          {audioURL && (
            <div className="mt-4">
              <audio controls src={audioURL} className="w-full" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
