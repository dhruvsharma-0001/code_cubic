"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface UseVoiceRecognitionProps {
  lang?: "en-IN" | "hi-IN";
  onResult: (transcript: string) => void;
}

export function useVoiceRecognition({ lang = "en-IN", onResult }: UseVoiceRecognitionProps) {
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition =
        (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = lang;

        recognition.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          setIsListening(false);
          onResult(transcript);
        };

        recognition.onerror = () => {
          setIsListening(false);
        };

        recognition.onend = () => {
          setIsListening(false);
        };

        recognitionRef.current = recognition;
      }
    }
  }, [lang, onResult]);

  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      alert("Web Speech API is not supported in this browser. Please use Google Chrome or Edge.");
      return;
    }
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setIsListening(true);
      recognitionRef.current.start();
    }
  }, [isListening]);

  return {
    isListening,
    toggleListening,
  };
}
