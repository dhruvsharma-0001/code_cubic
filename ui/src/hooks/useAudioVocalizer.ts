"use client";

import { useState, useCallback } from "react";

export function useAudioVocalizer(speechLang: "en" | "hi" = "en") {
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handleSpeak = useCallback(
    (textToSpeak?: string) => {
      if (!textToSpeak) return;

      if (isSpeaking) {
        if (typeof window !== "undefined" && window.speechSynthesis) {
          window.speechSynthesis.cancel();
        }
        setIsSpeaking(false);
        return;
      }

      try {
        if (typeof window === "undefined") return;

        // 1. Play Authentic Soundbox 3-Tone Ascending Chime: C5 (523Hz) -> E5 (659Hz) -> G5 (784Hz)
        const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtxClass) {
          const audioCtx = new AudioCtxClass();
          const now = audioCtx.currentTime;
          const notes = [523.25, 659.25, 783.99];
          notes.forEach((freq, idx) => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(freq, now + idx * 0.11);
            gain.gain.setValueAtTime(0.2, now + idx * 0.11);
            gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.11 + 0.28);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start(now + idx * 0.11);
            osc.stop(now + idx * 0.11 + 0.28);
          });
        }

        // 2. Vocalize after chime delay
        setTimeout(() => {
          if (!window.speechSynthesis) return;
          window.speechSynthesis.cancel();

          let cleanText = textToSpeak
            .replace(/\[M-\w+\]/g, "")
            .replace(/\*\*/g, "")
            .replace(/•/g, "")
            .slice(0, 260);

          if (speechLang === "hi") {
            cleanText = `Paytm Soundbox: Offline vector satyapan samapt. ${cleanText}`;
          } else {
            cleanText = `Paytm Soundbox Voice Engine: ${cleanText}`;
          }

          const utterance = new SpeechSynthesisUtterance(cleanText);
          utterance.lang = speechLang === "hi" ? "hi-IN" : "en-IN";
          utterance.rate = 1.05;

          utterance.onstart = () => setIsSpeaking(true);
          utterance.onend = () => setIsSpeaking(false);
          utterance.onerror = () => setIsSpeaking(false);

          window.speechSynthesis.speak(utterance);
        }, 400);
      } catch {
        setIsSpeaking(false);
      }
    },
    [isSpeaking, speechLang]
  );

  return { isSpeaking, handleSpeak };
}
