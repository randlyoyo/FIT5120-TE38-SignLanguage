import { useEffect, useRef, useState } from "react";
import { StickFigureStage } from "../components/StickFigureStage";

interface Message {
  id: number;
  text: string;
  status: "generating" | "done";
}

// Stand-in for what a real sign-to-text pass would transcribe -- there's no
// camera capture wired up yet, so the record button cycles through a small
// rotating set instead of leaving the "+" a dead, disabled control.
const SAMPLE_TRANSCRIPTS = [
  "How are you",
  "Thank you",
  "See you later",
  "Nice to meet you",
];

/** Shell for the text-to-sign avatar feature -- real generation runs on a
 *  separate GPU inference service that isn't wired up yet (recognition's
 *  measured pipeline runs 5-8s round trip), so "Send" here only drives the
 *  placeholder stage through its states. The reply area splits into a
 *  narrower transcript column and a wide avatar column, roughly 2:5 -- the
 *  avatar is the actual reply, so it gets the bigger share. The input bar
 *  is independent of that split: a single pill centered on the page below
 *  both columns, the same as a typical chat app's composer. */
export function ConversationPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const [transcriptIndex, setTranscriptIndex] = useState(0);
  const logRef = useRef<HTMLDivElement | null>(null);

  const last = messages[messages.length - 1];
  const isGenerating = last?.status === "generating";

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const text = input.trim();
    if (!text || isGenerating) return;
    const id = Date.now();
    setMessages((prev) => [...prev, { id, text, status: "generating" }]);
    setInput("");
    window.setTimeout(() => {
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, status: "done" } : m)));
    }, 1400);
  }

  function toggleRecording() {
    if (recording) {
      setRecording(false);
      setInput(SAMPLE_TRANSCRIPTS[transcriptIndex % SAMPLE_TRANSCRIPTS.length]);
      setTranscriptIndex((i) => i + 1);
      return;
    }
    setRecording(true);
  }

  return (
    <div className="conversation-shell">
      <div className="conversation-panels">
        <div className="conversation-chat-col">
          <div className="conversation-chat-log" ref={logRef}>
            {messages.length === 0 ? (
              <p className="conversation-chat-empty">
                Type a sentence below and the avatar will perform it in Auslan, sign by sign.
              </p>
            ) : (
              messages.map((m) => (
                <div key={m.id} className="conversation-message-group">
                  <p className="conversation-message conversation-message-user">{m.text}</p>
                  <p className="conversation-message conversation-message-status">
                    {m.status === "generating" ? "Generating…" : "✓ Signed"}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="conversation-stage-area">
          <StickFigureStage
            className="conversation-stage"
            status={!last ? "Waiting for input" : isGenerating ? "Generating…" : "Signed"}
            signing={isGenerating}
          />
        </div>
      </div>

      <div className="conversation-input-bar">
        <div className="conversation-input-pill">
          <button
            type="button"
            className={`conversation-record-button ${recording ? "recording" : ""}`}
            onClick={toggleRecording}
            title={recording ? "Stop recording" : "Record a sign instead of typing"}
            aria-label={recording ? "Stop recording" : "Record a sign instead of typing"}
            aria-pressed={recording}
          >
            {recording ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="2.5" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" />
              </svg>
            )}
          </button>
          <input
            id="conversation-input"
            type="text"
            value={input}
            placeholder={recording ? "Recording… click again to stop" : "Type a sentence to sign…"}
            disabled={recording}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            type="button"
            className="conversation-send-button"
            onClick={handleSend}
            disabled={!input.trim() || isGenerating || recording}
            aria-label="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="6 11 12 5 18 11" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
