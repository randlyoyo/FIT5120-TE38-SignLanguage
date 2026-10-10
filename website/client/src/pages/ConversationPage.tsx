import { useEffect, useRef, useState } from "react";
import { StickFigureStage } from "../components/StickFigureStage";
import { SmplxAvatar, smplxAvatarConfigured } from "../components/SmplxAvatar";
import { SignCaptureModal } from "../components/SignCaptureModal";
import { parseSmplxPose, type SmplxClip } from "../lib/smplx";
import {
  deleteChatSession,
  fetchSignPose,
  sendChatSign,
  sendChatText,
  signChatMediaUrl,
  waitForSignVideo,
  SignChatApiError,
  type ChatResponse,
} from "../api/signChat";

interface Message {
  id: number;
  /** "Signing…" until the signed clip is recognised, then the recognised text. */
  userText: string;
  signed: boolean;
  status: "generating" | "done" | "error";
  reply?: ChatResponse["reply"];
  errorMessage?: string;
}

function messageFor(err: unknown): string {
  if (err instanceof SignChatApiError) return err.message;
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "Camera access was denied — allow it in your browser and try again.";
  }
  console.error("ConversationPage failed:", err);
  return "The avatar couldn't reply just now. Please try again.";
}

/** Sign-chat page, wired to the real backend (sign_chat/sign_chat_backend).
 *  It runs on a separate GPU host -- set VITE_SIGNCHAT_API_BASE_URL, or
 *  point it at a `SIGNCHAT_MOCK=1` instance for frontend work without a
 *  GPU. Reply video is played directly (`reply.sign.video_url`); the
 *  backend's own README calls this the quickest route to a working avatar,
 *  short of a full SMPL-X rig. The reply area splits into a narrower
 *  transcript column and a wide avatar column, roughly 2:5 -- the avatar is
 *  the actual reply, so it gets the bigger share. The input bar is a single
 *  pill centered below both columns, the same as a typical chat composer. */
export function ConversationPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [captureOpen, setCaptureOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [poseClips, setPoseClips] = useState<Record<number, SmplxClip>>({});
  const logRef = useRef<HTMLDivElement | null>(null);

  const selected = messages.find((m) => m.id === selectedId) ?? messages[messages.length - 1];
  const isGenerating = messages[messages.length - 1]?.status === "generating";

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function runTurn(id: number, call: () => Promise<ChatResponse>) {
    try {
      const res = await call();
      setSessionId(res.session_id);
      // The VRM avatar never plays the debug mp4, so there's nothing to
      // gain by waiting on it -- that used to add several needless seconds
      // (video render time) before the avatar could even start, while the
      // pose JSON it actually needs is already written by the time this
      // response comes back. Fetched here (not in a useEffect keyed on
      // `selected`) so it lands in the same state update as the reply
      // text, instead of a visible beat later.
      let clip: SmplxClip | null = null;
      if (smplxAvatarConfigured) {
        try {
          clip = parseSmplxPose(await fetchSignPose(res.reply.sign.pose_url));
        } catch (err) {
          console.error("Failed to load pose data:", err);
        }
      } else if (res.reply.sign.video_status !== "ready") {
        await waitForSignVideo(res.reply.sign.video_url);
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                status: "done",
                reply: res.reply,
                userText: res.input.recognition?.raw_text ?? m.userText,
              }
            : m
        )
      );
      if (clip) setPoseClips((prev) => ({ ...prev, [id]: clip }));
      setSelectedId(id);
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, status: "error", errorMessage: messageFor(err) } : m))
      );
      setSelectedId(id);
    }
  }

  function handleSend() {
    const text = input.trim();
    if (!text || isGenerating) return;
    const id = Date.now();
    setMessages((prev) => [...prev, { id, userText: text, signed: false, status: "generating" }]);
    setSelectedId(id);
    setInput("");
    void runTurn(id, () => sendChatText(text, sessionId));
  }

  function handleCaptureConfirmed(video: Blob) {
    setCaptureOpen(false);
    const id = Date.now();
    setMessages((prev) => [...prev, { id, userText: "Signing…", signed: true, status: "generating" }]);
    setSelectedId(id);
    // The front camera's own feed, so (like a selfie) it's mirrored --
    // README.md "mirrored: ... most front-camera recordings".
    void runTurn(id, () => sendChatSign(video, { mirrored: true, sessionId }));
  }

  function newChat() {
    if (sessionId) void deleteChatSession(sessionId);
    setMessages([]);
    setSelectedId(null);
    setSessionId(undefined);
    setInput("");
  }

  const stageStatus = !selected
      ? "Waiting for input"
      : selected.status === "generating"
        ? "Generating…"
        : selected.status === "error"
          ? "Couldn't reply"
          : "Signed";

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
                <button
                  key={m.id}
                  type="button"
                  className={`conversation-message-group ${m.id === selected?.id ? "selected" : ""}`}
                  onClick={() => m.status === "done" && setSelectedId(m.id)}
                  disabled={m.status !== "done"}
                >
                  <p className="conversation-message conversation-message-user">
                    {m.signed && <span aria-hidden="true">🤟 </span>}
                    {m.userText}
                  </p>
                  {m.status === "generating" && <p className="conversation-message-status">Generating…</p>}
                  {m.status === "done" && m.reply && (
                    <p className="conversation-message conversation-message-avatar">{m.reply.text}</p>
                  )}
                  {m.status === "error" && (
                    <p className="conversation-message-status" role="alert">
                      {m.errorMessage}
                    </p>
                  )}
                </button>
              ))
            )}
          </div>
          {messages.length > 0 && (
            <button type="button" className="conversation-new-chat" onClick={newChat}>
              New chat
            </button>
          )}
        </div>

        <div className="conversation-stage-area">
          {smplxAvatarConfigured ? (
            // Shown in every other state too (idle, generating), not just
            // once a reply lands -- a relaxed idle pose (SmplxAvatar's own
            // IDLE_POSE fallback when clip is null) reads as "waiting", not
            // the frozen T-pose rest, or the SVG placeholder this replaces.
            <div className="stick-figure-stage stick-figure-stage--avatar">
              <span className="stick-figure-status">{stageStatus}</span>
              <SmplxAvatar
                className="conversation-avatar-3d"
                clip={(selected?.status === "done" && poseClips[selected.id]) || null}
                playing
              />
            </div>
          ) : selected?.status === "done" && selected.reply ? (
            <div className="stick-figure-stage">
              <span className="stick-figure-status">{stageStatus}</span>
              <video
                key={selected.id}
                className="conversation-avatar-video"
                src={signChatMediaUrl(selected.reply.sign.video_url)}
                controls
                autoPlay
              >
                {selected.reply.sign.subtitle_url && (
                  <track
                    kind="subtitles"
                    srcLang="en"
                    src={signChatMediaUrl(selected.reply.sign.subtitle_url)}
                    default
                  />
                )}
              </video>
              <p className="stick-figure-caption">Click a message on the left to replay its reply.</p>
            </div>
          ) : (
            <StickFigureStage className="conversation-stage" status={stageStatus} signing={isGenerating} />
          )}
        </div>
      </div>

      <div className="conversation-input-bar">
        <div className="conversation-input-pill">
          <button
            type="button"
            className="conversation-record-button"
            onClick={() => setCaptureOpen(true)}
            disabled={isGenerating}
            title="Record a sign instead of typing"
            aria-label="Record a sign instead of typing"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <circle cx="12" cy="12" r="9" />
              <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" />
            </svg>
          </button>
          <input
            id="conversation-input"
            type="text"
            value={input}
            placeholder="Type a sentence to sign…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            type="button"
            className="conversation-send-button"
            onClick={handleSend}
            disabled={!input.trim() || isGenerating}
            aria-label="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="6 11 12 5 18 11" />
            </svg>
          </button>
        </div>
      </div>

      {captureOpen && (
        <SignCaptureModal onConfirm={handleCaptureConfirmed} onCancel={() => setCaptureOpen(false)} />
      )}
    </div>
  );
}
