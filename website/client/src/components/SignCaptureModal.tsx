import { useEffect, useRef, useState } from "react";
import { captureLandmarks, type CapturePhase } from "../lib/landmarks";

// Same constants, same meaning, as PracticeVerify.tsx -- a safety net, not a
// target duration: capture normally ends on its own once the signer's
// motion settles (captureLandmarks), not on this timer.
const MAX_CAPTURE_MS = 8000;
const COUNTDOWN_S = 3;

type Phase = "requesting-camera" | "countdown" | "capturing" | "review" | "error";

interface Props {
  /** The recorded clip, once the user reviews it and confirms. */
  onConfirm: (video: Blob) => void;
  onCancel: () => void;
}

function messageFor(err: unknown): string {
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "Camera access was denied — allow it in your browser and try again.";
  }
  console.error("SignCaptureModal failed:", err);
  return "Couldn't record that. Please try again.";
}

/**
 * Record-a-sign popup for Sign Chat: countdown → record → auto-stop →
 * review the clip → confirm or retake. The auto-stop is the exact same
 * motion-based logic Practice uses (lib/landmarks.ts captureLandmarks) --
 * run here purely for its timing (when the signer goes still after
 * moving), with its landmark output thrown away, while a MediaRecorder on
 * the same stream captures the real video Sign Chat actually sends.
 */
export function SignCaptureModal({ onConfirm, onCancel }: Props) {
  const [phase, setPhase] = useState<Phase>("requesting-camera");
  const [countdown, setCountdown] = useState(COUNTDOWN_S);
  const [capturePhase, setCapturePhase] = useState<CapturePhase>("waiting");
  const [errorMessage, setErrorMessage] = useState("");
  const [reviewUrl, setReviewUrl] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const blobRef = useRef<Blob | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  // Bumped on every start() call and on unmount; a stale call checks this
  // after each await and bails out instead of touching state/the <video> --
  // needed because React StrictMode's dev-only double-invoke (mount →
  // cleanup → mount) runs this effect twice, so two start() calls race for
  // the same <video> element and the loser's play() throws AbortError.
  const generationRef = useRef(0);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function start() {
    const myGeneration = ++generationRef.current;
    const stale = () => myGeneration !== generationRef.current;

    setErrorMessage("");
    setPhase("requesting-camera");
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      if (stale()) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Camera preview isn't ready.");
      video.srcObject = stream;
      await video.play();
      if (stale()) return;

      setPhase("countdown");
      for (let n = COUNTDOWN_S; n > 0; n--) {
        setCountdown(n);
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (stale()) return;
      }

      setPhase("capturing");
      setCapturePhase("waiting");

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
      recorder.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      recorder.start();

      // video, not the mirrored on-screen preview -- same reasoning as
      // PracticeVerify: a mirrored tensor would swap left and right hands
      // in the motion signal (harmless here, since only the timing is
      // used, but there's no reason to feed it the wrong orientation).
      await captureLandmarks(video, {
        maxDurationMs: MAX_CAPTURE_MS,
        onFrame: (_elapsed, _max, p) => setCapturePhase(p),
      });
      if (stale()) return;

      const blob = await new Promise<Blob>((resolve) => {
        recorder.onstop = () => resolve(new Blob(chunksRef.current, { type: "video/webm" }));
        recorder.stop();
      });
      stopCamera();
      if (stale()) return;
      blobRef.current = blob;
      setReviewUrl(URL.createObjectURL(blob));
      setPhase("review");
    } catch (err) {
      stream?.getTracks().forEach((t) => t.stop());
      if (stale()) return;
      stopCamera();
      setErrorMessage(messageFor(err));
      setPhase("error");
    }
  }

  useEffect(() => {
    void start();
    return () => {
      generationRef.current++;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function retake() {
    if (reviewUrl) URL.revokeObjectURL(reviewUrl);
    setReviewUrl(null);
    blobRef.current = null;
    void start();
  }

  function confirm() {
    if (!blobRef.current) return;
    if (reviewUrl) URL.revokeObjectURL(reviewUrl);
    onConfirm(blobRef.current);
  }

  // Same dialog keyboard pattern as PersonalizeSessionPanel's wizard
  // (WAI-ARIA APG): focus starts inside, Tab/Shift+Tab stay confined to it,
  // Escape cancels.
  useEffect(() => {
    const modal = modalRef.current;
    const focusable = modal?.querySelectorAll<HTMLElement>(
      'button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])'
    );
    focusable?.[0]?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCancel();
        return;
      }
      if (e.key !== "Tab" || !focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  const cameraActive = phase === "requesting-camera" || phase === "countdown" || phase === "capturing";

  return (
    <div className="wizard-overlay" onClick={onCancel}>
      <div
        ref={modalRef}
        className="wizard-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Record a sign"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="wizard-header">
          <div>
            <h2 className="wizard-title">
              {phase === "review" ? "Check your video" : "Record your sign"}
            </h2>
          </div>
          <button type="button" className="wizard-close" onClick={onCancel} aria-label="Close">
            &times;
          </button>
        </div>

        {cameraActive && (
          <div className="practice-stage">
            <video ref={videoRef} className="practice-video" muted playsInline />
            {phase === "countdown" && <div className="practice-overlay">{countdown}</div>}
            {phase === "capturing" && (
              <div className="practice-overlay practice-recording">
                {capturePhase === "waiting" && "Go ahead…"}
                {capturePhase === "active" && "Recording…"}
                {capturePhase === "settling" && "Got it — finishing up…"}
              </div>
            )}
          </div>
        )}

        {phase === "review" && reviewUrl && (
          <>
            <div className="practice-stage">
              <video src={reviewUrl} className="practice-video" controls autoPlay loop playsInline />
            </div>
            <p className="sign-capture-review-hint">
              Does that look right? You can re-record if you missed the start or end.
            </p>
            <div className="sign-capture-review-actions">
              <button type="button" className="back-link" onClick={retake}>
                ↺ Retake
              </button>
              <button type="button" className="sign-capture-confirm" onClick={confirm}>
                Use this video
              </button>
            </div>
          </>
        )}

        {phase === "error" && (
          <>
            <p role="alert" className="practice-error">
              {errorMessage}
            </p>
            <div className="sign-capture-review-actions">
              <button type="button" className="back-link" onClick={onCancel}>
                Cancel
              </button>
              <button type="button" className="sign-capture-confirm" onClick={retake}>
                Try again
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
