import { useEffect, useRef, useState } from "react";
import { RecognizeApiError, verifySign, type VerifyResult } from "../api/recognize";
import { captureLandmarks, type CapturePhase } from "../lib/landmarks";

// A safety net, not a target duration: capture normally ends on its own a
// short moment after the signer goes still (captureLandmarks' motion-based
// auto-stop), not on a fixed timer -- a fixed window either rushes a slower
// signer or leaves a fast one waiting out silence for no reason. This only
// bounds the worst case (motion never settles, e.g. poor lighting). The
// countdown is what answers "when do I start" -- capture begins the
// instant it hits zero.
const MAX_CAPTURE_MS = 8000;
const COUNTDOWN_S = 3;

type Phase = "idle" | "requesting-camera" | "countdown" | "capturing" | "checking" | "result" | "error";

const CAMERA_ACTIVE_PHASES: Phase[] = ["requesting-camera", "countdown", "capturing", "checking"];

interface Props {
  gloss: string;
}

function messageFor(err: unknown): string {
  if (err instanceof RecognizeApiError) {
    switch (err.code) {
      case "unusable_capture":
        return "Didn't catch enough movement there — make sure both hands are in frame and try again.";
      case "model_unavailable":
        return "Practice mode isn't switched on for this site yet.";
      case "unknown_word":
        return "This sign isn't in the recognizer's vocabulary yet.";
      default:
        return "Something went wrong checking that. Please try again.";
    }
  }
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "Camera access was denied — allow it in your browser and try again.";
  }
  // Anything else (a MediaPipe WASM/model load failure, in particular) is an
  // internal error whose real message is a raw engine stack trace, not
  // something a learner should ever see -- log it for us, show them a plain
  // "try again" instead.
  console.error("PracticeVerify failed:", err);
  return "Practice mode couldn't start. Please try again in a moment.";
}

/**
 * "Verify" practice flow (recognition/API.md §5): the learner has already
 * chosen this sign, so this only asks "did that attempt match it" -- the
 * reliable mode (EER 0.73%), unlike "identify" which guesses from the whole
 * vocabulary and is wrong roughly one time in eight.
 */
export function PracticeVerify({ gloss }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [countdown, setCountdown] = useState(COUNTDOWN_S);
  const [capturePhase, setCapturePhase] = useState<CapturePhase>("waiting");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  useEffect(() => stopCamera, []);

  async function start() {
    setResult(null);
    setErrorMessage("");
    setPhase("requesting-camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) throw new Error("Camera preview isn't ready.");
      video.srcObject = stream;
      await video.play();

      setPhase("countdown");
      for (let n = COUNTDOWN_S; n > 0; n--) {
        setCountdown(n);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      setPhase("capturing");
      setCapturePhase("waiting");
      // video, not the mirrored on-screen preview -- API.md §2: a mirrored
      // tensor swaps left and right hands and scores a correct sign wrong.
      const capture = await captureLandmarks(video, {
        maxDurationMs: MAX_CAPTURE_MS,
        onFrame: (_elapsed, _max, p) => setCapturePhase(p),
      });

      setPhase("checking");
      setResult(await verifySign(gloss, capture));
      setPhase("result");
    } catch (err) {
      setErrorMessage(messageFor(err));
      setPhase("error");
    } finally {
      stopCamera();
    }
  }

  const cameraActive = CAMERA_ACTIVE_PHASES.includes(phase);

  return (
    <div className="practice-verify">
      {(phase === "idle" || phase === "error" || phase === "result") && (
        <button type="button" className="practice-start-button" onClick={start}>
          {phase === "idle" ? "Try it yourself" : "Try again"}
        </button>
      )}

      {cameraActive && (
        <div className="practice-stage">
          {/* Mirrored for the user's own comfort -- a CSS transform on the
              display only. The underlying video element MediaPipe reads
              from is never flipped. */}
          <video ref={videoRef} className="practice-video" muted playsInline />
          {phase === "countdown" && <div className="practice-overlay">{countdown}</div>}
          {phase === "capturing" && (
            <div className="practice-overlay practice-recording">
              {capturePhase === "waiting" && "Go ahead…"}
              {capturePhase === "active" && "Recording…"}
              {capturePhase === "settling" && "Got it — finishing up…"}
            </div>
          )}
          {phase === "checking" && <div className="practice-overlay">Checking…</div>}
        </div>
      )}

      {phase === "result" && result && (
        <div className={`practice-result ${result.matched ? "matched" : "not-matched"}`}>
          <p className="practice-result-verdict">
            {result.matched ? "That's a match." : "Not quite — give it another go."}
          </p>
          <p className="practice-result-detail">
            distance {result.distance.toFixed(3)} (threshold {result.threshold.toFixed(3)}) ·{" "}
            {result.frames} frames used
          </p>
        </div>
      )}

      {phase === "error" && (
        <p role="alert" className="practice-error">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
