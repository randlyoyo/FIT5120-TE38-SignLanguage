import { useEffect, useRef, useState } from "react";
import { identifySign, RecognizeApiError, type IdentifyCandidate } from "../api/recognize";
import { captureLandmarks, type CapturePhase } from "../lib/landmarks";

// Same safety net as PracticeVerify -- capture normally ends on its own once
// the signer goes still, this only bounds the worst case.
const MAX_CAPTURE_MS = 8000;
const COUNTDOWN_S = 3;

type Phase = "requesting-camera" | "countdown" | "capturing" | "checking" | "error";

const CAMERA_ACTIVE_PHASES: Phase[] = ["requesting-camera", "countdown", "capturing", "checking"];

interface Props {
  /** Called the moment the model returns its top-4 candidates -- there's no
   *  intermediate "pick one" step here, the caller jumps straight to showing
   *  all four (recognition/API.md's identify is a guess across the whole
   *  vocabulary, so handing back four real sign videos to compare against is
   *  more useful than committing to a single, roughly 1-in-8-wrong pick). */
  onResults: (candidates: IdentifyCandidate[]) => void;
  onClose: () => void;
}

function messageFor(err: unknown): string {
  if (err instanceof RecognizeApiError) {
    switch (err.code) {
      case "unusable_capture":
        return "Didn't catch enough movement there — make sure both hands are in frame and try again.";
      case "model_unavailable":
        return "Sign search isn't switched on for this site yet.";
      default:
        return "Something went wrong reading that. Please try again.";
    }
  }
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "Camera access was denied — allow it in your browser and try again.";
  }
  console.error("GestureSearch failed:", err);
  return "Sign search couldn't start. Please try again in a moment.";
}

/**
 * "Identify" search flow (recognition/API.md §5): the signer hasn't chosen a
 * word yet, so this guesses one from the whole vocabulary. Rather than
 * committing to a single (roughly 1-in-8 wrong, per API.md) guess, it hands
 * the top-4 candidates straight to the caller, which jumps to a results view
 * showing all four real sign videos side by side.
 */
export function GestureSearch({ onResults, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>("requesting-camera");
  const [countdown, setCountdown] = useState(COUNTDOWN_S);
  const [capturePhase, setCapturePhase] = useState<CapturePhase>("waiting");
  const [errorMessage, setErrorMessage] = useState("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  useEffect(() => stopCamera, []);

  async function start() {
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
      const capture = await captureLandmarks(video, {
        maxDurationMs: MAX_CAPTURE_MS,
        onFrame: (_elapsed, _max, p) => setCapturePhase(p),
      });

      setPhase("checking");
      const result = await identifySign(capture);
      onResults(result.candidates);
    } catch (err) {
      setErrorMessage(messageFor(err));
      setPhase("error");
    } finally {
      stopCamera();
    }
  }

  useEffect(() => {
    start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cameraActive = CAMERA_ACTIVE_PHASES.includes(phase);

  return (
    <div className="gesture-search">
      <div className="gesture-search-header">
        <p className="gesture-search-title">Search by signing</p>
        <button type="button" className="gesture-search-close" onClick={onClose} aria-label="Close">
          &times;
        </button>
      </div>

      {cameraActive && (
        <div className="practice-stage gesture-search-stage">
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

      {phase === "error" && (
        <div className="gesture-search-results">
          <p role="alert" className="practice-error">
            {errorMessage}
          </p>
          <button type="button" className="practice-start-button gesture-search-retry" onClick={start}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
