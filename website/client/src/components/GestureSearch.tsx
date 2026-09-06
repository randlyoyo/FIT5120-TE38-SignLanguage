import { useEffect, useRef, useState } from "react";
import { identifySign, RecognizeApiError, type IdentifyResult } from "../api/recognize";
import { captureLandmarks, type CapturePhase } from "../lib/landmarks";

// Same safety net as PracticeVerify -- capture normally ends on its own once
// the signer goes still, this only bounds the worst case.
const MAX_CAPTURE_MS = 8000;
const COUNTDOWN_S = 3;

type Phase = "requesting-camera" | "countdown" | "capturing" | "checking" | "result" | "error";

const CAMERA_ACTIVE_PHASES: Phase[] = ["requesting-camera", "countdown", "capturing", "checking"];

interface Props {
  /** Called when the signer picks one of the returned candidate words. The
   *  popover closes itself right after -- the caller decides what "picking a
   *  word" means (here: drop it into the search query). */
  onPick: (word: string) => void;
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
 * word yet, so this guesses one from the whole vocabulary and hands back its
 * top-4 candidates for the signer to pick from, rather than committing to a
 * single (roughly 1-in-8 wrong, per API.md) guess on its own.
 */
export function GestureSearch({ onPick, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>("requesting-camera");
  const [countdown, setCountdown] = useState(COUNTDOWN_S);
  const [capturePhase, setCapturePhase] = useState<CapturePhase>("waiting");
  const [result, setResult] = useState<IdentifyResult | null>(null);
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
      const capture = await captureLandmarks(video, {
        maxDurationMs: MAX_CAPTURE_MS,
        onFrame: (_elapsed, _max, p) => setCapturePhase(p),
      });

      setPhase("checking");
      setResult(await identifySign(capture));
      setPhase("result");
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

      {phase === "result" && result && (
        <div className="gesture-search-results">
          <p className="gesture-search-hint">
            {result.confident ? "Best matches:" : "Not sure — closest matches:"}
          </p>
          <ul className="gesture-search-candidates">
            {result.candidates.map((c) => (
              <li key={c.word}>
                <button type="button" onClick={() => onPick(c.word)}>
                  {c.word}
                </button>
              </li>
            ))}
          </ul>
          <button type="button" className="practice-start-button gesture-search-retry" onClick={start}>
            Try again
          </button>
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
