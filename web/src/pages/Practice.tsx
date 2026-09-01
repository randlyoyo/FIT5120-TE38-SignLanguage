import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { HandSkeletonCanvas } from "../components/HandSkeletonCanvas";
import { useSign, useSigns } from "../context/SignsContext";
import { detectFrame, getHandLandmarker } from "../lib/mediapipeHands";
import { SignRecorder, type RecognitionResult } from "../lib/signRecorder";
import { recordAttempt } from "../lib/storage";
import type { FrameLandmarks } from "../lib/types";

const VIDEO_CONSTRAINTS: MediaStreamConstraints = {
  video: { width: 480, height: 360, facingMode: "user" },
  audio: false,
};

export function Practice() {
  const { id } = useParams<{ id: string }>();
  const targetSign = useSign(id);
  const { signs, loading } = useSigns();

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const recorderRef = useRef(new SignRecorder());
  const signsRef = useRef(signs);
  signsRef.current = signs;

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveFrame, setLiveFrame] = useState<FrameLandmarks | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [lastResult, setLastResult] = useState<RecognitionResult | null>(null);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    recorderRef.current.reset();
    setCameraOn(false);
    setLiveFrame(null);
    setIsRecording(false);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();

      const landmarker = await getHandLandmarker();
      setCameraOn(true);

      const tick = () => {
        const currentVideo = videoRef.current;
        if (currentVideo && currentVideo.readyState >= 2) {
          const frame = detectFrame(landmarker, currentVideo, performance.now());
          setLiveFrame(frame);

          const recorder = recorderRef.current;
          const result = recorder.processFrame(frame, signsRef.current);
          setIsRecording(recorder.isRecording);

          if (result) {
            setLastResult(result);
            if (id && result.predictedId) {
              recordAttempt(id, result.predictedId === id, result.confidence);
            }
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : "Could not access the webcam.");
    }
  }, [id]);

  useEffect(() => stopCamera, [stopCamera]);

  return (
    <section>
      <p>
        <Link to="/">← Sign library</Link>
      </p>
      <h1>Webcam practice</h1>

      {id && targetSign ? (
        <p>
          Practising <strong>{targetSign.label}</strong>. Perform the sign in front of your webcam — recording
          starts automatically once your hand appears, and stops when your hand returns to rest.
        </p>
      ) : (
        <p>Free practice — perform any sign you've learned and see what the model recognises.</p>
      )}

      <p className="privacy-note">
        Your camera feed is processed entirely in your browser. No video is ever uploaded or stored.
      </p>

      <div className="camera-stage">
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video ref={videoRef} className="camera-video" muted playsInline />
        <div className="camera-overlay">
          <HandSkeletonCanvas
            frame={liveFrame}
            width={480}
            height={360}
            mirror
            recording={isRecording}
            transparentBackground
          />
        </div>
      </div>

      <div className="playback-controls">
        {!cameraOn ? (
          <button type="button" onClick={startCamera} disabled={loading}>
            Start camera
          </button>
        ) : (
          <button type="button" onClick={stopCamera}>
            Stop camera
          </button>
        )}
      </div>

      {cameraError && <p role="alert">{cameraError}</p>}
      {loading && <p>Loading reference signs…</p>}

      {lastResult && (
        <p aria-live="polite" className={lastResult.predictedId ? "result-ok" : "result-unknown"}>
          {lastResult.predictedId
            ? `Recognised: ${lastResult.predictedLabel} (confidence ${(lastResult.confidence * 100).toFixed(0)}%)`
            : "Not quite — try again."}
        </p>
      )}
    </section>
  );
}
