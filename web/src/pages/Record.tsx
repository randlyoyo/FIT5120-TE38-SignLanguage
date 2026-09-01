import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { HandSkeletonCanvas } from "../components/HandSkeletonCanvas";
import { detectFrame, getHandLandmarker } from "../lib/mediapipeHands";
import type { FrameLandmarks, SignTemplate } from "../lib/types";

const VIDEO_CONSTRAINTS: MediaStreamConstraints = {
  video: { width: 480, height: 360, facingMode: "user" },
  audio: false,
};

/**
 * Dev-only tool: the in-browser replacement for the reference repo's
 * `save_landmarks_from_video` + `data/videos` workflow. Lets the team record
 * their own consented webcam footage and export a landmark-only JSON
 * template — no raw video is ever produced or stored, only the extracted
 * hand landmarks.
 */
export function Record() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const bufferRef = useRef<FrameLandmarks[]>([]);

  const [cameraOn, setCameraOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [liveFrame, setLiveFrame] = useState<FrameLandmarks | null>(null);
  const [capturedFrames, setCapturedFrames] = useState<FrameLandmarks[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [signId, setSignId] = useState("new-sign");
  const [label, setLabel] = useState("New Sign");

  // Read inside the rAF loop without re-subscribing startCamera on every toggle.
  const recordingRef = useRef(false);
  recordingRef.current = recording;

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setRecording(false);
    setLiveFrame(null);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
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
          if (recordingRef.current) {
            bufferRef.current.push(frame);
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not access the webcam.");
    }
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startRecording = () => {
    bufferRef.current = [];
    setCapturedFrames(null);
    setRecording(true);
  };

  const stopRecording = () => {
    setRecording(false);
    setCapturedFrames([...bufferRef.current]);
  };

  const download = () => {
    if (!capturedFrames) return;
    const template: SignTemplate = { id: signId, label, frames: capturedFrames };
    const blob = new Blob([JSON.stringify(template)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${signId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section>
      <p>
        <Link to="/">← Sign library</Link>
      </p>
      <h1>Record a reference sign (dev tool)</h1>
      <p>
        Capture your own consented webcam footage as a landmark template. Nothing is uploaded — this produces a
        JSON file of hand landmarks only, which you can drop into <code>public/data/signs/</code> and add to{" "}
        <code>manifest.json</code>.
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
            recording={recording}
            transparentBackground
          />
        </div>
      </div>

      <div className="playback-controls">
        {!cameraOn ? (
          <button type="button" onClick={startCamera}>
            Start camera
          </button>
        ) : (
          <button type="button" onClick={stopCamera}>
            Stop camera
          </button>
        )}
        {cameraOn && !recording && (
          <button type="button" onClick={startRecording}>
            Start recording
          </button>
        )}
        {cameraOn && recording && (
          <button type="button" onClick={stopRecording}>
            Stop recording
          </button>
        )}
      </div>

      {error && <p role="alert">{error}</p>}

      {capturedFrames && (
        <div className="record-export">
          <p>Captured {capturedFrames.length} frames.</p>
          <label>
            Sign id (filename)
            <input value={signId} onChange={(e) => setSignId(e.target.value)} />
          </label>
          <label>
            Label
            <input value={label} onChange={(e) => setLabel(e.target.value)} />
          </label>
          <button type="button" onClick={download}>
            Download JSON
          </button>
        </div>
      )}
    </section>
  );
}
