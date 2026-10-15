import { useEffect, useRef, useState } from "react";
import type { SignVideo } from "../api/types";
import { fetchSignPose } from "../api/signs";
import { parseSmplxPose, type SmplxClip } from "../lib/smplx";
import { SmplxAvatar } from "./SmplxAvatar";

const SPEEDS = [0.5, 1, 2] as const;

interface Props {
  signId: number;
  gloss: string;
  videos: SignVideo[];
}

export function SignDemonstration({ signId, gloss, videos }: Props) {
  const [videoIndex, setVideoIndex] = useState(0);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [view, setView] = useState<"video" | "avatar">("video");
  // undefined = not fetched yet, null = this sign has no pose data.
  const [poseClip, setPoseClip] = useState<SmplxClip | null | undefined>(undefined);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    setPoseClip(undefined);
  }, [signId]);

  useEffect(() => {
    if (view !== "avatar" || poseClip !== undefined) return;
    const controller = new AbortController();
    fetchSignPose(signId, controller.signal)
      .then((pose) => setPoseClip(pose ? parseSmplxPose(pose) : null))
      .catch((err) => {
        if (controller.signal.aborted) return;
        console.error("Failed to load pose data:", err);
        setPoseClip(null);
      });
    return () => controller.abort();
  }, [view, signId, poseClip]);

  const speedControl = (
    <label>
      Speed{" "}
      <select
        value={speed}
        onChange={(e) => {
          const nextSpeed = Number(e.target.value) as (typeof SPEEDS)[number];
          setSpeed(nextSpeed);
          if (videoRef.current) {
            videoRef.current.playbackRate = nextSpeed;
          }
        }}
      >
        {SPEEDS.map((value) => (
          <option key={value} value={value}>
            {value}×
          </option>
        ))}
      </select>
    </label>
  );

  const availableVideos = videos.filter((video) => Boolean(video.videoUrl));

  const currentVideo = availableVideos[videoIndex];

  useEffect(() => {
    if (videoIndex >= availableVideos.length) {
      setVideoIndex(0);
    }
  }, [availableVideos.length, videoIndex]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  }, [speed, videoIndex]);

  return (
    <div className="sign-demo">
      {/* One segmented bar, same control as Learn / Practice. */}
      <div className="detail-mode-toggle sign-demo-view-toggle" role="tablist" aria-label="Video or 3D avatar">
        {(["video", "avatar"] as const).map((v) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            className={`detail-mode-tab ${view === v ? "active" : ""}`}
            onClick={() => setView(v)}
          >
            {v === "video" ? "Video" : "3D avatar"}
          </button>
        ))}
      </div>

      {view === "avatar" ? (
        poseClip === null ? (
          <p className="demo-empty">
            There’s no 3D avatar for this sign yet — please follow the demonstration video.
          </p>
        ) : (
          <>
            {/* While the motion is still loading (undefined) the avatar waits in its idle pose. */}
            <SmplxAvatar className="sign-demo-avatar" clip={poseClip ?? null} playing loop speed={speed} />
            <p className="avatar-reference-note">
              The 3D avatar is a guide to the overall movement. For exact handshapes and finger
              positions, follow the demonstration video.
            </p>
            <div className="playback-controls" role="group" aria-label="3D avatar playback controls">
              {speedControl}
            </div>
          </>
        )
      ) : availableVideos.length === 0 ? (
        <p className="demo-empty">
          No demonstration video is available for this sign yet.
        </p>
      ) : (
        <>
      <div className="sign-demo-stage">
        <video
          ref={videoRef}
          key={currentVideo.videoUrl ?? currentVideo.fileName}
          controls
          loop
          playsInline
          preload="metadata"
          className="sign-demo-video"
          aria-label={`${gloss} Auslan demonstration`}
        >
          <source
            src={currentVideo.videoUrl ?? undefined}
            type="video/mp4"
          />
          Your browser does not support video playback.
        </video>
      </div>

      {availableVideos.length > 1 && (
        <div
          className="video-variants"
          role="group"
          aria-label="Available sign demonstrations"
        >
          {availableVideos.map((video, index) => (
            <button
              key={video.sourceId}
              type="button"
              onClick={() => setVideoIndex(index)}
              className={videoIndex === index ? "active" : ""}
            >
              Version {index + 1}
            </button>
          ))}
        </div>
      )}

      <div
        className="playback-controls"
        role="group"
        aria-label="Video playback controls"
      >
        <button
          type="button"
          onClick={() => {
            if (!videoRef.current) return;
            videoRef.current.currentTime = 0;
            videoRef.current.play();
          }}
        >
          ↶ Replay
        </button>

        {speedControl}
      </div>
        </>
      )}
    </div>
  );
}