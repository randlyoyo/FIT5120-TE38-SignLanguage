import { useEffect, useRef, useState } from "react";
import type { SignVideo } from "../api/types";
import { fetchSignPose } from "../api/signs";
import { parseSmplxPose, type SmplxClip } from "../lib/smplx";
import { SmplxAvatar, smplxAvatarConfigured } from "./SmplxAvatar";
import { StickFigureStage } from "./StickFigureStage";

const SPEEDS = [0.5, 1, 2] as const;

interface Props {
  signId: number;
  gloss: string;
  videos: SignVideo[];
}

export function SignDemonstration({ signId, gloss, videos }: Props) {
  const [videoIndex, setVideoIndex] = useState(0);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [view, setView] = useState<"video" | "skeleton">("video");
  // undefined = not fetched yet, null = this sign has no pose data.
  const [poseClip, setPoseClip] = useState<SmplxClip | null | undefined>(undefined);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    setPoseClip(undefined);
  }, [signId]);

  useEffect(() => {
    if (view !== "skeleton" || poseClip !== undefined) return;
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
      <div className="video-variants sign-demo-view-toggle" role="group" aria-label="Video or AI skeleton preview">
        <button type="button" className={view === "video" ? "active" : ""} onClick={() => setView("video")}>
          Video
        </button>
        <button type="button" className={view === "skeleton" ? "active" : ""} onClick={() => setView("skeleton")}>
          Skeleton preview
        </button>
      </div>

      {view === "skeleton" ? (
        poseClip && smplxAvatarConfigured ? (
          <>
            <SmplxAvatar className="sign-demo-avatar" clip={poseClip} playing loop speed={speed} />
            <div className="playback-controls" role="group" aria-label="Skeleton playback controls">
              {speedControl}
            </div>
          </>
        ) : (
          <StickFigureStage
            status={poseClip === undefined ? "Loading…" : undefined}
            caption={
              poseClip === undefined
                ? "Loading the motion-capture preview…"
                : "No motion-capture preview for this sign yet."
            }
          />
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

      {/* Mirror view, not flipped footage -- the camera faces the signer, so
          on screen it works exactly like a bathroom mirror: whichever hand
          appears on the video's right side is the signer's actual left
          hand. Worth spelling out because it's the opposite of how a
          "selfie" video usually gets described, and getting it backwards
          means copying the sign with the wrong hand. Styled to stand out
          (not a quiet footnote) since missing it means practicing every
          sign with the wrong hand. */}
      <p className="mirror-view-hint">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          width={18}
          height={18}
          aria-hidden="true"
        >
          <path d="M12 3.5 21.5 20H2.5z" />
          <path d="M12 9.5v5" />
          <circle cx="12" cy="17.25" r="0.75" fill="currentColor" stroke="none" />
        </svg>
        <span className="mirror-view-hint-title">Before learning or practising:</span>
        <span className="mirror-view-hint-text">Auslan can be performed by both left- and right-handed users, but our learning and recognition system requires users to follow the hand orientation demonstrated in the instructional video.</span>
      </p>

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