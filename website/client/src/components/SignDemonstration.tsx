import { useEffect, useRef, useState } from "react";
import type { SignVideo } from "../api/types";

const SPEEDS = [0.5, 1, 2] as const;

interface Props {
  gloss: string;
  videos: SignVideo[];
}

export function SignDemonstration({ gloss, videos }: Props) {
  const [videoIndex, setVideoIndex] = useState(0);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const videoRef = useRef<HTMLVideoElement | null>(null);

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

  if (availableVideos.length === 0) {
    return (
      <div className="sign-demo">
        <p className="demo-empty">
          No demonstration video is available for this sign yet.
        </p>
      </div>
    );
  }

  return (
    <div className="sign-demo">
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
        Mirrored view: the hand shown on the right is the signer's left
        hand.
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

        <label>
          Speed{" "}
          <select
            value={speed}
            onChange={(e) => {
              const nextSpeed = Number(e.target.value) as
                (typeof SPEEDS)[number];

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
      </div>
    </div>
  );
}