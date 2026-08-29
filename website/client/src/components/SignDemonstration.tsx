import { useEffect, useRef, useState } from "react";
import type { SignVideo } from "../api/types";

const SPEEDS = [0.5, 1, 2] as const;

interface Props {
  gloss: string;
  steps: string[];
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