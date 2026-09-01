import { HandLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef } from "react";
import type { FrameLandmarks, HandLandmarks } from "../lib/types";

const CONNECTIONS: [number, number][] = HandLandmarker.HAND_CONNECTIONS.map((c) => [c.start, c.end]);

interface Props {
  frame: FrameLandmarks | null;
  width?: number;
  height?: number;
  mirror?: boolean;
  /** Draws a small filled circle in the corner while true (used to show live recording state). */
  recording?: boolean;
  /** Skip painting an opaque background — use when overlaid on top of a live video element. */
  transparentBackground?: boolean;
}

export function HandSkeletonCanvas({
  frame,
  width = 320,
  height = 240,
  mirror = false,
  recording = false,
  transparentBackground = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, width, height);
    if (!transparentBackground) {
      ctx.fillStyle = "#11151c";
      ctx.fillRect(0, 0, width, height);
    }

    const drawHand = (hand: HandLandmarks, color: string) => {
      if (!hand) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [a, b] of CONNECTIONS) {
        const p1 = hand[a];
        const p2 = hand[b];
        const x1 = (mirror ? 1 - p1.x : p1.x) * width;
        const y1 = p1.y * height;
        const x2 = (mirror ? 1 - p2.x : p2.x) * width;
        const y2 = p2.y * height;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      ctx.fillStyle = "#f5f2e2";
      for (const p of hand) {
        const x = (mirror ? 1 - p.x : p.x) * width;
        const y = p.y * height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    if (frame) {
      drawHand(frame.left, "#ffcc66");
      drawHand(frame.right, "#66d9ff");
    }

    if (recording) {
      ctx.fillStyle = "#f0233c";
      ctx.beginPath();
      ctx.arc(16, 16, 7, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [frame, width, height, mirror, recording, transparentBackground]);

  return <canvas ref={canvasRef} width={width} height={height} className="skeleton-canvas" />;
}
