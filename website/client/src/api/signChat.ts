// Client for the sign-chat backend (sign_chat/sign_chat_backend). Contract:
// sign_chat/sign_chat_backend/README.md "API". Runs on a separate GPU host,
// not this site's own Express server -- CORS is open there ("*"), so the
// browser calls it directly.

import type { SmplxPoseJson } from "../lib/smplx";

const API_BASE = import.meta.env.VITE_SIGNCHAT_API_BASE_URL ?? "http://localhost:8000";

export interface SignSubtitleCue {
  start: number;
  end: number;
  text: string;
}

export interface ChatReplySign {
  pose_url: string;
  video_url: string;
  video_status: "rendering" | "ready";
  subtitles: SignSubtitleCue[];
  subtitle_url: string;
  frames: number;
  fps: number;
  retrieved: string;
  seen: boolean;
}

export interface ChatResponse {
  session_id: string;
  input: {
    mode: "sign" | "text";
    text: string;
    recognition?: { raw_text: string; frames: number; signer_visible: number };
  };
  reply: {
    text: string;
    sign: ChatReplySign;
  };
}

export class SignChatApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Relative response URLs (`/media/...`) resolved against the backend host. */
export function signChatMediaUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

async function handle<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const message = data?.detail ?? `Sign-chat request failed (${res.status})`;
    throw new SignChatApiError(res.status, message);
  }
  return data as T;
}

export async function sendChatText(text: string, sessionId?: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, session_id: sessionId }),
  });
  return handle<ChatResponse>(res);
}

export async function sendChatSign(
  video: Blob,
  options: { mirrored?: boolean; sessionId?: string; filename?: string } = {}
): Promise<ChatResponse> {
  const form = new FormData();
  // The backend validates video type by this filename's extension
  // (server.py's VIDEO_TYPES check) -- an uploaded file keeps its own name
  // so e.g. a .mov isn't mislabeled as the recorder's default .webm.
  form.append("video", video, options.filename ?? "recording.webm");
  if (options.sessionId) form.append("session_id", options.sessionId);
  form.append("mirrored", String(options.mirrored ?? true));
  const res = await fetch(`${API_BASE}/api/chat/sign`, { method: "POST", body: form });
  return handle<ChatResponse>(res);
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
}

/** Polls `video_url` until it's done rendering (the backend returns 404 for
 *  ~1s after a reply while the mp4 is written). */
export async function waitForSignVideo(path: string, { tries = 6, intervalMs = 400 } = {}): Promise<void> {
  const url = signChatMediaUrl(path);
  for (let i = 0; i < tries; i++) {
    const res = await fetch(url, { method: "HEAD" });
    if (res.ok) return;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/** Fetches the pose JSON at `pose_url` -- the keypoints-only reply, for
 *  driving a real 3D avatar (SmplxAvatar) instead of playing `video_url`. */
export async function fetchSignPose(path: string): Promise<SmplxPoseJson> {
  const res = await fetch(signChatMediaUrl(path));
  if (!res.ok) throw new SignChatApiError(res.status, `Couldn't load pose data (${res.status})`);
  return res.json();
}
