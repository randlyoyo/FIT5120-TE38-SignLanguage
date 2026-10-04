// SignAvatar: the sign-chat avatar drawn on a <canvas>, like a person on the other end of a video call.
//
//   import { SignAvatar } from './avatar.js';
//   const avatar = new SignAvatar(canvas, {
//     api: 'https://xxxx.trycloudflare.com',          // the backend; '' = same origin
//     onCue: text => { subtitle.textContent = text; }, // the reply's subtitle at this moment ('' = none)
//   });
//   await avatar.load();                      // GET /api/avatar: from now on the avatar stands, hands down, breathing
//   await avatar.play(reply.sign.pose_url);   // signs the reply, then is back at rest when the promise resolves
//
// There is no <video> and nothing to press: between replies the avatar idles, and play() makes it sign.
// Every reply already starts and ends in the rest pose (the server adds the transitions), and the
// camera is the same for every reply, so one reply follows another without a jump.
//
// The drawing is vector, so it is sharp at any canvas size. Size the canvas with CSS; the avatar is
// drawn in the largest centred square that fits, on the rig's background colour.
//
// Data (see the backend README, "Drawing the avatar"):
//   GET /api/avatar -> { fps, bones: [{a, b, color, width}], points: {joints, color, radius},
//                        background, rest: [[x, y] x 127], idle: [[[x, y] x 127] x N] }
//   pose JSON (reply.sign.pose_url) -> { fps, joints2d: [[[x, y] x 127] x T], subtitles: [{start, end, text}] }
//   x, y are in [0, 1] of the square, y down; widths and radii are fractions of the square's side.

export class SignAvatar {
  constructor(canvas, { api = '', onCue = null, onState = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.api = api.replace(/\/$/, '');
    this.onCue = onCue;          // (text) => void
    this.onState = onState;      // ('idle' | 'signing') => void
    this.rig = null;
    this.clip = null;
    this._ready = null;
    this._raf = 0;
    this._cueText = null;
  }

  // Fetch the rig and start drawing. Safe to call once; play() waits for it.
  load() {
    if (!this._ready) {
      this._ready = fetch(this._url('/api/avatar')).then(async r => {
        if (!r.ok) throw new Error(`GET /api/avatar: HTTP ${r.status}`);
        this.rig = await r.json();
        this._idleStart = performance.now();
        this._state('idle');
        const step = now => {
          this._draw(this._pose(now));
          this._raf = requestAnimationFrame(step);
        };
        this._raf = requestAnimationFrame(step);
      });
      this._ready.catch(() => { this._ready = null; });     // allow a retry
    }
    return this._ready;
  }

  // Sign one reply: a pose URL (reply.sign.pose_url) or the pose JSON itself. Resolves true when the
  // signing finished, false when it was interrupted (stop(), or another play()).
  async play(pose) {
    await this.load();
    if (typeof pose === 'string') {
      const r = await fetch(this._url(pose));
      if (!r.ok) throw new Error(`pose file: HTTP ${r.status}`);
      pose = await r.json();
    }
    if (!pose.joints2d || !pose.joints2d.length) throw new Error('pose file has no joints2d');
    this._finish(false);
    return new Promise(resolve => {
      this.clip = { frames: pose.joints2d, cues: pose.subtitles || [], fps: pose.fps || this.rig.fps,
                    start: performance.now(), resolve };
      this._state('signing');
    });
  }

  // Back to rest at once.
  stop() { this._finish(false); }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._finish(false);
  }

  // ------------------------------------------------------------------ internals
  _url(u) { return /^https?:/.test(u) ? u : this.api + u; }

  _state(s) { if (this.onState) this.onState(s); }

  _cue(text) {
    if (text === this._cueText) return;
    this._cueText = text;
    if (this.onCue) this.onCue(text);
  }

  _finish(done) {
    const c = this.clip;
    if (!c) return;
    this.clip = null;
    this._idleStart = performance.now();     // the clip ends at rest = the idle loop's first frame
    this._cue('');
    this._state('idle');
    c.resolve(done);
  }

  // The pose to draw now: the reply's frame, or the idle loop's. Frames are blended linearly, so the
  // motion stays smooth on screens faster than 25 fps.
  _pose(now) {
    const c = this.clip;
    if (c) {
      const t = (now - c.start) / 1000, f = t * c.fps;
      if (f < c.frames.length - 1) {
        const cue = c.cues.find(q => t >= q.start && t < q.end);
        this._cue(cue ? cue.text : '');
        return blend(c.frames, f, false);
      }
      this._finish(true);
    }
    const idle = this.rig.idle;
    return blend(idle, ((now - this._idleStart) / 1000 * this.rig.fps) % idle.length, true);
  }

  _draw(P) {
    const cv = this.canvas, ctx = this.ctx, rig = this.rig;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.fillStyle = rig.background;
    ctx.fillRect(0, 0, w, h);
    const side = Math.min(w, h), ox = (w - side) / 2, oy = (h - side) / 2;
    const X = j => ox + P[j][0] * side, Y = j => oy + P[j][1] * side;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const b of rig.bones) {
      ctx.strokeStyle = b.color;
      ctx.lineWidth = b.width * side;
      ctx.beginPath();
      ctx.moveTo(X(b.a), Y(b.a));
      ctx.lineTo(X(b.b), Y(b.b));
      ctx.stroke();
    }
    const pts = rig.points;
    ctx.fillStyle = pts.color;
    const r = pts.radius * side;
    for (const j of pts.joints) {
      ctx.beginPath();
      ctx.arc(X(j), Y(j), r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function blend(frames, f, loop) {
  const n = frames.length, i = Math.floor(f), k = f - i;
  const a = frames[i % n], b = frames[loop ? (i + 1) % n : Math.min(i + 1, n - 1)];
  if (k === 0 || a === b) return a;
  return a.map((p, j) => [p[0] + (b[j][0] - p[0]) * k, p[1] + (b[j][1] - p[1]) * k]);
}
