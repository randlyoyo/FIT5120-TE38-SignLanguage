const express = require("express");

const { UnusableCapture } = require("../recognition/features");
const {
  load, embed, distanceToWord, distanceToAll, confidenceFromMargin,
} = require("../recognition/model");

const router = express.Router();

const TOP_K = 4;

/**
 * Landmarks are extracted in the browser by MediaPipe and posted here as
 * numbers; video never leaves the device. The feature pipeline and the encoder
 * run server-side so there is exactly one implementation to keep in step with
 * the training code, and so the model can be replaced without shipping a new
 * client. Contract: recognition/API.md.
 */

function readCapture(body) {
  const c = body?.capture;
  if (!c) throw new UnusableCapture("missing `capture`");
  const { width, height, fps, pose, left_hand: l, right_hand: r } = c;
  if (![width, height, fps].every((v) => typeof v === "number" && v > 0)) {
    throw new UnusableCapture("capture needs numeric width, height and fps");
  }
  if (!Array.isArray(pose) || !Array.isArray(l) || !Array.isArray(r)) {
    throw new UnusableCapture("capture needs pose, left_hand and right_hand arrays");
  }
  return c;
}

/** Turn a thrown error into the right status: a bad capture is the caller's
 *  problem (400), missing model assets are ours (503). */
function fail(err, res, next) {
  if (err instanceof UnusableCapture) {
    return res.status(400).json({ error: "unusable_capture", detail: err.message });
  }
  if (err.code === "ENOENT") {
    return res.status(503).json({
      error: "model_unavailable",
      detail: "recognition assets are not deployed -- see recognition/ASSETS.md",
    });
  }
  return next(err);
}

// POST /api/recognize/verify  { word, capture } -> did this attempt match?
// The learner already chose the word, so this compares against that word only.
// This is the reliable mode: EER 0.73% on the validation split.
router.post("/verify", async (req, res, next) => {
  try {
    const s = await load();
    const capture = readCapture(req.body);
    const word = String(req.body?.word ?? "").trim();
    const idx = s.wordIndex.get(word);
    if (idx === undefined) {
      return res.status(404).json({ error: "unknown_word", word });
    }

    const { embedding, frames } = await embed(capture);
    const distance = distanceToWord(s, embedding, idx);

    res.json({
      word,
      distance,
      threshold: s.tauVerify,
      matched: distance < s.tauVerify,
      frames,
    });
  } catch (err) {
    fail(err, res, next);
  }
});

// POST /api/recognize/identify  { capture } -> top-4 candidates
// No target word. Always returns four candidates rather than one answer:
// measured top-1 is 82-89% but top-4 is 93-98%, so letting the learner pick
// from a short list is far more useful than asserting a single guess.
router.post("/identify", async (req, res, next) => {
  try {
    const s = await load();
    const capture = readCapture(req.body);
    const { embedding, frames } = await embed(capture);

    const dist = distanceToAll(s, embedding);
    const order = Array.from(dist.keys()).sort((a, b) => dist[a] - dist[b]);
    const top = order.slice(0, TOP_K);
    const margin = dist[order[1]] - dist[order[0]];

    res.json({
      candidates: top.map((i) => ({ word: s.words[i], distance: dist[i] })),
      // Calibrated: of the captures scored at 0.85, about 85% really do have
      // the right word first. Fitted on the validation split, so it inherits
      // that split's conditions -- see API.md section 9.
      confidence: confidenceFromMargin(s, margin),
      margin,
      // Advisory only. Low margin does NOT mean the list is wrong: among
      // low-margin captures the correct word is still in the top four 79-91%
      // of the time, so never hide the candidates -- only soften how the first
      // one is presented.
      confident: margin >= s.marginConfident,
      marginThreshold: s.marginConfident,
      frames,
    });
  } catch (err) {
    fail(err, res, next);
  }
});

// GET /api/recognize/vocabulary -- which glosses can be recognised at all
router.get("/vocabulary", async (req, res, next) => {
  try {
    const s = await load();
    res.json({ count: s.words.length, words: s.words });
  } catch (err) {
    fail(err, res, next);
  }
});

// The original stub posted to the router root. Keep a pointer so an older
// client gets told where to go rather than a bare 404.
router.post("/", (req, res) => {
  res.status(410).json({
    error: "moved",
    detail: "use POST /api/recognize/verify or POST /api/recognize/identify",
  });
});

module.exports = router;
