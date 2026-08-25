const express = require("express");

const router = express.Router();

/**
 * POST /api/recognize -- STUB, not implemented.
 *
 * Reserves the interface contract for a future AI sign-recognition model,
 * which the team has not yet decided on. Landmarks are expected to be
 * pre-extracted client-side (same in-browser MediaPipe approach as the
 * `web/` prototype), so no video ever needs to reach the server.
 *
 * Expected request body once implemented:
 *   {
 *     targetSignId: number,
 *     landmarks: Array<{
 *       timestamp: number,
 *       hand: "left" | "right",
 *       points: Array<{ x: number, y: number, z: number }>
 *     }>
 *   }
 *
 * Expected response body once implemented:
 *   {
 *     status: "ok",
 *     targetSignId: number,
 *     predictedLabel: string,
 *     confidence: number, // 0-1
 *     isMatch: boolean
 *   }
 */
router.post("/", (req, res) => {
  res.status(501).json({
    status: "not_implemented",
    message:
      "Sign recognition is not implemented yet -- this endpoint reserves the contract for a future AI model.",
    expectedRequestShape: {
      targetSignId: "number",
      landmarks:
        'Array<{timestamp:number, hand:"left"|"right", points:Array<{x:number,y:number,z:number}>}>',
    },
    expectedResponseShapeWhenImplemented: {
      status: "ok",
      targetSignId: "number",
      predictedLabel: "string",
      confidence: "number (0-1)",
      isMatch: "boolean",
    },
  });
});

module.exports = router;
