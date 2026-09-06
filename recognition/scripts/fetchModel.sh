#!/bin/sh
# Download the pinned MediaPipe Holistic bundle used to extract the training
# keypoints. A different bundle silently shifts the landmark distribution --
# see API.md section 2.
set -e
cd "$(dirname "$0")/../models"
URL="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
EXPECT=e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
curl -fSL --progress-bar -o holistic_landmarker.task "$URL"
GOT=$(shasum -a 256 holistic_landmarker.task | cut -d' ' -f1)
if [ "$GOT" != "$EXPECT" ]; then
  echo "HASH MISMATCH -- do not use this bundle"
  echo "  expected $EXPECT"
  echo "  got      $GOT"
  echo "Google has replaced the pinned artefact. Recover the copy used for"
  echo "training rather than proceeding; a different bundle costs accuracy"
  echo "without raising an error."
  exit 1
fi
ls -lh holistic_landmarker.task
echo "sha256 OK: $GOT"
