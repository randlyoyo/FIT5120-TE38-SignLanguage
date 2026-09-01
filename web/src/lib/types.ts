export interface Point3D {
  x: number;
  y: number;
  z: number;
}

/** 21 MediaPipe hand landmarks, or null if that hand was not detected in the frame. */
export type HandLandmarks = Point3D[] | null;

export interface FrameLandmarks {
  left: HandLandmarks;
  right: HandLandmarks;
}

export interface SignTemplate {
  id: string;
  label: string;
  frames: FrameLandmarks[];
}

export interface SignManifestEntry {
  id: string;
  label: string;
}
