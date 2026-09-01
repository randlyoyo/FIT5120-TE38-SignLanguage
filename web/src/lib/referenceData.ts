import { buildSignEmbedding, type SignEmbedding } from "./signModel";
import type { FrameLandmarks, SignManifestEntry, SignTemplate } from "./types";

export interface ReferenceSign extends SignEmbedding {
  id: string;
  label: string;
  frames: FrameLandmarks[];
}

/**
 * Loads every reference sign template from `public/data/signs/` and builds
 * its DTW-ready embedding once, up front. Equivalent to the reference repo's
 * `utils/dataset_utils.load_reference_signs`, but reading pre-computed JSON
 * (produced by `scripts/generate-placeholder-signs.mjs`) instead of pickling
 * landmarks extracted from local video files.
 */
export async function loadReferenceSigns(): Promise<ReferenceSign[]> {
  const manifestRes = await fetch("/data/signs/manifest.json");
  if (!manifestRes.ok) {
    throw new Error(`Failed to load sign manifest: ${manifestRes.status}`);
  }
  const manifest: SignManifestEntry[] = await manifestRes.json();

  return Promise.all(
    manifest.map(async (entry): Promise<ReferenceSign> => {
      const res = await fetch(`/data/signs/${entry.id}.json`);
      if (!res.ok) {
        throw new Error(`Failed to load sign template "${entry.id}": ${res.status}`);
      }
      const template: SignTemplate = await res.json();
      const embedding = buildSignEmbedding(template.frames);
      return {
        id: template.id,
        label: template.label,
        frames: template.frames,
        ...embedding,
      };
    }),
  );
}
