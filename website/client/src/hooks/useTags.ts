import { useEffect, useState } from "react";
import { fetchTags } from "../api/signs";
import type { TagCount } from "../api/types";

/** Distinct tag categories with counts, for the library page's category rail. */
export function useTags() {
  const [tags, setTags] = useState<TagCount[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    fetchTags(controller.signal)
      .then(setTags)
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  return tags;
}
