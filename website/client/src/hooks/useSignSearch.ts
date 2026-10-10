import { useEffect, useState } from "react";
import { fetchSigns, type SignSort } from "../api/signs";
import type { SignLevel, SignsResponse } from "../api/types";

interface Params {
  query: string;
  tag: string;
  level: SignLevel | "";
  sort: SignSort | "";
  page: number;
}

export function useSignSearch({ query, tag, level, sort, page }: Params) {
  const [data, setData] = useState<SignsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoading(true);
    setIsError(false);

    fetchSigns({
      query,
      tag,
      level: level || undefined,
      sort: sort || undefined,
      page,
      signal: controller.signal,
    })
      .then(setData)
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error(err);
          setIsError(true);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false);
      });

    return () => controller.abort();
  }, [query, tag, level, sort, page]);

  return { data, isLoading, isError };
}
