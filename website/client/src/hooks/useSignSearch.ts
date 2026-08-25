import { useEffect, useState } from "react";
import { fetchSigns } from "../api/signs";
import type { SignsResponse } from "../api/types";

interface Params {
  query: string;
  page: number;
}

export function useSignSearch({ query, page }: Params) {
  const [data, setData] = useState<SignsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoading(true);
    setIsError(false);

    fetchSigns({ query, page, signal: controller.signal })
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
  }, [query, page]);

  return { data, isLoading, isError };
}
