import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { loadReferenceSigns, type ReferenceSign } from "../lib/referenceData";

interface SignsContextValue {
  signs: ReferenceSign[];
  loading: boolean;
  error: string | null;
}

const SignsContext = createContext<SignsContextValue>({ signs: [], loading: true, error: null });

export function SignsProvider({ children }: { children: ReactNode }) {
  const [signs, setSigns] = useState<ReferenceSign[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadReferenceSigns()
      .then((loaded) => {
        if (!cancelled) setSigns(loaded);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo(() => ({ signs, loading, error }), [signs, loading, error]);

  return <SignsContext.Provider value={value}>{children}</SignsContext.Provider>;
}

export function useSigns(): SignsContextValue {
  return useContext(SignsContext);
}

export function useSign(id: string | undefined): ReferenceSign | undefined {
  const { signs } = useSigns();
  return useMemo(() => signs.find((s) => s.id === id), [signs, id]);
}
