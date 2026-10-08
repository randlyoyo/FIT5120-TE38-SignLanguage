import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { fetchSignById } from "../api/signs";
import { isLearned, toggleLearned } from "../lib/learnedSigns";
import type { Sign } from "../api/types";

interface WordPreviewModalProps {
  signId: number | null;
  word: string;
  onClose: () => void;
  onLoadingStateChange?: (isLoaded: boolean) => void;
}

export function WordPreviewModal({ signId, word, onClose, onLoadingStateChange }: WordPreviewModalProps) {
  const [sign, setSign] = useState<Sign | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isWordLearned, setIsWordLearned] = useState(false);
  const isLoaded = sign !== null && !loading;

  useEffect(() => {
    if (signId === null || signId === undefined) {
      setSign(null);
      return;
    }

    setLoading(true);
    setError(null);
    setSign(null);
    setIsWordLearned(isLearned(signId));

    const controller = new AbortController();
    fetchSignById(signId, controller.signal)
      .then((data) => {
        setSign(data);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error("Failed to load sign:", err);
          setError("Failed to load sign");
        }
      })
      .finally(() => {
        setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [signId]);

  // Notify parent when loading state changes
  useEffect(() => {
    onLoadingStateChange?.(isLoaded);
  }, [isLoaded, onLoadingStateChange]);

  if (signId === null || signId === undefined) return null;

  const videoUrl = sign?.previewVideo?.videoUrl || sign?.videos?.[0]?.videoUrl;

  const handleModalMouseLeave = () => {
    // Only close if video has finished loading
    if (isLoaded) {
      onClose();
    }
  };

  const handleToggleLearned = () => {
    if (signId !== null && signId !== undefined) {
      const newState = toggleLearned(signId);
      setIsWordLearned(newState);
    }
  };

  const content = (
    <div className="word-preview-overlay" onClick={onClose}>
      <div 
        className="word-preview-modal" 
        onClick={(e) => e.stopPropagation()}
        onMouseLeave={handleModalMouseLeave}
      >
        <div className="word-preview-header">
          <h2 className="word-preview-title">{word}</h2>
        </div>

        <div className="word-preview-content">
          {loading && (
            <div className="word-preview-loading">
              <div className="spinner"></div>
              <p>Loading sign...</p>
            </div>
          )}

          {error && (
            <div className="word-preview-error">
              <p>{error}</p>
            </div>
          )}

          {sign && !loading && (
            <>
              {videoUrl ? (
                <video
                  className="word-preview-video"
                  src={videoUrl}
                  controls
                  autoPlay
                  loop
                  playsInline
                />
              ) : (
                <div className="word-preview-no-video">
                  <p>No video available for this sign</p>
                </div>
              )}

              <div className="word-preview-bottom-section">
                <div>
                  {sign.definitions && sign.definitions.length > 0 && (
                    <div className="word-preview-definitions">
                      <h3>Definitions</h3>
                      {sign.definitions.map((group, idx) => (
                        <div key={idx} className="definition-group">
                          <p className="definition-pos">{group.partOfSpeech}</p>
                          <ul>
                            {group.senses.map((sense, senseIdx) => (
                              <li key={senseIdx}>{sense}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  className={`word-preview-learn-btn ${isWordLearned ? "learned" : ""}`}
                  onClick={handleToggleLearned}
                  aria-label={isWordLearned ? "Mark as not learned" : "Mark as learned"}
                  title={isWordLearned ? "Unmark as learned" : "Mark as learned"}
                >
                  {isWordLearned ? "✓ Learned" : "Mark as Learned"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
