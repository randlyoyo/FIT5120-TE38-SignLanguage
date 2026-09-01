import { useEffect, useState } from "react";
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type AccessibilitySettings } from "../lib/storage";

const MIN_SCALE = 0.85;
const MAX_SCALE = 1.5;
const SCALE_STEP = 0.1;

export function AccessibilityBar() {
  const [settings, setSettings] = useState<AccessibilitySettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty("--font-scale", String(settings.textScale));
    root.classList.toggle("high-contrast", settings.highContrast);
    root.classList.toggle("reduce-motion", settings.reduceMotion);
    saveSettings(settings);
  }, [settings]);

  const update = (patch: Partial<AccessibilitySettings>) => setSettings((prev) => ({ ...prev, ...patch }));

  return (
    <div className="accessibility-bar" role="group" aria-label="Accessibility settings">
      <button
        type="button"
        onClick={() => update({ textScale: Math.max(MIN_SCALE, settings.textScale - SCALE_STEP) })}
        aria-label="Decrease text size"
      >
        A-
      </button>
      <button
        type="button"
        onClick={() => update({ textScale: Math.min(MAX_SCALE, settings.textScale + SCALE_STEP) })}
        aria-label="Increase text size"
      >
        A+
      </button>
      <label>
        <input
          type="checkbox"
          checked={settings.highContrast}
          onChange={(e) => update({ highContrast: e.target.checked })}
        />
        High contrast
      </label>
      <label>
        <input
          type="checkbox"
          checked={settings.reduceMotion}
          onChange={(e) => update({ reduceMotion: e.target.checked })}
        />
        Reduce motion
      </label>
    </div>
  );
}
