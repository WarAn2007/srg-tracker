import type { AppSettings } from "./types";

export const DEFAULT_SETTINGS: AppSettings = {
  colors: ["#9ae6b4", "#28a96b", "#67e8f9"],
  backgroundColor: "#0a2a22",
  movingBackground: true,
};

const STORAGE_KEY = "srg-v4-settings";

export function loadSettings(): AppSettings {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as Partial<AppSettings> | null;
    if (!stored || !Array.isArray(stored.colors) || stored.colors.length !== 3) return DEFAULT_SETTINGS;
    return {
      colors: stored.colors as [string, string, string],
      backgroundColor: stored.backgroundColor ?? DEFAULT_SETTINGS.backgroundColor,
      movingBackground: stored.movingBackground ?? DEFAULT_SETTINGS.movingBackground,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: AppSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}
