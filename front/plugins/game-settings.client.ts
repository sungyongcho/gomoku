import type { Settings } from "~/types/game";
import {
  defaultSettings,
  GOMOKU_SETTINGS_KEY,
} from "~/stores/game.store";

export default defineNuxtPlugin(() => {
  try {
    const raw = localStorage.getItem(GOMOKU_SETTINGS_KEY);
    if (!raw) return;

    const parsed = JSON.parse(raw) as Partial<Settings>;
    if (typeof parsed !== "object" || parsed === null) return;

    const store = useGameStore();
    store.settings = { ...defaultSettings, ...parsed };
  } catch {
    // ignore invalid or missing data
  }
});
