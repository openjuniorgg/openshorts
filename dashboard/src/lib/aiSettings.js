// The model choice the user makes in Settings. It lives in this browser only, and nothing
// is preselected: the provider stays empty until the user picks one.
import { useEffect, useState } from 'react';

const STORAGE_KEY = 'ai_settings_v1';
const CHANGED = 'os:ai-settings';
const EMPTY = { provider: '', openrouterModel: '', ollamaModel: '' };

export function getAiSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...EMPTY, ...JSON.parse(raw) } : { ...EMPTY };
  } catch (_) {
    return { ...EMPTY };
  }
}

export function setAiSettings(patch) {
  const next = { ...getAiSettings(), ...patch };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch (_) { /* storage blocked: the choice then lasts for this page only */ }
  window.dispatchEvent(new CustomEvent(CHANGED, { detail: next }));
  return next;
}

export function useAiSettings() {
  const [settings, setSettings] = useState(getAiSettings);
  useEffect(() => {
    const onChange = (e) => setSettings(e.detail);
    window.addEventListener(CHANGED, onChange);
    return () => window.removeEventListener(CHANGED, onChange);
  }, []);
  return settings;
}

// The model for the provider the user picked.
export function chosenModel(settings) {
  return settings.provider === 'ollama' ? settings.ollamaModel : settings.openrouterModel;
}

// The choice as request headers: only what the user actually picked.
export function aiHeaders(settings = getAiSettings()) {
  const headers = {};
  if (settings.provider) headers['X-AI-Provider'] = settings.provider;
  const model = chosenModel(settings);
  if (model) headers['X-AI-Model'] = model;
  return headers;
}

// True when a request can run: a provider, its model, and for OpenRouter a key (typed
// here, or set on the server).
export function isChoiceReady(settings, { hasKey = false, keyOnServer = false } = {}) {
  if (settings.provider === 'ollama') return !!settings.ollamaModel;
  if (settings.provider === 'openrouter') return !!settings.openrouterModel && (hasKey || keyOnServer);
  return false;
}

// What one job's model calls cost: the video's analysis plus each clip's own calls.
// ``share`` is the analysis split evenly over the clips: a shown figure, not a meter.
export function jobAiCost(results) {
  const clips = results?.clips || [];
  const analysis = Number(results?.cost_analysis?.total_cost) || 0;
  const own = clips.reduce((sum, c) => sum + (Number(c?.ai_cost?.total_cost) || 0), 0);
  return { analysis, own, total: analysis + own, share: clips.length ? analysis / clips.length : 0 };
}
