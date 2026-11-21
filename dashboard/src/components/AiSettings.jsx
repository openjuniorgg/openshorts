import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Cloud, Cpu, RefreshCw } from 'lucide-react';
import KeyInput from './KeyInput';
import { apiJson } from '../lib/api';
import { setAiSettings, useAiSettings } from '../lib/aiSettings';

const PROVIDERS = [
  { id: 'openrouter', label: 'OpenRouter', icon: <Cloud size={16} />,
    note: 'Models in the cloud, through an OpenRouter key.' },
  { id: 'ollama', label: 'Ollama', icon: <Cpu size={16} />,
    note: 'Models already downloaded on this machine. No key, no model cost.' },
];

function Panel({ children }) {
  return <div className="card p-4 sm:p-6 mb-6 animate-fade">{children}</div>;
}

function Heading({ children }) {
  return <h2 className="font-display lowercase text-lg text-ink mb-3">{children}</h2>;
}

function Warn({ children }) {
  return (
    <p className="mt-2 text-xs text-warn flex items-start gap-1.5">
      <AlertTriangle size={12} className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

function ModelField({ id, label, value, onChange, models, hint }) {
  // Free text with the provider's list as suggestions: any id the provider serves works.
  return (
    <div className="mt-4">
      <label htmlFor={id} className="eyebrow block mb-1">{label}</label>
      <input
        id={id}
        list={`${id}-options`}
        value={value}
        onChange={(e) => onChange(e.target.value.trim())}
        placeholder="Pick from the list or type a model id"
        className="input-field font-mono"
      />
      <datalist id={`${id}-options`}>
        {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </datalist>
      {hint && <p className="mt-1 text-xs text-ink2">{hint}</p>}
    </div>
  );
}

function OpenRouterModels({ settings, keyOnServer, hasKey }) {
  const [lists, setLists] = useState(null);
  useEffect(() => {
    let alive = true;
    apiJson('/api/ai/openrouter/models')
      .then((d) => { if (alive) setLists(d.available ? d : { error: d.reason || 'The model list is unavailable.' }); })
      .catch((e) => { if (alive) setLists({ error: String(e.message || e) }); });
    return () => { alive = false; };
  }, []);

  const text = lists?.text || [];
  const chosen = text.find((m) => m.id === settings.openrouterModel);
  const listLoaded = text.length > 0;

  return (
    <Panel>
      <Heading>OpenRouter models</Heading>
      {keyOnServer && !hasKey && (
        <p className="text-xs text-ink2 mb-2">The server has its own OpenRouter key, so typing one above is optional.</p>
      )}
      {lists?.error && <Warn>{lists.error} You can still type a model id.</Warn>}

      <ModelField
        id="openrouter-model"
        label="Model that finds and titles clips"
        value={settings.openrouterModel}
        onChange={(v) => setAiSettings({ openrouterModel: v })}
        models={text}
        hint="Any text model works; a model that reads pictures sees the frames, and one that handles JSON keeps the pipeline reliable."
      />
      {settings.openrouterModel && listLoaded && !chosen && (
        <Warn>This id is not in OpenRouter's list. The request will fail if the id is wrong.</Warn>
      )}
      {chosen && !chosen.json && (
        <Warn>This model does not list JSON output, so the clip pipeline may fail on it.</Warn>
      )}
      {chosen && !chosen.vision && (
        <p className="mt-2 text-xs text-ink2">Text only: the steps that read frames (layout, hook, silent videos) fall back.</p>
      )}

    </Panel>
  );
}

function ollamaLabel(m) {
  const size = m.size ? `${(m.size / 1e9).toFixed(1)} GB` : null;
  const vision = m.vision === true ? 'reads pictures' : m.vision === false ? 'text only' : 'picture support unknown';
  return [m.name, m.parameterSize, m.quantization, size, vision].filter(Boolean).join(' · ');
}

function OllamaModels({ settings }) {
  const [state, setState] = useState({ loading: true, data: null });
  const load = useCallback(() => {
    setState((s) => ({ ...s, loading: true }));
    apiJson('/api/ai/ollama/models')
      .then((data) => setState({ loading: false, data }))
      .catch((e) => setState({ loading: false, data: { available: false, reason: String(e.message || e), models: [] } }));
  }, []);
  useEffect(() => { load(); }, [load]);

  const models = state.data?.models || [];
  const chosen = models.find((m) => m.name === settings.ollamaModel);

  return (
    <Panel>
      <div className="flex items-center justify-between gap-3 mb-3">
        <Heading>Ollama models on this machine</Heading>
        <button type="button" onClick={load} disabled={state.loading} className="btn-quiet px-3 py-1.5 text-xs shrink-0">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>
      {state.loading && <p className="text-xs text-ink2">Looking for Ollama…</p>}
      {!state.loading && state.data && !state.data.available && <Warn>{state.data.reason}</Warn>}
      {!state.loading && state.data?.available && models.length === 0 && (
        <Warn>Ollama is running, but no model is downloaded yet. Download one with `ollama pull`, then refresh.</Warn>
      )}
      {!state.loading && models.length > 0 && (
        <div className="mt-2">
          <label htmlFor="ollama-model" className="eyebrow block mb-1">Model</label>
          <select
            id="ollama-model"
            value={settings.ollamaModel}
            onChange={(e) => setAiSettings({ ollamaModel: e.target.value })}
            className="input-field font-mono"
          >
            <option value="">Choose a downloaded model</option>
            {models.map((m) => <option key={m.name} value={m.name}>{ollamaLabel(m)}</option>)}
          </select>
          {chosen && chosen.vision === false && (
            <Warn>This model cannot read pictures, so the frame-based steps fall back to plain cropping.</Warn>
          )}
          {chosen && chosen.vision == null && (
            <p className="mt-2 text-xs text-ink2">Ollama did not say whether this model reads pictures.</p>
          )}
        </div>
      )}
    </Panel>
  );
}

export default function AiSettings({ apiKey, onKeySet, keyOnServer = false }) {
  const settings = useAiSettings();
  return (
    <>
      <Panel>
        <Heading>AI provider</Heading>
        <p className="text-xs text-ink2 mb-4">
          Choose the provider that runs your clips. OpenShorts never picks one for you.
        </p>
        <div role="radiogroup" aria-label="AI provider" className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {PROVIDERS.map(({ id, label, icon, note }) => {
            const active = settings.provider === id;
            return (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => setAiSettings({ provider: id })}
                className={`text-left p-4 rounded-input border transition-colors ${
                  active ? 'border-brass bg-paper3' : 'border-rule hover:bg-paper2'}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={active ? 'text-brass' : 'text-ink2'}>{icon}</span>
                  <span className="font-display lowercase text-ink">{label}</span>
                </div>
                <p className="text-xs text-ink2">{note}</p>
              </button>
            );
          })}
        </div>
        {!settings.provider && (
          <Warn>Choose a provider before you generate clips.</Warn>
        )}
      </Panel>

      {settings.provider === 'openrouter' && (
        <>
          <KeyInput onKeySet={onKeySet} savedKey={apiKey} />
          <OpenRouterModels settings={settings} keyOnServer={keyOnServer} hasKey={!!apiKey} />
        </>
      )}
      {settings.provider === 'ollama' && <OllamaModels settings={settings} />}
    </>
  );
}
