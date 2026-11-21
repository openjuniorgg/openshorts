import React, { useMemo } from 'react';
import { Activity, AlertCircle, Loader2, Sparkles, Square, Terminal } from 'lucide-react';

// Shown in place of the form while the live monitor runs. Left: every
// monitored live plus System Logs. Right: the clips the monitor makes, rendered
// by the same ResultCard the Clip Generator uses (preview, viral score, actions),
// ordered by predicted_score the same way App orders the Clip Generator results.
// renderResultCard(clip, index, jobId) is App's own ResultCard wiring.

const PENDING_LABEL = {
  queued: 'queued',
  processing: 'processing',
  failed: 'failed',
  completed: 'no clips rendered',
};

function timeOf(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString() : '';
}

export default function LiveWorkspace({ streams, clips, logs, onStop, renderResultCard, transcribing = false }) {
  const { ready, pending } = useMemo(() => {
    const readyItems = [];
    const pendingItems = [];
    for (const c of clips) {
      const rendered = c.status === 'completed' && Array.isArray(c.result?.clips) ? c.result.clips : [];
      if (rendered.length > 0) {
        rendered.forEach((clip, index) => readyItems.push({
          key: `${c.job_id}-${index}`,
          jobId: c.job_id,
          index,
          clip,
        }));
      } else {
        pendingItems.push({
          key: c.job_id || `${c.channel}-${c.timestamp}`,
          channel: c.channel,
          record: c,
          timestamp: c.timestamp,
        });
      }
    }
    readyItems.sort((a, b) => (b.clip.predicted_score ?? -1) - (a.clip.predicted_score ?? -1));
    pendingItems.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
    return { ready: readyItems, pending: pendingItems };
  }, [clips]);

  return (
    <div className="h-full flex flex-col md:flex-row gap-3 md:gap-4 p-3 md:p-4 overflow-y-auto md:overflow-y-hidden custom-scrollbar animate-fade">

      {/* Left: monitored lives and System Logs */}
      <div className="w-full md:w-[40%] lg:w-[35%] md:h-full flex flex-col shrink-0 md:shrink card p-3.5 sm:p-6 md:overflow-y-auto custom-scrollbar">
        <div className="mb-4 flex items-center justify-between gap-2">
          <h2 className="text-sm font-medium text-ink lowercase flex items-center gap-2">
            <Activity className="text-brass animate-pulse" size={18} />
            Live Analysis
          </h2>
          <span className="badge-ok">LIVE</span>
        </div>

        <p className="text-xs text-muted mb-3">
          {streams.length} {streams.length === 1 ? 'live' : 'lives'} monitored
          {transcribing ? ' · live transcript on' : ''}
        </p>

        <div className="space-y-2.5 mb-4">
          {streams.map((s) => (
            <div key={s.id || s.url} className="rounded-card border border-rule2 bg-paper2 overflow-hidden">
              {/* The live itself: the platform's own muted player */}
              {s.embed_url ? (
                <iframe
                  src={s.embed_url}
                  title={`${s.name} live`}
                  className="w-full aspect-video bg-black border-0"
                  allow="autoplay; fullscreen; encrypted-media"
                  allowFullScreen
                />
              ) : (
                <div className="w-full aspect-video bg-black flex items-center justify-center text-xs text-muted">
                  no player for this live
                </div>
              )}
              <div className="p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-sm text-ink truncate">{s.name}</span>
                <span className="text-[10px] uppercase tracking-wider text-muted shrink-0">{s.platform}</span>
              </div>
              <div className="flex items-center justify-between gap-2 text-[11px] text-muted">
                <span className="truncate" title={s.game_category || ''}>{s.game_category || 'category unknown'}</span>
                <span>{s.stream_audio_status || s.status || 'monitoring'}</span>
                <span>{Number.isFinite(s.audio_peak_db) ? `peak ${s.audio_peak_db} dB` : ''}</span>
              </div>
              <div className="h-1.5 rounded-full bg-paper3 overflow-hidden">
                <div
                  className="h-full bg-brass transition-all duration-500"
                  style={{ width: `${Math.round(Math.min(1, Math.max(0, s.audio_level || 0)) * 100)}%` }}
                />
              </div>
              {s.last_error && (
                <p className="text-[11px] text-muted break-words">{s.last_error}</p>
              )}
              <p className="text-[11px] text-muted">
                clips {s.clips_captured ?? 0}{s.clip_failures ? ` · failed ${s.clip_failures}` : ''}
              </p>
              </div>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={onStop}
          className="btn-ghost mb-4 w-full text-xs py-2 flex items-center justify-center gap-2"
        >
          <Square size={12} /> stop monitoring
        </button>

        <div className="bg-paper rounded-card border border-rule overflow-hidden flex flex-col min-h-[180px]">
          <div className="px-3.5 py-2.5 border-b border-rule bg-paper2 readout flex items-center gap-2">
            <Terminal size={12} /> System Logs
          </div>
          <div className="p-3.5 max-h-72 overflow-y-auto font-mono text-[11px] space-y-1.5 custom-scrollbar text-ink2 break-words">
            {logs.length === 0 && <div className="text-muted">waiting for activity…</div>}
            {logs.map((line, i) => (
              <div key={i} className={/fail|error/i.test(line) ? 'text-danger' : ''}>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: clips made by the monitor, rendered by the Clip Generator's ResultCard */}
      <div className="w-full md:w-[60%] lg:w-[65%] md:h-full flex flex-col shrink-0 md:shrink card p-3.5 sm:p-6 md:overflow-y-auto custom-scrollbar">
        <div className="mb-4 sm:mb-6 flex items-center justify-between gap-2">
          <h2 className="font-display lowercase text-lg sm:text-xl text-ink">Generated Shorts</h2>
          {ready.length > 0 && (
            <span className="readout bg-paper3 px-2.5 py-1 rounded-full">{ready.length} clips</span>
          )}
        </div>

        {ready.length === 0 && pending.length === 0 && (
          <p className="text-sm text-muted">
            Waiting for the first viral moment. Clips appear here as soon as they are made, best viral score first.
          </p>
        )}

        <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4 pb-6">
          {pending.map((item) => {
            const st = item.record.status || 'queued';
            const failed = st === 'failed';
            return (
              <div key={item.key} className="rounded-card border border-rule2 bg-paper2 p-4 flex flex-col gap-2 min-h-[160px]">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs text-ink truncate">{item.channel}</span>
                  <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded border border-rule2 text-muted shrink-0 flex items-center gap-1">
                    {failed ? <AlertCircle size={10} className="text-danger" /> : (st === 'processing' ? <Loader2 size={10} className="animate-spin" /> : <Sparkles size={10} />)}
                    {PENDING_LABEL[st] || st}
                  </span>
                </div>
                <span className="text-[11px] text-muted">{timeOf(item.timestamp)}</span>
                {failed && <p className="text-xs text-danger break-words">{item.record.error || 'Failed'}</p>}
                {!failed && item.record.last_log && (
                  <p className="text-[11px] text-muted font-mono truncate" title={item.record.last_log}>{item.record.last_log}</p>
                )}
              </div>
            );
          })}

          {ready.map((item) => (
            <div key={item.key}>
              {renderResultCard(item.clip, item.index, item.jobId)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
