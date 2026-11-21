// System Logs lines for the live monitor, made by comparing two snapshots of
// /api/live-monitor/status. Pure: no React, no network.

const clipKey = (c) => c.job_id || `${c.channel}-${c.timestamp}`;

export function liveLogLines(prev, next) {
  const lines = [];
  if (!next) return lines;

  if (!prev) {
    if (next.running) lines.push(`monitor running: ${next.streams.length} live(s)`);
    return lines;
  }

  if (prev.running !== next.running) {
    lines.push(next.running ? 'monitor started' : 'monitor stopped');
  }

  const before = new Map(prev.streams.map((s) => [s.id, s]));
  for (const s of next.streams) {
    const p = before.get(s.id);
    if (!p) {
      lines.push(`watching ${s.name} (${s.platform})`);
      continue;
    }
    if (p.stream_audio_status !== s.stream_audio_status) {
      lines.push(`${s.name}: ${s.stream_audio_status}`);
    }
    if (s.last_error && s.last_error !== p.last_error) {
      lines.push(`${s.name}: ${s.last_error}`);
    }
    if (s.last_spike_reason && s.last_spike_reason !== p.last_spike_reason) {
      lines.push(`${s.name}: spike, ${s.last_spike_reason}`);
    }
  }

  const seen = new Map(prev.clips.map((c) => [clipKey(c), c.status]));
  for (const c of next.clips) {
    const key = clipKey(c);
    if (!seen.has(key)) {
      lines.push(`${c.channel}: clip made (${c.reason || 'spike'})`);
    } else if (seen.get(key) !== c.status) {
      const err = c.error ? ` - ${c.error}` : '';
      lines.push(`${c.channel}: clip ${c.status}${err}`);
    }
  }
  return lines;
}
