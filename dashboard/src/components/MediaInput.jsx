import React, { useState, useEffect, useRef } from 'react';
import {
    Link2,
    Upload,
    FileVideo,
    X,
    Info,
    Loader2,
    ChevronDown,
    Radio,
    Plus,
    Sparkles,
    Square,
    AlertCircle,
    Volume2,
    Film,
    Scissors,
    Zap,
    Download,
} from 'lucide-react';
import { track } from '../lib/analytics';
import { getApiUrl } from '../config';
import { startLiveMonitor, stopLiveMonitor, triggerLiveClip } from '../lib/api';
import SourceCost from './SourceCost';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch', 'Kick',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

// Mirrors the server's MIN_SOURCE_SECONDS: shorter sources are rejected with a
// 400 after the whole file was uploaded. Checking the duration in the browser
// says so the moment the file is picked (a 27 s upload used to end in a bare
// "That run failed").
const MIN_SOURCE_SECONDS = 45;

// Duration of a local video file in seconds, or null when the browser cannot
// read it (unsupported codec): then the server stays the judge.
const readVideoDuration = (file) => new Promise((resolve) => {
    try {
        const url = URL.createObjectURL(file);
        const v = document.createElement('video');
        v.preload = 'metadata';
        const done = (d) => { URL.revokeObjectURL(url); resolve(d); };
        v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
        v.onerror = () => done(null);
        setTimeout(() => done(null), 8000);
        v.src = url;
    } catch { resolve(null); }
});

// Sanitizes and strictly deduplicates channel lists. Strips objects to strings,
// removes duplicate URLs/handles, and ensures each channel exists at most once.
function sanitizeChannelList(rawList) {
    if (!Array.isArray(rawList)) return ['kick.com/bistecone', 'twitch.tv/alanzoka'];
    const seen = new Set();
    const result = [];
    for (const item of rawList) {
        let str = '';
        if (typeof item === 'string') {
            str = item.trim();
        } else if (item && typeof item === 'object') {
            str = (item.url || item.channel || item.name || '').trim();
        }
        if (!str) continue;
        const normKey = str.toLowerCase()
            .replace(/^(https?:\/\/)?(www\.)?/, '')
            .replace(/\/+$/, '')
            .replace(/\?.*$/, '');
        if (normKey && !seen.has(normKey)) {
            seen.add(normKey);
            result.push(str);
        }
    }
    return result;
}

export default function MediaInput({ onProcess, isProcessing, liveState, onLiveChange }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // 3 modes: 'file' (Upload File), 'url' (Paste Link), 'live' (Live Monitor)
    const [mode, setMode] = useState('file');
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [fileSeconds, setFileSeconds] = useState(null);
    const fileTooShort = fileSeconds != null && fileSeconds < MIN_SOURCE_SECONDS;
    const [acknowledged, setAcknowledged] = useState(false);
    // {start, end} in seconds when the user picked a stretch under the cost line.
    const [range, setRange] = useState(null);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [showInfo, setShowInfo] = useState(false);
    // Advanced generation controls — empty string means "let the AI decide",
    // which keeps the default pipeline behavior untouched.
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [targetClips, setTargetClips] = useState('');
    const [clipMinSeconds, setClipMinSeconds] = useState('');
    const [clipMaxSeconds, setClipMaxSeconds] = useState('');
    // Auto-hook: burn the AI hook text into every clip. Off by default; the
    // choice persists so user preference sticks across sessions.
    const [autoHook, setAutoHook] = useState(() => {
        try { return localStorage.getItem('os_auto_hook') === '1'; } catch { return false; }
    });
    const [autoHookStyle, setAutoHookStyle] = useState(() => {
        // v2 key: the default became 'pill'; an old saved 'classic' was just the old default.
        try { return localStorage.getItem('os_auto_hook_style_v2') || 'pill'; } catch { return 'pill'; }
    });
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    const infoRef = useRef(null);

    // Live Monitoring state
    const [liveChannels, setLiveChannels] = useState(() => {
        try {
            const saved = localStorage.getItem('os_live_channels');
            if (saved) {
                const parsed = JSON.parse(saved);
                const clean = sanitizeChannelList(parsed);
                if (clean.length > 0) {
                    try { localStorage.setItem('os_live_channels', JSON.stringify(clean)); } catch (_) {}
                    return clean;
                }
            }
        } catch (_) { /* ignore */ }
        return ['kick.com/bistecone', 'twitch.tv/alanzoka'];
    });
    const [channelInput, setChannelInput] = useState('');
    const [liveStarting, setLiveStarting] = useState(false);
    const [liveError, setLiveError] = useState(null);
    // The live monitor is polled by App, which keeps polling while this form is hidden.
    const isLiveRunning = !!liveState?.running;
    const capturedClips = liveState?.clips || [];
    const liveStreamsData = liveState?.streams || [];
    const [clippingStreams, setClippingStreams] = useState({});
    const [clippingFeedback, setClippingFeedback] = useState({});

    // Strict deduplication of channel list for display and grid rendering
    const distinctChannels = React.useMemo(() => {
        return sanitizeChannelList(liveChannels);
    }, [liveChannels]);

    // Save live channels to local storage (strictly sanitized and deduplicated)
    useEffect(() => {
        try {
            const clean = sanitizeChannelList(liveChannels);
            localStorage.setItem('os_live_channels', JSON.stringify(clean));
        } catch (_) { /* ignore */ }
    }, [liveChannels]);

    // Close the compatibility popover on any outside click.
    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        let cancelled = false;
        setFileSeconds(null);
        if (!file) return undefined;
        readVideoDuration(file).then((d) => {
            if (cancelled) return;
            setFileSeconds(d);
            if (d != null && d < MIN_SOURCE_SECONDS) {
                track('SourceTooShort', { props: { seconds: String(Math.round(d)) } });
            }
        });
        return () => { cancelled = true; };
    }, [file]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode((m) => (m === 'url' ? 'file' : m));
                }
            })
            .catch(() => {});
    }, []);

    // A link pasted in the landing hero: preload it here so the user picks up
    // where they left off. Not auto-submitted — the rights attestation below
    // has to be ticked by the user.
    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('url');
            setUrl(pending);
        }
    }, []);

    const handleAddChannel = (channelToAdd) => {
        const raw = (channelToAdd || channelInput).trim();
        if (!raw) return;
        const items = raw.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
        const combined = [...liveChannels, ...items];
        const clean = sanitizeChannelList(combined);
        setLiveChannels(clean);
        setChannelInput('');
        setLiveError(null);
    };

    const handleRemoveChannel = (chToRemove) => {
        const removeNorm = (typeof chToRemove === 'string' ? chToRemove : (chToRemove?.url || chToRemove?.channel || chToRemove?.name || ''))
            .toLowerCase().replace(/^https?:\/\/(www\.)?/, '').replace(/\/+$/, '').replace(/\?.*$/, '');
        setLiveChannels((prev) => {
            const remaining = prev.filter((ch) => {
                const itemNorm = (typeof ch === 'string' ? ch : (ch?.url || ch?.channel || ch?.name || ''))
                    .toLowerCase().replace(/^https?:\/\/(www\.)?/, '').replace(/\/+$/, '').replace(/\?.*$/, '');
                return itemNorm !== removeNorm;
            });
            return sanitizeChannelList(remaining);
        });
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            handleAddChannel();
        }
    };

    const getFriendlyErrorMessage = (err) => {
        if (!err) return null;
        if (err.status === 404) {
            return 'Backend API endpoint not found (404). Please ensure the backend server has been restarted with the latest update.';
        }
        if (err.status === 500) {
            return `Backend server error (500): ${err.detail || 'Internal error starting stream monitoring. Please check server logs.'}`;
        }
        if (err.status === 502 || err.status === 503 || err.status === 504) {
            return 'Backend server is currently unreachable. Please try again shortly.';
        }
        if (err.detail) {
            return typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
        }
        if (err.message && err.message.includes('Failed to fetch')) {
            return 'Cannot connect to server. Please check that the backend is running on port 8000.';
        }
        return err.message || 'Failed to start live monitoring. Please try again.';
    };

    const getPlatformTag = (ch) => {
        if (!ch) return { label: 'LIVE', color: 'bg-amber-500/15 text-amber-600 border-amber-500/30' };
        if (typeof ch === 'object') {
            const p = (ch.platform || '').toLowerCase();
            if (p === 'kick') return { label: 'KICK', color: 'bg-emerald-500/15 text-emerald-600 border-emerald-500/30' };
            if (p === 'twitch') return { label: 'TWITCH', color: 'bg-purple-500/15 text-purple-600 border-purple-500/30' };
            if (p === 'youtube') return { label: 'YOUTUBE', color: 'bg-red-500/15 text-red-600 border-red-500/30' };
            ch = ch.url || ch.channel || ch.name || '';
        }
        const lower = String(ch || '').toLowerCase();
        if (lower.includes('kick.com')) return { label: 'KICK', color: 'bg-emerald-500/15 text-emerald-600 border-emerald-500/30' };
        if (lower.includes('twitch.tv') || (!lower.includes('.') && !lower.includes('/'))) return { label: 'TWITCH', color: 'bg-purple-500/15 text-purple-600 border-purple-500/30' };
        if (lower.includes('youtube.com') || lower.includes('youtu.be')) return { label: 'YOUTUBE', color: 'bg-red-500/15 text-red-600 border-red-500/30' };
        return { label: 'LIVE', color: 'bg-amber-500/15 text-amber-600 border-amber-500/30' };
    };

    const getStreamEmbedInfo = (channel) => {
        if (!channel) return null;
        const hostname = typeof window !== 'undefined' && window.location.hostname ? window.location.hostname : 'localhost';

        if (typeof channel === 'object' && channel.embed_url) {
            const platform = (channel.platform || 'live').toLowerCase();
            const badge = platform === 'kick'
                ? { label: 'KICK', color: 'bg-emerald-500/15 text-emerald-600 border-emerald-500/30' }
                : platform === 'twitch'
                ? { label: 'TWITCH', color: 'bg-purple-500/15 text-purple-600 border-purple-500/30' }
                : platform === 'youtube'
                ? { label: 'YOUTUBE', color: 'bg-red-500/15 text-red-600 border-red-500/30' }
                : { label: 'LIVE', color: 'bg-amber-500/15 text-amber-600 border-amber-500/30' };
            return {
                platform,
                name: channel.name || channel.channel || channel.id || 'Live Stream',
                url: channel.embed_url,
                badge,
                id: channel.id || channel.name,
            };
        }

        const raw = typeof channel === 'string' ? channel : (channel.url || channel.channel || channel.name || '');
        const clean = String(raw).trim();

        // Kick: https://player.kick.com/<channel>?muted=true&autoplay=true
        if (clean.toLowerCase().includes('kick.com') || (typeof channel === 'object' && channel.platform === 'kick')) {
            let name = clean.replace(/^(https?:\/\/)?(www\.)?kick\.com\//i, '').replace(/^kick:/i, '').split('/')[0].split('?')[0];
            if (typeof channel === 'object' && (channel.channel || channel.name)) name = channel.channel || channel.name;
            name = (name || '').replace(/^@/, '');
            return {
                platform: 'kick',
                name: name || clean,
                url: `https://player.kick.com/${name}?muted=true&autoplay=true`,
                badge: { label: 'KICK', color: 'bg-emerald-500/15 text-emerald-600 border-emerald-500/30' },
                id: `kick_${name}`,
            };
        }

        // YouTube: https://www.youtube.com/embed/<videoId>?autoplay=1&mute=1&playsinline=1
        if (clean.toLowerCase().includes('youtube.com') || clean.toLowerCase().includes('youtu.be') || (typeof channel === 'object' && channel.platform === 'youtube')) {
            let videoId = '';
            if (clean.includes('watch?v=')) {
                videoId = clean.split('watch?v=')[1]?.split('&')[0];
            } else if (clean.includes('/live/')) {
                videoId = clean.split('/live/')[1]?.split('?')[0];
            } else if (clean.includes('youtu.be/')) {
                videoId = clean.split('youtu.be/')[1]?.split('?')[0];
            }
            if (!videoId && typeof channel === 'object' && (channel.channel || channel.name)) {
                videoId = channel.channel || channel.name;
            }
            const embedUrl = videoId
                ? `https://www.youtube.com/embed/${videoId}?autoplay=1&mute=1&playsinline=1`
                : `https://www.youtube.com/embed/live_stream?channel=${encodeURIComponent(clean)}&autoplay=1&mute=1`;

            return {
                platform: 'youtube',
                name: videoId || clean,
                url: embedUrl,
                badge: { label: 'YOUTUBE', color: 'bg-red-500/15 text-red-600 border-red-500/30' },
                id: `youtube_${videoId || clean}`,
            };
        }

        // Twitch: https://player.twitch.tv/?channel=<channel>&parent=<hostname>&muted=true&autoplay=true
        let name = clean.replace(/^(https?:\/\/)?(www\.)?twitch\.tv\//i, '').replace(/^twitch:/i, '').split('/')[0].split('?')[0];
        if (typeof channel === 'object' && (channel.channel || channel.name)) name = channel.channel || channel.name;
        name = (name || '').replace(/^@/, '');
        return {
            platform: 'twitch',
            name: name || clean,
            url: `https://player.twitch.tv/?channel=${encodeURIComponent(name)}&parent=${hostname}&muted=true&autoplay=true`,
            badge: { label: 'TWITCH', color: 'bg-purple-500/15 text-purple-600 border-purple-500/30' },
            id: `twitch_${name}`,
        };
    };

    const getRecentClip = (chName) => {
        if (!Array.isArray(capturedClips) || capturedClips.length === 0) return null;
        const lower = (chName || '').toLowerCase();
        return [...capturedClips].reverse().find((c) => {
            const cChan = (c.channel || c.channel_name || '').toLowerCase();
            return cChan.includes(lower) || lower.includes(cChan);
        });
    };

    // The render's clips live in result.clips[] (each with its own download link).
    const firstClipHref = (c) => {
        const k = c?.result?.clips?.[0];
        if (!k) return null;
        return k.download_url || (k.video_url ? getApiUrl(k.video_url) : null);
    };

    const handleStartLive = async () => {
        if (distinctChannels.length === 0) return;
        setLiveStarting(true);
        setLiveError(null);
        try {
            await startLiveMonitor(distinctChannels);
            onLiveChange?.(true);
        } catch (err) {
            console.error('Failed to start live monitor:', err);
            setLiveError(getFriendlyErrorMessage(err));
        } finally {
            setLiveStarting(false);
        }
    };

    const handleStopLive = async () => {
        setLiveStarting(true);
        setLiveError(null);
        try {
            await stopLiveMonitor();
            onLiveChange?.(false);
        } catch (err) {
            console.error('Failed to stop live monitor:', err);
            setLiveError(getFriendlyErrorMessage(err));
        } finally {
            setLiveStarting(false);
        }
    };

    const handleTriggerClip = async (channelRaw, streamerName) => {
        const key = streamerName || channelRaw;
        setClippingStreams((prev) => ({ ...prev, [key]: true }));
        setClippingFeedback((prev) => ({
            ...prev,
            [key]: '⚡ Recording highlight... Sending it to the Shorts AI',
        }));
        try {
            await triggerLiveClip(channelRaw);
            setClippingFeedback((prev) => ({
                ...prev,
                [key]: '✅ Highlight sent to the AI. The clip will appear on the right.',
            }));
            setTimeout(() => {
                setClippingFeedback((prev) => ({ ...prev, [key]: null }));
            }, 6000);
        } catch (err) {
            console.error('Failed to trigger live clip:', err);
            setClippingFeedback((prev) => ({
                ...prev,
                [key]: `❌ Error: ${err.detail || err.message || 'Failed to extract clip'}`,
            }));
            setTimeout(() => {
                setClippingFeedback((prev) => ({ ...prev, [key]: null }));
            }, 6000);
        } finally {
            setClippingStreams((prev) => ({ ...prev, [key]: false }));
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (mode === 'live') {
            if (!isLiveRunning && liveChannels.length > 0) {
                handleStartLive();
            }
            return;
        }
        if (!acknowledged) return;
        const advanced = {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            autoHook,
            autoHookStyle,
            layout,
            sourceStart: range ? Math.round(range.start) : null,
            sourceEnd: range ? Math.round(range.end) : null,
            captions: '0',
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style_v2', autoHookStyle);
            localStorage.setItem('os_layout', layout);
        } catch { /* ignore */ }
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, acknowledged: true, outputFormat, captions: '0', ...advanced });
        } else if (mode === 'file' && file && !fileTooShort) {
            onProcess({ type: 'file', payload: file, acknowledged: true, outputFormat, captions: '0', ...advanced });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="card p-4 sm:p-6 animate-fade bg-paper">
            {/* 3 Source Tabs: Upload File, Paste Link, Live Monitor */}
            <div className="flex gap-4 sm:gap-6 mb-6 border-b border-rule" data-tutorial="source-tabs">
                <button
                    type="button"
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'file'
                        ? 'text-ink border-brass font-medium'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Upload size={16} className={`hidden sm:block ${mode === 'file' ? 'text-brass' : ''}`} />
                    Upload File
                </button>
                {youtubeUrlEnabled && (
                    <button
                        type="button"
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'url'
                            ? 'text-ink border-brass font-medium'
                            : 'text-muted border-transparent hover:text-ink2'
                            }`}
                    >
                        <Link2 size={16} className={`hidden sm:block ${mode === 'url' ? 'text-brass' : ''}`} />
                        Paste Link
                    </button>
                )}
                <button
                    type="button"
                    onClick={() => setMode('live')}
                    className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'live'
                        ? 'text-ink border-brass font-medium'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Radio size={16} className={`hidden sm:block ${mode === 'live' ? 'text-brass' : ''}`} />
                    Live Monitor
                    {isLiveRunning && (
                        <span className="relative flex h-2 w-2 ml-0.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-ok opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-ok"></span>
                        </span>
                    )}
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                {/* 1. PASTE LINK MODE */}
                {mode === 'url' && (
                    <div className="space-y-4" data-tutorial="drop-zone">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://... paste a video link (YouTube, Twitch, Kick, etc.)"
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade bg-paper2">
                                        <p className="eyebrow mb-2">Paste a link from</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 border border-rule text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                            …and 1,000+ more sites. If a link has a public video, we can usually fetch it.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* 2. UPLOAD FILE MODE */}
                {mode === 'file' && (
                    <div
                        data-tutorial="drop-zone"
                        className={`border-2 border-dashed rounded-card p-6 sm:p-8 text-center transition-colors ${file ? 'border-brass bg-paper2' : 'border-rule2 hover:border-brass bg-paper'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : null}
                        {file && fileTooShort ? (
                            <p className="text-danger text-sm mt-3" role="alert">
                                This video is {Math.round(fileSeconds)}s long. Clip generation needs at least {MIN_SOURCE_SECONDS}s
                                of footage to cut from: it already is a short. Pick a longer video.
                            </p>
                        ) : null}
                        {!file && (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-ink2" size={18} />
                                <p className="text-ink font-medium lowercase">Click to upload or drag and drop</p>
                                <p className="readout text-ink2 font-medium mt-2">MP4, MOV up to 500MB · at least {MIN_SOURCE_SECONDS}s long</p>
                            </label>
                        )}
                    </div>
                )}

                {/* 3. LIVE MONITOR MODE */}
                {mode === 'live' && (
                    <div className="space-y-4 animate-fade">
                        {/* Active animated banner when monitoring */}
                        {isLiveRunning && (
                            <div className="flex items-center justify-between gap-3 p-3.5 rounded-card bg-paper2 border border-rule2 animate-fade">
                                <div className="flex items-center gap-2.5 min-w-0">
                                    <span className="relative flex h-2.5 w-2.5 shrink-0">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-ok opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-ok"></span>
                                    </span>
                                    <div className="truncate text-xs">
                                        <span className="font-semibold text-ink">
                                            Monitoring {distinctChannels.length} stream{distinctChannels.length === 1 ? '' : 's'} in real-time
                                        </span>
                                        <span className="text-ink2 ml-1 hidden sm:inline">
                                            · Analyzing in-stream audio peaks, streamer voice energy & highlight moments
                                        </span>
                                    </div>
                                </div>
                                <span className="badge-ok shrink-0">
                                    <Radio size={10} className="animate-pulse" /> LIVE ACTIVE
                                </span>
                            </div>
                        )}

                        {/* LIVE MULTI-PLAYER STREAM GRID (Active Mode) */}
                        {isLiveRunning && distinctChannels.length > 0 && (
                            <div className="space-y-3 pt-1 animate-fade">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Film size={15} className="text-brass" />
                                        <p className="eyebrow text-ink">
                                            Live Stream Monitor ({distinctChannels.length})
                                        </p>
                                    </div>
                                    <span className="text-[11px] text-muted font-mono flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-ok animate-pulse" />
                                        Real-time stream feeds
                                    </span>
                                </div>

                                <div className={`grid gap-4 ${
                                    distinctChannels.length === 1
                                        ? 'grid-cols-1 max-w-2xl mx-auto'
                                        : 'grid-cols-1 md:grid-cols-2'
                                }`}>
                                    {distinctChannels.map((ch) => {
                                        const info = getStreamEmbedInfo(ch);
                                        if (!info) return null;
                                        const recentClip = getRecentClip(info.name);
                                        const tileKey = `${info.platform}_${info.name}`;
                                        const streamObj = liveStreamsData.find((s) => s.name === info.name || s.channel === info.name || s.url === ch);
                                        const audioLevel = streamObj?.audio_level != null ? streamObj.audio_level : null;
                                        const audioPeakDb = streamObj?.audio_peak_db != null ? streamObj.audio_peak_db : null;

                                        return (
                                            <div
                                                key={tileKey}
                                                className="rounded-card border border-rule2 bg-paper2 overflow-hidden shadow-sm flex flex-col transition-all"
                                            >
                                                {/* Stream Header */}
                                                <div className="flex items-center justify-between px-3 py-2 bg-paper border-b border-rule2 text-xs">
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded border uppercase tracking-wider ${info.badge.color}`}>
                                                            {info.badge.label}
                                                        </span>
                                                        <span className="font-semibold text-ink truncate font-mono">
                                                            {info.name}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-2 shrink-0">
                                                        <div className="flex items-center gap-1 text-[10px] text-muted font-mono" title="In-Stream Audio Spike & Hype Energy AI Active">
                                                            <span className="flex items-end gap-[1.5px] h-3">
                                                                <span className="w-0.5 h-2 bg-brass animate-pulse rounded-full" />
                                                                <span className="w-0.5 h-3 bg-brass animate-pulse rounded-full delay-75" />
                                                                <span className="w-0.5 h-1.5 bg-brass animate-pulse rounded-full delay-150" />
                                                            </span>
                                                            <span className="hidden sm:inline">AUDIO AI</span>
                                                        </div>
                                                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-danger px-1.5 py-0.5 rounded-full bg-danger/10 border border-danger/25">
                                                            <span className="w-1.5 h-1.5 rounded-full bg-danger animate-ping" />
                                                            LIVE
                                                        </span>
                                                    </div>
                                                </div>

                                                {/* Embedded Stream Video Player */}
                                                <div className="relative aspect-video w-full bg-black/90">
                                                    <iframe
                                                        src={info.url}
                                                        title={`${info.name} Live Stream`}
                                                        className="w-full h-full border-0"
                                                        allow="autoplay; fullscreen; encrypted-media"
                                                        allowFullScreen
                                                    />
                                                    {recentClip && (
                                                        <div className={`absolute top-2 left-2 right-2 backdrop-blur text-white text-xs px-3 py-1.5 rounded-card shadow-lg flex items-center justify-between animate-fade z-10 border ${
                                                            recentClip.status === 'failed'
                                                                ? 'bg-danger/90 border-danger/30'
                                                                : recentClip.status === 'completed'
                                                                ? 'bg-ok/95 border-emerald-400/30'
                                                                : 'bg-amber-600/90 border-amber-400/30'
                                                        }`}>
                                                            <div className="flex items-center gap-1.5 font-semibold">
                                                                {recentClip.status === 'failed' ? (
                                                                    <AlertCircle size={14} className="text-white fill-danger" />
                                                                ) : recentClip.status === 'completed' ? (
                                                                    <Sparkles size={14} className="text-yellow-300 fill-current animate-pulse" />
                                                                ) : (
                                                                    <Sparkles size={14} className="text-yellow-300 fill-current animate-spin" />
                                                                )}
                                                                <span>
                                                                    {recentClip.status === 'completed'
                                                                        ? '✨ Highlight ready!'
                                                                        : recentClip.status === 'failed'
                                                                        ? 'Processing failed'
                                                                        : recentClip.status === 'queued'
                                                                        ? 'Queued...'
                                                                        : 'Processing Short...'}
                                                                </span>
                                                            </div>
                                                            <div className="flex items-center gap-1.5">
                                                                <span className="text-[10px] bg-white/20 px-1.5 py-0.5 rounded font-mono">
                                                                    {recentClip.status || 'processing'}
                                                                </span>
                                                                {recentClip.status === 'completed' && firstClipHref(recentClip) && (
                                                                    <a
                                                                        href={firstClipHref(recentClip)}
                                                                        download
                                                                        className="inline-flex items-center gap-1 text-[10px] font-bold bg-white text-ink px-2 py-0.5 rounded hover:bg-neutral-100 transition-colors shadow-xs"
                                                                    >
                                                                        <Download size={10} />
                                                                        Download
                                                                    </a>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}
                                                    {clippingFeedback[info.name] && (
                                                        <div className="absolute top-2 left-2 right-2 bg-paper/95 border border-[var(--color-accent)] text-ink text-xs p-2.5 rounded-card shadow-lg flex items-center justify-between animate-fade z-20 backdrop-blur">
                                                            <div className="flex items-center gap-2 min-w-0">
                                                                <span className="relative flex h-2 w-2 shrink-0">
                                                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--color-accent)] opacity-75"></span>
                                                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--color-accent)]"></span>
                                                                </span>
                                                                <span className="font-semibold text-xs leading-tight truncate">{clippingFeedback[info.name]}</span>
                                                            </div>
                                                            <button
                                                                type="button"
                                                                onClick={() => setClippingFeedback((prev) => ({ ...prev, [info.name]: null }))}
                                                                className="p-0.5 text-muted hover:text-ink shrink-0 ml-1"
                                                            >
                                                                <X size={12} />
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Card Controls & Real-Time Audio Meter */}
                                                <div className="px-3 py-2.5 bg-paper2 border-t border-rule flex flex-col sm:flex-row items-center justify-between gap-2 text-xs">
                                                    <div className="flex items-center gap-2 w-full sm:w-auto min-w-0">
                                                        {/* Audio VU meter */}
                                                        <div className="flex items-center gap-1.5 bg-paper px-2 py-1 rounded border border-rule shrink-0" title={`Áudio da live: ${audioPeakDb != null ? `${audioPeakDb.toFixed(1)} dB` : 'ativo'}`}>
                                                            <Volume2 size={12} className={audioLevel && audioLevel > 0.4 ? 'text-ok' : 'text-brass'} />
                                                            <div className="flex items-center gap-0.5 h-2.5">
                                                                {[0.15, 0.35, 0.65, 0.85].map((thresh, i) => {
                                                                    const active = (audioLevel ?? 0.3) >= thresh;
                                                                    const barColor = i === 3 ? 'bg-danger' : i === 2 ? 'bg-amber-400' : 'bg-ok';
                                                                    return (
                                                                        <span
                                                                            key={i}
                                                                            className={`w-1 h-full rounded-[1px] transition-all duration-300 ${
                                                                                active ? barColor : 'bg-rule2 opacity-30'
                                                                            }`}
                                                                        />
                                                                    );
                                                                })}
                                                            </div>
                                                            <span className="text-[10px] font-mono text-ink2 font-medium">
                                                                {audioPeakDb != null ? `${audioPeakDb.toFixed(0)} dB` : 'audio ai'}
                                                            </span>
                                                        </div>
                                                        <span className="text-[11px] text-ink2 truncate hidden lg:inline">
                                                            Voz & energia ativas
                                                        </span>
                                                    </div>

                                                    {/* Instant Clip Button */}
                                                    <button
                                                        type="button"
                                                        onClick={() => handleTriggerClip(ch, info.name)}
                                                        disabled={clippingStreams[info.name]}
                                                        className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-full bg-[var(--color-accent)] text-[var(--color-accent-ink)] hover:brightness-95 transition-all shadow-sm shrink-0 cursor-pointer disabled:opacity-50"
                                                        title="Record the current highlight now and send it to the Shorts AI"
                                                    >
                                                        {clippingStreams[info.name] ? (
                                                            <>
                                                                <Loader2 size={13} className="animate-spin" />
                                                                <span>Clipando...</span>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Zap size={13} className="fill-current text-amber-300" />
                                                                <span>Clip now</span>
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Stream tag input (Setup Mode - when not running) */}
                        {!isLiveRunning && (
                            <>
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <label className="block eyebrow text-ink">
                                            Live stream channels or URLs
                                        </label>
                                        <span className="text-[11px] text-muted">
                                            Kick · Twitch · YouTube Live
                                        </span>
                                    </div>
                                    <div className="flex gap-2">
                                        <div className="relative flex-1">
                                            <input
                                                type="text"
                                                value={channelInput}
                                                onChange={(e) => setChannelInput(e.target.value)}
                                                onKeyDown={handleKeyDown}
                                                placeholder="e.g. kick.com/bistecone, twitch.tv/alanzoka, youtube.com/watch?v=..."
                                                disabled={isLiveRunning}
                                                className="input-field"
                                            />
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => handleAddChannel()}
                                            disabled={isLiveRunning || !channelInput.trim()}
                                            className="btn-quiet px-4 shrink-0 font-medium"
                                        >
                                            <Plus size={15} />
                                            Add
                                        </button>
                                    </div>
                                </div>

                                {/* Channel Pills */}
                                <div>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <p className="eyebrow text-ink">
                                            Configured Channels ({distinctChannels.length})
                                        </p>
                                        {distinctChannels.length > 0 && !isLiveRunning && (
                                            <button
                                                type="button"
                                                onClick={() => setLiveChannels([])}
                                                className="text-[11px] text-muted hover:text-danger transition-colors font-mono"
                                            >
                                                clear all
                                            </button>
                                        )}
                                    </div>

                                    {distinctChannels.length > 0 ? (
                                        <div className="flex flex-wrap gap-2 p-3 rounded-card bg-paper border border-rule2 min-h-[50px] items-center">
                                            {distinctChannels.map((ch) => {
                                                const tag = getPlatformTag(ch);
                                                const chDisplay = typeof ch === 'object' ? (ch.name || ch.channel || ch.url || '') : String(ch);
                                                const pillKey = `pill_${tag.label}_${chDisplay}`;
                                                return (
                                                    <span
                                                        key={pillKey}
                                                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-paper2 border border-rule2 text-ink shadow-sm animate-fade"
                                                    >
                                                        <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded border uppercase tracking-wider ${tag.color}`}>
                                                            {tag.label}
                                                        </span>
                                                        <span className="truncate max-w-[220px]">{chDisplay}</span>
                                                        {!isLiveRunning && (
                                                            <button
                                                                type="button"
                                                                onClick={() => handleRemoveChannel(ch)}
                                                                aria-label={`Remove ${chDisplay}`}
                                                                className="p-0.5 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors ml-0.5"
                                                            >
                                                                <X size={12} />
                                                            </button>
                                                        )}
                                                    </span>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="p-4 rounded-card bg-paper border border-dashed border-rule2 text-center text-xs text-ink2">
                                            <p>No streams added yet. Enter Kick, Twitch, or YouTube Live stream links above.</p>
                                            <div className="mt-2.5 flex flex-wrap justify-center items-center gap-1.5">
                                                <span className="text-muted text-[11px]">Popular:</span>
                                                {['kick.com/bistecone', 'kick.com/coringa', 'twitch.tv/alanzoka', 'youtube.com/@channel/live'].map((preset) => (
                                                    <button
                                                        key={preset}
                                                        type="button"
                                                        onClick={() => handleAddChannel(preset)}
                                                        className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-paper2 border border-rule2 text-ink2 hover:text-ink hover:border-brass transition-colors"
                                                    >
                                                        + {preset}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </>
                        )}

                        {/* Friendly notification/helper copy */}
                        <div className="rounded-card p-3.5 bg-paper2 border border-rule2 text-xs leading-relaxed text-ink2 space-y-2">
                            <div className="flex items-center gap-1.5 font-medium text-ink">
                                <Sparkles size={14} className="text-brass shrink-0" />
                                <span>How live clips are made:</span>
                            </div>
                            <p>
                                OpenShorts listens to each live's audio in real time. When a spike happens (or when you click <strong>Clip now</strong>), the system:
                            </p>
                            <ol className="list-decimal pl-5 space-y-1 text-ink leading-relaxed">
                                <li>Records the stretch around the spike straight from the live's HLS stream, starting 15 seconds before it was detected and running up to 120 seconds, without recording the whole stream.</li>
                                <li>Sends it through the same AI pipeline as the Clip Generator (Whisper, Gemini viral scoring, 9:16 vertical reframing with the streamer on top and gameplay below, captions).</li>
                                <li>Each finished clip shows on the right with its viral score, the same way the Clip Generator shows its clips.</li>
                            </ol>
                        </div>

                        {/* Live error alert if any with clear, actionable message */}
                        {liveError && (
                            <div className="p-3.5 rounded-card bg-red-500/10 border border-danger/40 text-xs flex items-start gap-2.5 animate-fade" role="alert">
                                <AlertCircle size={16} className="text-danger shrink-0 mt-0.5" />
                                <div className="flex-1 min-w-0">
                                    <p className="font-semibold text-danger">Monitoring Request Failed</p>
                                    <p className="text-ink mt-0.5 leading-relaxed">{liveError}</p>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => setLiveError(null)}
                                    className="p-1 text-muted hover:text-ink rounded-full transition-colors shrink-0"
                                    aria-label="Dismiss error"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        )}

                        {/* Start / Stop Monitoring Button (NO manual 'Clip Now' button) */}
                        {isLiveRunning ? (
                            <button
                                type="button"
                                onClick={handleStopLive}
                                disabled={liveStarting}
                                className="w-full btn-danger mt-2 flex items-center justify-center gap-2 py-3 font-medium cursor-pointer"
                            >
                                {liveStarting ? (
                                    <>
                                        <Loader2 size={16} className="animate-spin" />
                                        Stopping Live Monitoring...
                                    </>
                                ) : (
                                    <>
                                        <Square size={14} className="fill-current" />
                                        Stop Monitoring
                                    </>
                                )}
                            </button>
                        ) : (
                            <button
                                type="button"
                                onClick={handleStartLive}
                                disabled={liveStarting || distinctChannels.length === 0}
                                className="w-full btn-primary mt-2 flex items-center justify-center gap-2 font-medium cursor-pointer"
                            >
                                {liveStarting ? (
                                    <>
                                        <Loader2 size={16} className="animate-spin" />
                                        Starting Live Monitor...
                                    </>
                                ) : (
                                    <>
                                        <Radio size={16} />
                                        Start Live Monitoring ({distinctChannels.length} stream{distinctChannels.length === 1 ? '' : 's'})
                                    </>
                                )}
                            </button>
                        )}
                    </div>
                )}

                {/* FILE AND URL WORKFLOW CONTROLS (Cost, Output Format, Advanced, Generate Clips) */}
                {mode !== 'live' && (
                    <>
                        <SourceCost
                            mode={mode}
                            url={url}
                            fileSeconds={fileSeconds}
                            range={range}
                            onRangeChange={setRange}
                        />

                        {/* Output format selector */}
                        <div className="mt-5" data-tutorial="output-format">
                            <p className="eyebrow mb-2">Output format</p>
                            <div className="grid grid-cols-3 gap-2">
                                {[
                                    { value: 'vertical', label: '9:16', hint: 'Shorts · Reels · TikTok', w: 18, h: 32 },
                                    { value: 'square', label: '1:1', hint: 'Feed posts', w: 28, h: 28 },
                                    { value: 'horizontal', label: '16:9', hint: 'Keep landscape · YouTube', w: 36, h: 20 },
                                ].map((f) => {
                                    const active = outputFormat === f.value;
                                    return (
                                        <button
                                            key={f.value}
                                            type="button"
                                            onClick={() => setOutputFormat(f.value)}
                                            className={`py-3 px-2 rounded-input border flex flex-col items-center gap-2 transition-colors
                                                ${active ? 'border-[color:var(--color-accent)] text-ink font-semibold' : 'border-rule2 text-ink2 hover:border-[color:var(--color-accent)]'}`}
                                        >
                                            <span
                                                className="rounded-[3px] border-2 transition-colors"
                                                style={{
                                                    width: `${f.w}px`,
                                                    height: `${f.h}px`,
                                                    borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                                                    backgroundColor: active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent',
                                                }}
                                            />
                                            <span className="block font-mono text-sm leading-none">{f.label}</span>
                                            <span className="block text-[11px] sm:text-[10px] leading-tight text-center text-ink2 font-medium">{f.hint}</span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Advanced generation controls — collapsed by default */}
                        <div className="mt-4">
                            <button
                                type="button"
                                onClick={() => setShowAdvanced((v) => !v)}
                                className="flex items-center gap-1.5 text-xs text-ink2 font-medium hover:text-ink lowercase transition-colors"
                            >
                                <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                                advanced options
                                {(targetClips || clipMinSeconds || clipMaxSeconds || !autoHook) && (
                                    <span className="text-brass">·</span>
                                )}
                            </button>
                            {showAdvanced && (
                                <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2 animate-fade">
                                    <div>
                                        <p className="eyebrow mb-1.5 text-ink font-semibold">clips to aim for</p>
                                        <input
                                            type="number" min="1" max="15" step="1"
                                            value={targetClips}
                                            onChange={(e) => setTargetClips(e.target.value)}
                                            placeholder="auto"
                                            className="input-field"
                                        />
                                    </div>
                                    <div>
                                        <p className="eyebrow mb-1.5 text-ink font-semibold">min length (s)</p>
                                        <input
                                            type="number" min="15" max="120" step="1"
                                            value={clipMinSeconds}
                                            onChange={(e) => setClipMinSeconds(e.target.value)}
                                            placeholder="15"
                                            className="input-field"
                                        />
                                    </div>
                                    <div>
                                        <p className="eyebrow mb-1.5 text-ink font-semibold">max length (s)</p>
                                        <input
                                            type="number" min="15" max="120" step="1"
                                            value={clipMaxSeconds}
                                            onChange={(e) => setClipMaxSeconds(e.target.value)}
                                            placeholder="120"
                                            className="input-field"
                                        />
                                    </div>
                                    <p className="col-span-1 sm:col-span-3 text-[11px] leading-relaxed text-ink2 font-medium">
                                        Targets, not guarantees: the AI returns fewer clips when the
                                        material doesn&apos;t hold them. Leave blank to let it decide.
                                    </p>
                                    <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                        <span className="text-xs text-ink font-medium">vertical layout</span>
                                        <select
                                            value={layout}
                                            onChange={(e) => setLayout(e.target.value)}
                                            className="input-field !w-auto text-xs py-1.5"
                                            aria-label="vertical layout"
                                        >
                                            <option value="auto">Auto (AI picks per video)</option>
                                            <option value="split">Two speakers stacked</option>
                                            <option value="screencast">Screen over presenter</option>
                                            <option value="none">Single crop only</option>
                                        </select>
                                    </div>
                                    <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                        <label className="flex items-center gap-2 text-xs text-ink font-medium cursor-pointer select-none">
                                            <input
                                                type="checkbox"
                                                checked={autoHook}
                                                onChange={(e) => setAutoHook(e.target.checked)}
                                                className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                                            />
                                            auto hook titles on clips
                                        </label>
                                        {autoHook && (
                                            <select
                                                value={autoHookStyle}
                                                onChange={(e) => setAutoHookStyle(e.target.value)}
                                                className="input-field !w-auto text-xs py-1.5"
                                            >
                                                <option value="pill">Pills</option>
                                                <option value="classic">Classic</option>
                                                <option value="dark">Dark</option>
                                                <option value="yellow">Yellow</option>
                                                <option value="red">Red</option>
                                                <option value="outline">Outline</option>
                                                <option value="outline_yellow">Outline+</option>
                                            </select>
                                        )}
                                        {autoHook && (
                                            <p className="w-full text-[11px] leading-relaxed text-muted">
                                                You can change each clip&apos;s hook text, style, position and size
                                                afterwards with its hook button.
                                            </p>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        <label className="flex items-start gap-2.5 mt-5 text-left text-[13px] sm:text-xs leading-relaxed text-ink2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={acknowledged}
                                onChange={(e) => setAcknowledged(e.target.checked)}
                                className="mt-0.5 w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                            />
                            <span>
                                I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-ink underline underline-offset-2 hover:text-brass transition-colors font-medium" onClick={(e) => e.stopPropagation()}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-ink underline underline-offset-2 hover:text-brass transition-colors font-medium" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
                            </span>
                        </label>

                        <button
                            type="submit"
                            data-tutorial="generate"
                            disabled={isProcessing || !acknowledged || (mode === 'url' && !url) || (mode === 'file' && (!file || fileTooShort))}
                            className="w-full btn-primary mt-4 cursor-pointer"
                        >
                            {isProcessing ? (
                                <>
                                    <Loader2 size={16} className="animate-spin" />
                                    Processing Video...
                                </>
                            ) : (
                                <>
                                    Generate Clips
                                </>
                            )}
                        </button>
                    </>
                )}
            </form>
        </div>
    );
}
