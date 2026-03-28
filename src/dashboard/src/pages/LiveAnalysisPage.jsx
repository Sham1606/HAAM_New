import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Mic, MicOff, Square, AlertTriangle, Activity,
    Brain, Volume2, FileText, TrendingUp, TrendingDown, Minus
} from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Cell, BarChart, Bar
} from 'recharts';
import { useAuth } from '../services/AuthContext';

// ─── Constants ────────────────────────────────────────────────────────────────
const WS_BASE = 'ws://localhost:8000/ws/mic-stream';
const TARGET_SR = 16000;

const EMOTION_CONFIG = {
    neutral: { color: '#94a3b8', bg: 'bg-slate-100', text: 'text-slate-700', icon: '😐', label: 'Neutral' },
    anger: { color: '#ef4444', bg: 'bg-red-100', text: 'text-red-700', icon: '😠', label: 'Anger' },
    disgust: { color: '#8b5cf6', bg: 'bg-purple-100', text: 'text-purple-700', icon: '🤢', label: 'Disgust' },
    fear: { color: '#f59e0b', bg: 'bg-amber-100', text: 'text-amber-700', icon: '😨', label: 'Fear' },
    sadness: { color: '#3b82f6', bg: 'bg-blue-100', text: 'text-blue-700', icon: '😢', label: 'Sadness' },
};

const getRiskColor = (risk) => {
    if (risk >= 0.6) return { bg: 'bg-red-50', text: 'text-red-700', label: 'High Risk', border: 'border-red-200' };
    if (risk >= 0.3) return { bg: 'bg-amber-50', text: 'text-amber-700', label: 'Medium Risk', border: 'border-amber-200' };
    return { bg: 'bg-green-50', text: 'text-green-700', label: 'Low Risk', border: 'border-green-200' };
};

// ─── Resampler helper (Float32Array at browserSR → 16000Hz) ──────────────────
function downsample(buffer, fromSR, toSR) {
    if (fromSR === toSR) return buffer;
    const ratio = fromSR / toSR;
    const outLength = Math.floor(buffer.length / ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
        out[i] = buffer[Math.floor(i * ratio)];
    }
    return out;
}

// ─── Emotion Distribution Mini-Bar ────────────────────────────────────────────
const EmotionDistBar = ({ distribution }) => {
    const data = Object.entries(EMOTION_CONFIG).map(([key, cfg]) => ({
        name: cfg.label, value: Math.round((distribution[key] || 0) * 100), color: cfg.color,
    }));
    return (
        <ResponsiveContainer width="100%" height={110}>
            <BarChart data={data} margin={{ left: -10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {data.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};

// ─── Emotion Timeline Chart ────────────────────────────────────────────────────
const EmotionTimeline = ({ turns }) => {
    const emotionToScore = { neutral: 0, sadness: -1, fear: -2, disgust: -2, anger: -3 };
    const startIndex = Math.max(0, turns.length - 15);
    const data = turns.slice(-15).map((t, i) => ({
        turn: `T${startIndex + i + 1}`, score: emotionToScore[t.emotion] ?? 0, emotion: t.emotion,
    }));
    return (
        <ResponsiveContainer width="100%" height={130}>
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="turn" tick={{ fontSize: 10 }} />
                <YAxis domain={[-3, 1]} ticks={[-3, -2, -1, 0, 1]}
                    tickFormatter={v => ['😠', '😨', '😢', '😐', '😊'][v + 3]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v, _, p) => [p.payload.emotion, 'Emotion']} />
                <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2}
                    dot={{ r: 4, fill: '#6366f1' }} activeDot={{ r: 6 }} isAnimationActive={false} />
            </LineChart>
        </ResponsiveContainer>
    );
};

// ─── Main Page ────────────────────────────────────────────────────────────────
const LiveAnalysisPage = () => {
    const { user } = useAuth();
    const [status, setStatus] = useState('idle');     // idle | connecting | recording | error
    const [turns, setTurns] = useState([]);
    const [currentTurn, setCurrentTurn] = useState(null);
    const [session, setSession] = useState(null);
    const [errorMsg, setErrorMsg] = useState('');
    const [micLevel, setMicLevel] = useState(0);
    const [micStatus, setMicStatus] = useState('');   // '' | 'Listening...' | 'Processing...'

    const wsRef = useRef(null);
    const audioCtxRef = useRef(null);
    const streamRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);
    const scrollRef = useRef(null);

    // Auto-scroll transcript
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [turns]);

    // Mic level animation
    const updateMicLevel = useCallback(() => {
        if (analyserRef.current) {
            const data = new Uint8Array(analyserRef.current.fftSize);
            analyserRef.current.getByteTimeDomainData(data);
            const rms = Math.sqrt(data.reduce((s, v) => s + (v - 128) ** 2, 0) / data.length);
            setMicLevel(Math.min(rms / 30, 1));
        }
        animFrameRef.current = requestAnimationFrame(updateMicLevel);
    }, []);

    const stopAll = useCallback(() => {
        cancelAnimationFrame(animFrameRef.current);
        if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
        if (sourceRef.current) { sourceRef.current.disconnect(); sourceRef.current = null; }
        if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
        if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
        setMicLevel(0);
        setStatus('idle');
    }, []);

    const startRecording = useCallback(async () => {
        setErrorMsg('');
        setStatus('connecting');

        // 1. Connect WebSocket with agent_id for admin monitoring
        const wsUrl = user?.id ? `${WS_BASE}?agent_id=${user.id}` : WS_BASE;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = async () => {
            // 2. Request mic permission
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                streamRef.current = stream;

                // 3. Web Audio pipeline
                const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
                audioCtxRef.current = ctx;

                const source = ctx.createMediaStreamSource(stream);
                sourceRef.current = source;

                // Analyser for level visualisation
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 256;
                analyserRef.current = analyser;
                source.connect(analyser);

                await ctx.audioWorklet.addModule('/audio-processor.js');
                const workletNode = new AudioWorkletNode(ctx, 'mic-audio-processor');
                processorRef.current = workletNode;

                workletNode.port.onmessage = (e) => {
                    if (ws.readyState !== WebSocket.OPEN) return;
                    const raw = e.data;
                    const resampled = downsample(raw, ctx.sampleRate, TARGET_SR);
                    ws.send(resampled.buffer);
                };

                source.connect(workletNode);
                workletNode.connect(ctx.destination);

                setStatus('recording');
                updateMicLevel();
            } catch (err) {
                setErrorMsg(`Mic access denied: ${err.message}`);
                setStatus('error');
                ws.close();
            }
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'listening') {
                    setMicStatus('🎤 Listening…');
                } else if (data.type === 'processing') {
                    setMicStatus('⚙️ Processing…');
                } else if (data.type === 'turn_result') {
                    setMicStatus('');
                    const turn = {
                        id: Date.now(),
                        emotion: data.emotion,
                        confidence: data.confidence,
                        transcript: data.transcript,
                        distribution: data.emotion_distribution || {},
                        fusion: data.fusion_weights || {},
                    };
                    setCurrentTurn(turn);
                    setTurns(prev => [...prev, turn]);
                    setSession(data.session);
                } else if (data.type === 'error') {
                    setErrorMsg(data.message);
                    setStatus('error');
                    stopAll();
                }
            } catch (e) { /* ignore parse errors */ }
        };

        ws.onerror = () => {
            setErrorMsg('WebSocket connection failed. Is the API running on port 8000?');
            setStatus('error');
            stopAll();
        };

        ws.onclose = () => {
            if (status === 'recording') stopAll();
        };
    }, [status, stopAll, updateMicLevel]);

    const emo = currentTurn ? EMOTION_CONFIG[currentTurn.emotion] || EMOTION_CONFIG.neutral : null;
    const risk = session ? getRiskColor(session.risk_score) : null;
    const isRec = status === 'recording';

    return (
        <div className="space-y-6 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-6 text-white shadow-lg">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-3">
                            <Brain className="h-7 w-7" /> Live Mic Analysis
                        </h1>
                        <p className="text-indigo-200 mt-1 text-sm">
                            Real-time emotion detection from your microphone · HAAM Hybrid Model
                        </p>
                    </div>
                    {/* Mic Status Badge */}
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full font-semibold text-sm ${isRec ? 'bg-white/20 text-white' : 'bg-white/10 text-indigo-200'
                        }`}>
                        <span className={`w-2 h-2 rounded-full ${isRec ? 'bg-red-400 animate-pulse' : 'bg-gray-300'}`} />
                        {status === 'idle' && 'Ready'}
                        {status === 'connecting' && 'Connecting…'}
                        {status === 'recording' && 'LIVE'}
                        {status === 'error' && 'Error'}
                    </div>
                </div>

                {/* Mic Level Bar */}
                {isRec && (
                    <div className="mt-4">
                        <p className="text-xs text-indigo-200 mb-1">Microphone Level</p>
                        <div className="w-full bg-white/20 rounded-full h-2">
                            <div
                                className="h-2 rounded-full bg-white transition-all duration-75"
                                style={{ width: `${micLevel * 100}%` }}
                            />
                        </div>
                    </div>
                )}

                {/* Controls */}
                <div className="flex gap-3 mt-4">
                    {!isRec ? (
                        <button
                            onClick={startRecording}
                            disabled={status === 'connecting'}
                            className="flex items-center gap-2 px-6 py-2.5 bg-white text-indigo-700 font-bold rounded-xl hover:bg-indigo-50 transition disabled:opacity-50"
                        >
                            <Mic className="h-4 w-4" />
                            {status === 'connecting' ? 'Connecting…' : 'Start Listening'}
                        </button>
                    ) : (
                        <button
                            onClick={stopAll}
                            className="flex items-center gap-2 px-6 py-2.5 bg-red-500 text-white font-bold rounded-xl hover:bg-red-600 transition"
                        >
                            <Square className="h-4 w-4" /> Stop
                        </button>
                    )}
                    {turns.length > 0 && !isRec && (
                        <button
                            onClick={() => { setTurns([]); setCurrentTurn(null); setSession(null); }}
                            className="px-4 py-2.5 bg-white/20 text-white font-semibold rounded-xl hover:bg-white/30 transition text-sm"
                        >
                            Clear
                        </button>
                    )}
                </div>

                {errorMsg && (
                    <div className="mt-3 flex items-center gap-2 text-red-200 text-sm">
                        <AlertTriangle className="h-4 w-4 flex-shrink-0" /> {errorMsg}
                    </div>
                )}
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* LEFT: Current Emotion + Distribution */}
                <div className="space-y-4">
                    {/* Current Emotion Card */}
                    <div className={`rounded-2xl border p-6 transition-all duration-500 ${emo ? emo.bg + ' border-transparent' : 'bg-white border-gray-100'}`}>
                        <p className="text-xs font-bold uppercase tracking-wider text-gray-400 mb-3">Current Emotion</p>
                        {currentTurn && emo ? (
                            <div className="text-center">
                                <div className="text-6xl mb-3">{emo.icon}</div>
                                <p className={`text-2xl font-bold capitalize ${emo.text}`}>{currentTurn.emotion}</p>
                                <p className="text-sm text-gray-500 mt-1">
                                    {Math.round(currentTurn.confidence * 100)}% confidence
                                </p>
                            </div>
                        ) : (
                            <div className="text-center text-gray-400 py-6">
                                <MicOff className="h-12 w-12 mx-auto mb-3 opacity-30" />
                                <p className="text-sm">{isRec ? 'Listening…' : 'Press Start to begin'}</p>
                            </div>
                        )}
                    </div>

                    {/* Modality Split */}
                    {currentTurn?.fusion && (
                        <div className="bg-white rounded-2xl border border-gray-100 p-4">
                            <p className="text-xs font-bold uppercase tracking-wider text-gray-400 mb-3">
                                Modality Attention
                            </p>
                            {[
                                { label: 'Acoustic', icon: <Volume2 className="h-3 w-3 text-blue-500" />, val: currentTurn.fusion.acoustic ?? 0.5, color: 'bg-blue-500' },
                                { label: 'Text', icon: <FileText className="h-3 w-3 text-purple-500" />, val: currentTurn.fusion.text ?? 0.5, color: 'bg-purple-500' },
                            ].map(m => (
                                <div key={m.label} className="mb-2">
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="flex items-center gap-1">{m.icon} {m.label}</span>
                                        <span className="font-bold">{Math.round(m.val * 100)}%</span>
                                    </div>
                                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                        <div className={`h-2 rounded-full ${m.color} transition-all duration-500`}
                                            style={{ width: `${m.val * 100}%` }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Session Risk */}
                    {session && (
                        <div className={`rounded-2xl border p-4 ${risk.bg} ${risk.border}`}>
                            <p className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Session Risk</p>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className={`text-2xl font-bold ${risk.text}`}>
                                        {Math.round(session.risk_score * 100)}%
                                    </p>
                                    <p className={`text-sm font-semibold ${risk.text}`}>{risk.label}</p>
                                </div>
                                <AlertTriangle className={`h-8 w-8 ${risk.text}`} />
                            </div>
                            <div className="mt-2 flex items-center gap-1 text-xs text-gray-500">
                                {session.trend?.includes('Worsening') ? <TrendingDown className="h-3 w-3 text-red-500" /> :
                                    session.trend?.includes('Improving') ? <TrendingUp className="h-3 w-3 text-green-500" /> :
                                        <Minus className="h-3 w-3" />}
                                Trend: <strong>{session.trend}</strong> · {session.turn_count} turns
                            </div>
                        </div>
                    )}
                </div>

                {/* MIDDLE: Transcript Feed */}
                <div className="bg-white rounded-2xl border border-gray-100 shadow-sm flex flex-col" style={{ minHeight: '460px' }}>
                    <div className="p-4 border-b border-gray-100 flex items-center gap-2">
                        <FileText className="h-4 w-4 text-indigo-500" />
                        <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider">Live Transcript</h3>
                        {micStatus && (
                            <span className="ml-2 text-xs text-indigo-500 font-semibold animate-pulse">{micStatus}</span>
                        )}
                        {isRec && <span className="ml-auto text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-semibold animate-pulse">● REC</span>}
                    </div>
                    <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3 max-h-96">
                        {turns.length === 0 ? (
                            <p className="text-sm text-gray-300 italic text-center mt-10">
                                Transcript will appear here…
                            </p>
                        ) : (
                            turns.map((t, i) => {
                                const cfg = EMOTION_CONFIG[t.emotion] || EMOTION_CONFIG.neutral;
                                return (
                                    <div key={t.id} className={`rounded-xl p-3 border ${cfg.bg} border-transparent`}>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span>{cfg.icon}</span>
                                            <span className={`text-xs font-bold uppercase ${cfg.text}`}>{t.emotion}</span>
                                            <span className="ml-auto text-xs text-gray-400">{Math.round(t.confidence * 100)}%</span>
                                        </div>
                                        <p className="text-sm text-gray-700 leading-relaxed">
                                            {t.transcript || <span className="italic text-gray-400">…</span>}
                                        </p>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>

                {/* RIGHT: Emotion Timeline + Session Distribution */}
                <div className="space-y-4">
                    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <Activity className="h-4 w-4 text-indigo-500" />
                            <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider">Emotion Timeline</h3>
                        </div>
                        {turns.length > 1 ? (
                            <EmotionTimeline turns={turns} />
                        ) : (
                            <p className="text-center text-sm text-gray-300 italic py-10">
                                Chart appears after 2+ turns
                            </p>
                        )}
                    </div>

                    {currentTurn?.distribution && (
                        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
                            <div className="flex items-center gap-2 mb-3">
                                <Brain className="h-4 w-4 text-indigo-500" />
                                <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider">Turn Distribution</h3>
                            </div>
                            <EmotionDistBar distribution={currentTurn.distribution} />
                        </div>
                    )}

                    {session?.emotion_counts && (
                        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
                            <p className="text-xs font-bold uppercase tracking-wider text-gray-400 mb-3">Session Summary</p>
                            <div className="space-y-1.5">
                                {Object.entries(session.emotion_counts)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([emo, count]) => {
                                        const cfg = EMOTION_CONFIG[emo] || EMOTION_CONFIG.neutral;
                                        const pct = Math.round((count / session.turn_count) * 100);
                                        return (
                                            <div key={emo} className="flex items-center gap-2">
                                                <span className="w-5 text-center">{cfg.icon}</span>
                                                <span className="text-xs text-gray-600 w-16 capitalize">{emo}</span>
                                                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                                    <div className="h-2 rounded-full transition-all duration-300"
                                                        style={{ width: `${pct}%`, background: cfg.color }} />
                                                </div>
                                                <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
                                            </div>
                                        );
                                    })}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LiveAnalysisPage;
