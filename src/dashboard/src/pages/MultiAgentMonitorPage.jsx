import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Users, Plus, Mic, MicOff, Square, AlertTriangle,
    Activity, Brain, X, Wifi, WifiOff
} from 'lucide-react';

const WS_URL = 'ws://localhost:8000/ws/mic-stream';
const TARGET_SR = 16000;

const EMOTION_CONFIG = {
    neutral: { color: '#94a3b8', bg: 'from-slate-50 to-slate-100', badge: 'bg-slate-100 text-slate-700', icon: '😐' },
    anger: { color: '#ef4444', bg: 'from-red-50 to-red-100', badge: 'bg-red-100 text-red-700', icon: '😠' },
    disgust: { color: '#8b5cf6', bg: 'from-purple-50 to-purple-100', badge: 'bg-purple-100 text-purple-700', icon: '🤢' },
    fear: { color: '#f59e0b', bg: 'from-amber-50 to-amber-100', badge: 'bg-amber-100 text-amber-700', icon: '😨' },
    sadness: { color: '#3b82f6', bg: 'from-blue-50 to-blue-100', badge: 'bg-blue-100 text-blue-700', icon: '😢' },
};

function downsample(buffer, fromSR, toSR) {
    if (fromSR === toSR) return buffer;
    const ratio = fromSR / toSR;
    const out = new Float32Array(Math.floor(buffer.length / ratio));
    for (let i = 0; i < out.length; i++) out[i] = buffer[Math.floor(i * ratio)];
    return out;
}

// ─── Single Agent Tile ────────────────────────────────────────────────────────
const AgentTile = ({ agent, onRemove }) => {
    const [status, setStatus] = useState('idle');
    const [turn, setTurn] = useState(null);
    const [session, setSession] = useState(null);
    const [micLevel, setMicLevel] = useState(0);

    const wsRef = useRef(null);
    const audioCtxRef = useRef(null);
    const streamRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);

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

    const levelLoop = useCallback(() => {
        if (analyserRef.current) {
            const data = new Uint8Array(analyserRef.current.fftSize);
            analyserRef.current.getByteTimeDomainData(data);
            const rms = Math.sqrt(data.reduce((s, v) => s + (v - 128) ** 2, 0) / data.length);
            setMicLevel(Math.min(rms / 28, 1));
        }
        animFrameRef.current = requestAnimationFrame(levelLoop);
    }, []);

    const start = useCallback(async () => {
        setStatus('connecting');
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                streamRef.current = stream;
                const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
                audioCtxRef.current = ctx;
                const source = ctx.createMediaStreamSource(stream);
                sourceRef.current = source;
                const analyser = ctx.createAnalyser(); analyser.fftSize = 256;
                analyserRef.current = analyser;
                source.connect(analyser);
                await ctx.audioWorklet.addModule('/audio-processor.js');
                const workletNode = new AudioWorkletNode(ctx, 'mic-audio-processor');
                processorRef.current = workletNode;
                workletNode.port.onmessage = (e) => {
                    if (ws.readyState !== WebSocket.OPEN) return;
                    const raw = e.data;
                    ws.send(downsample(raw, ctx.sampleRate, TARGET_SR).buffer);
                };
                source.connect(workletNode);
                workletNode.connect(ctx.destination);
                setStatus('recording');
                levelLoop();
            } catch (err) {
                setStatus('error');
                ws.close();
            }
        };

        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'turn_result') {
                    setTurn(data);
                    setSession(data.session);
                }
            } catch (_) { }
        };

        ws.onerror = () => { setStatus('error'); stopAll(); };
        ws.onclose = () => { if (status === 'recording') stopAll(); };
    }, [levelLoop, stopAll, status]);

    // Cleanup on unmount
    useEffect(() => () => stopAll(), [stopAll]);

    const emo = turn ? EMOTION_CONFIG[turn.emotion] || EMOTION_CONFIG.neutral : null;
    const isRec = status === 'recording';
    const risk = session?.risk_score ?? 0;
    const riskClr = risk >= 0.6 ? 'text-red-600' : risk >= 0.3 ? 'text-amber-600' : 'text-green-600';

    return (
        <div className={`rounded-2xl border overflow-hidden transition-all duration-500 ${emo ? `bg-gradient-to-br ${emo.bg} border-transparent` : 'bg-white border-gray-100'
            } shadow-sm`}>
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/50">
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isRec ? 'bg-red-500 animate-pulse' : 'bg-gray-300'}`} />
                    <span className="font-bold text-gray-800 text-sm">{agent.name}</span>
                </div>
                <div className="flex items-center gap-1">
                    {isRec ? (
                        <button onClick={stopAll} className="p-1.5 rounded-lg bg-red-500 text-white hover:bg-red-600 transition">
                            <Square className="h-3 w-3" />
                        </button>
                    ) : (
                        <button onClick={start} className="p-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition">
                            <Mic className="h-3 w-3" />
                        </button>
                    )}
                    <button onClick={onRemove} className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition">
                        <X className="h-3 w-3" />
                    </button>
                </div>
            </div>

            {/* Body */}
            <div className="p-4 space-y-3">
                {/* Mic level */}
                {isRec && (
                    <div className="h-1.5 bg-white/50 rounded-full overflow-hidden">
                        <div className="h-full rounded-full bg-indigo-500 transition-all duration-75"
                            style={{ width: `${micLevel * 100}%` }} />
                    </div>
                )}

                {/* Emotion */}
                <div className="text-center py-2">
                    {turn && emo ? (
                        <>
                            <div className="text-4xl mb-1">{emo.icon}</div>
                            <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded-full ${emo.badge}`}>
                                {turn.emotion}
                            </span>
                            <p className="text-xs text-gray-500 mt-1">{Math.round(turn.confidence * 100)}% confidence</p>
                        </>
                    ) : (
                        <div className="text-gray-300 py-2">
                            <MicOff className="h-8 w-8 mx-auto" />
                            <p className="text-xs mt-1">{isRec ? 'Listening…' : 'Press mic to start'}</p>
                        </div>
                    )}
                </div>

                {/* Transcript */}
                {turn?.transcript && (
                    <p className="text-xs text-gray-600 italic bg-white/60 rounded-lg p-2 leading-relaxed line-clamp-2">
                        "{turn.transcript}"
                    </p>
                )}

                {/* Session stats */}
                {session && (
                    <div className="flex justify-between text-xs bg-white/60 rounded-lg p-2">
                        <span className="text-gray-500">{session.turn_count} turns</span>
                        <span className={`font-bold ${riskClr}`}>Risk: {Math.round(risk * 100)}%</span>
                        <span className="text-gray-500">{session.trend}</span>
                    </div>
                )}

                {status === 'error' && (
                    <p className="text-xs text-red-500 text-center">Mic access denied</p>
                )}
            </div>
        </div>
    );
};

// ─── Main Page ────────────────────────────────────────────────────────────────
let nextId = 1;

const MultiAgentMonitorPage = () => {
    const [agents, setAgents] = useState([]);
    const [newName, setNewName] = useState('');

    const addAgent = () => {
        const name = newName.trim() || `Agent ${nextId}`;
        setAgents(prev => [...prev, { id: nextId++, name }]);
        setNewName('');
    };

    const removeAgent = (id) => setAgents(prev => prev.filter(a => a.id !== id));

    return (
        <div className="space-y-6 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-violet-600 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-3">
                            <Users className="h-7 w-7" /> Multi-Agent Live Monitor
                        </h1>
                        <p className="text-violet-200 mt-1 text-sm">
                            Monitor multiple agents simultaneously — each tile has its own mic stream
                        </p>
                    </div>
                    <div className="text-right">
                        <p className="text-3xl font-bold">{agents.length}</p>
                        <p className="text-violet-200 text-xs">Active Agents</p>
                    </div>
                </div>

                {/* Add Agent form */}
                <div className="mt-4 flex gap-2">
                    <input
                        value={newName}
                        onChange={e => setNewName(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && addAgent()}
                        placeholder="Agent name (e.g. Alice, Agent-7)"
                        className="flex-1 px-4 py-2 rounded-xl text-gray-800 text-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50"
                    />
                    <button
                        onClick={addAgent}
                        className="flex items-center gap-2 px-5 py-2 bg-white text-indigo-700 font-bold rounded-xl hover:bg-indigo-50 transition"
                    >
                        <Plus className="h-4 w-4" /> Add Agent
                    </button>
                </div>
            </div>

            {agents.length === 0 ? (
                <div className="text-center py-20 text-gray-400">
                    <Users className="h-16 w-16 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-semibold">No agents yet</p>
                    <p className="text-sm mt-1">Add an agent above to begin monitoring</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {agents.map(agent => (
                        <AgentTile key={agent.id} agent={agent} onRemove={() => removeAgent(agent.id)} />
                    ))}
                </div>
            )}

            <div className="text-center text-xs text-gray-400 border-t pt-4">
                Each agent tile maintains an independent WebSocket connection to <code>/ws/mic-stream</code>.
                In production, each agent uses a separate device/browser tab.
            </div>
        </div>
    );
};

export default MultiAgentMonitorPage;
