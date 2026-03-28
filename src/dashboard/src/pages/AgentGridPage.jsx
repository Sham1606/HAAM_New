/**
 * AgentGrid: Admin dashboard showing live agent cards with WebSocket status updates.
 * 
 * Each card shows: avatar, name, status badge, LIVE emotion prediction,
 * latest transcript snippet, AI feedback chip, session stats (turn count, risk, trend),
 * and emotion distribution bar.
 */

import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../services/AuthContext';
import {
    Users, Wifi, WifiOff, Brain, Shield,
    RefreshCw, Clock, Activity, AlertTriangle,
    TrendingUp, TrendingDown, Minus, Phone, MessageSquare
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/agents';

const STATUS_CONFIG = {
    'online': { color: 'bg-green-400', text: 'text-green-700', bg: 'bg-green-50', label: '🟢 Online', border: 'border-green-200', ring: 'ring-green-300' },
    'on-call': { color: 'bg-amber-400', text: 'text-amber-700', bg: 'bg-amber-50', label: '🟡 On Call', border: 'border-amber-200', ring: 'ring-amber-300' },
    'offline': { color: 'bg-gray-300', text: 'text-gray-500', bg: 'bg-gray-50', label: '⚪ Offline', border: 'border-gray-200', ring: 'ring-gray-200' },
};

const EMOTION_CONFIG = {
    neutral: { icon: '😐', color: '#94a3b8', bg: 'bg-slate-50', text: 'text-slate-700' },
    anger: { icon: '😠', color: '#ef4444', bg: 'bg-red-50', text: 'text-red-700' },
    disgust: { icon: '🤢', color: '#8b5cf6', bg: 'bg-purple-50', text: 'text-purple-700' },
    fear: { icon: '😨', color: '#f59e0b', bg: 'bg-amber-50', text: 'text-amber-700' },
    sadness: { icon: '😢', color: '#3b82f6', bg: 'bg-blue-50', text: 'text-blue-700' },
};

const AgentCard = ({ agent, liveData }) => {
    const status = liveData?.status || agent.status || 'offline';
    const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.offline;
    const emotion = liveData?.live_emotion;
    const emoCfg = EMOTION_CONFIG[emotion] || null;
    const feedback = liveData?.feedback;
    const transcript = liveData?.transcript;
    const confidence = liveData?.confidence;
    const turnCount = liveData?.turn_count;
    const riskScore = liveData?.risk_score;
    const trend = liveData?.trend;
    const dominantEmotion = liveData?.dominant_emotion;
    const emotionCounts = liveData?.emotion_counts;

    const timeAgo = (isoStr) => {
        if (!isoStr) return 'Never';
        const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        return `${Math.floor(diff / 3600)}h ago`;
    };

    const isLive = status === 'on-call' && emotion;

    return (
        <div className={`rounded-2xl border overflow-hidden transition-all duration-500 shadow-sm hover:shadow-lg ${isLive
            ? `${emoCfg?.bg || 'bg-white'} border-transparent ring-2 ${cfg.ring}`
            : `bg-white ${cfg.border}`
            }`}>
            {/* Header Bar */}
            <div className={`px-4 py-3 flex items-center justify-between ${isLive ? 'bg-white/60' : 'border-b border-gray-50'}`}>
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 flex items-center justify-center text-white font-bold text-sm shadow-sm">
                            {(agent.display_name || agent.username || '?')[0].toUpperCase()}
                        </div>
                        <div className={`absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full border-2 border-white ${cfg.color} ${status !== 'offline' ? 'animate-pulse' : ''}`} />
                    </div>
                    <div>
                        <p className="font-bold text-gray-800 text-sm">{agent.display_name || agent.username}</p>
                        <p className="text-xs text-gray-400">{agent.id}</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {agent.role === 'admin' && (
                        <span className="flex items-center gap-1 text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-semibold">
                            <Shield className="h-2.5 w-2.5" /> Admin
                        </span>
                    )}
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${cfg.text} ${cfg.bg}`}>
                        {cfg.label}
                    </span>
                </div>
            </div>

            {/* Body */}
            <div className="p-4 space-y-3">
                {/* Live Emotion Display */}
                {isLive ? (
                    <div className="text-center py-2">
                        <div className="text-5xl mb-2 animate-bounce" style={{ animationDuration: '2s' }}>
                            {emoCfg?.icon || '❓'}
                        </div>
                        <p className={`text-lg font-bold capitalize ${emoCfg?.text || 'text-gray-700'}`}>
                            {emotion}
                        </p>
                        {confidence != null && (
                            <p className="text-xs text-gray-400 mt-0.5">
                                {Math.round(confidence * 100)}% confidence
                            </p>
                        )}
                    </div>
                ) : (
                    <div className="text-center py-4 text-gray-300">
                        <Activity className="h-10 w-10 mx-auto mb-2 opacity-20" />
                        <p className="text-xs">{status === 'online' ? 'Waiting for live session…' : 'Agent offline'}</p>
                    </div>
                )}

                {/* Last transcript */}
                {transcript && (
                    <div className="bg-white/70 rounded-xl px-3 py-2 border border-gray-100">
                        <p className="text-xs text-gray-400 mb-0.5 flex items-center gap-1">
                            <MessageSquare className="h-3 w-3" /> Last utterance
                        </p>
                        <p className="text-xs text-gray-600 italic leading-relaxed line-clamp-2">"{transcript}"</p>
                    </div>
                )}

                {/* AI Feedback Chip */}
                {feedback && (
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl px-3 py-2 border border-indigo-100">
                        <p className="text-xs text-gray-400 mb-0.5 flex items-center gap-1">
                            <Brain className="h-3 w-3 text-indigo-500" /> AI Coaching
                        </p>
                        <p className="text-xs text-gray-700 leading-relaxed">{feedback}</p>
                    </div>
                )}

                {/* Session stats bar */}
                {turnCount > 0 && (
                    <div className="flex items-center justify-between text-xs bg-gray-50 rounded-xl px-3 py-2">
                        <div className="flex items-center gap-1 text-gray-500">
                            <Phone className="h-3 w-3" />
                            <span>{turnCount} turns</span>
                        </div>
                        <div className="flex items-center gap-1">
                            {trend?.includes('Worsening') ? <TrendingDown className="h-3 w-3 text-red-500" /> :
                                trend?.includes('Improving') ? <TrendingUp className="h-3 w-3 text-green-500" /> :
                                    <Minus className="h-3 w-3 text-gray-400" />}
                            <span className="text-gray-500">{trend || 'Stable'}</span>
                        </div>
                        {riskScore != null && (
                            <span className={`font-bold ${riskScore >= 0.6 ? 'text-red-600' : riskScore >= 0.3 ? 'text-amber-600' : 'text-green-600'}`}>
                                Risk: {Math.round(riskScore * 100)}%
                            </span>
                        )}
                    </div>
                )}

                {/* Emotion distribution mini bars */}
                {emotionCounts && Object.keys(emotionCounts).length > 0 && turnCount > 0 && (
                    <div className="space-y-1">
                        {Object.entries(emotionCounts)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 4)
                            .map(([emo, count]) => {
                                const ec = EMOTION_CONFIG[emo] || EMOTION_CONFIG.neutral;
                                const pct = Math.round((count / turnCount) * 100);
                                return (
                                    <div key={emo} className="flex items-center gap-2">
                                        <span className="text-xs w-4">{ec.icon}</span>
                                        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                            <div className="h-1.5 rounded-full transition-all duration-500"
                                                style={{ width: `${pct}%`, background: ec.color }} />
                                        </div>
                                        <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
                                    </div>
                                );
                            })}
                    </div>
                )}

                {/* Last ping */}
                <div className="flex items-center gap-1.5 text-xs text-gray-300 pt-1">
                    <Clock className="h-3 w-3" />
                    {timeAgo(liveData?.last_update || agent.last_ping)}
                </div>
            </div>
        </div>
    );
};


const AgentGridPage = () => {
    const { token, isAdmin } = useAuth();
    const [agents, setAgents] = useState([]);
    const [liveStates, setLiveStates] = useState({});
    const [wsConnected, setWsConnected] = useState(false);
    const [loading, setLoading] = useState(true);
    const wsRef = useRef(null);

    // Fetch registered agents
    useEffect(() => {
        setLoading(true);
        axios.get(`${API_BASE}/api/agents/registered`)
            .then(res => { setAgents(res.data); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    // WebSocket connection for live updates
    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => setWsConnected(true);

            ws.onmessage = (e) => {
                try {
                    const msg = JSON.parse(e.data);
                    if (msg.type === 'snapshot') {
                        setLiveStates(msg.agents || {});
                    } else if (msg.type === 'agent_update') {
                        setLiveStates(prev => ({
                            ...prev,
                            [msg.agent_id]: { ...prev[msg.agent_id], ...msg.data },
                        }));
                    }
                } catch (_) { }
            };

            ws.onclose = () => {
                setWsConnected(false);
                setTimeout(connect, 3000);
            };
            ws.onerror = () => ws.close();
        };

        connect();
        return () => { if (wsRef.current) wsRef.current.close(); };
    }, []);

    const onCallCount = agents.filter(a => (liveStates[a.id]?.status || a.status) === 'on-call').length;
    const onlineCount = agents.filter(a => {
        const st = liveStates[a.id]?.status || a.status;
        return st === 'online' || st === 'on-call';
    }).length;

    return (
        <div className="space-y-6 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-2xl p-6 text-white shadow-lg">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-3">
                            <Users className="h-7 w-7" /> Agent Monitoring Grid
                        </h1>
                        <p className="text-indigo-200 mt-1 text-sm">
                            Real-time monitoring — see each agent's live emotion, feedback, and risk
                        </p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-center">
                            <p className="text-3xl font-bold">{onlineCount}</p>
                            <p className="text-indigo-200 text-xs">Online</p>
                        </div>
                        <div className="text-center">
                            <p className="text-3xl font-bold">{onCallCount}</p>
                            <p className="text-amber-200 text-xs">On Call</p>
                        </div>
                        <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${wsConnected
                            ? 'bg-green-400/20 text-green-200' : 'bg-red-400/20 text-red-200'
                            }`}>
                            {wsConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
                            {wsConnected ? 'Live' : 'Reconnecting…'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Grid */}
            {loading ? (
                <div className="text-center py-20 text-gray-400">
                    <RefreshCw className="h-8 w-8 mx-auto mb-3 animate-spin opacity-30" />
                    <p>Loading agents…</p>
                </div>
            ) : agents.length === 0 ? (
                <div className="text-center py-20 text-gray-400">
                    <Users className="h-16 w-16 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-semibold">No agents registered</p>
                    <p className="text-sm mt-1">Run <code>python migrate.py</code> to seed demo agents</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                    {agents
                        .filter(a => a.role !== 'admin')  // Only show agent-role users
                        .map(agent => (
                            <AgentCard
                                key={agent.id}
                                agent={agent}
                                liveData={liveStates[agent.id]}
                            />
                        ))}
                </div>
            )}
        </div>
    );
};

export default AgentGridPage;
