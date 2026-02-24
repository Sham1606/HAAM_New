import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { agentsAPI } from '../services/api';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, LineChart, Line, CartesianGrid
} from 'recharts';
import {
    Users, AlertTriangle, Shield, TrendingUp, TrendingDown,
    Minus, ChevronRight, X, Phone, Mic, Activity, Brain,
    CheckCircle, Search, RefreshCw, Info, Volume2, MessageSquare
} from 'lucide-react';

// ─── Constants ─────────────────────────────────────────────────────────────
const EMOTION_COLORS = {
    neutral: '#94a3b8',
    anger: '#ef4444',
    disgust: '#8b5cf6',
    fear: '#f59e0b',
    sadness: '#3b82f6',
};
const EMOTION_ICONS = { neutral: '😐', anger: '😠', disgust: '🤢', fear: '😨', sadness: '😢' };
const TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness'];

const RISK_CONFIG = {
    low: { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200', badge: 'bg-green-500', icon: <Shield className="h-4 w-4" />, label: 'Low Risk' },
    medium: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200', badge: 'bg-yellow-400', icon: <AlertTriangle className="h-4 w-4" />, label: 'Medium Risk' },
    high: { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200', badge: 'bg-orange-500', icon: <AlertTriangle className="h-4 w-4" />, label: 'High Risk' },
    critical: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200', badge: 'bg-red-500', icon: <AlertTriangle className="h-4 w-4" />, label: 'Critical Risk' },
};

// ─── Risk Badge ────────────────────────────────────────────────────────────
const RiskBadge = ({ level }) => {
    const cfg = RISK_CONFIG[level?.toLowerCase()] || RISK_CONFIG.low;
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold text-white ${cfg.badge} ${level === 'critical' ? 'animate-pulse' : ''}`}>
            {cfg.icon} {cfg.label}
        </span>
    );
};

// ─── Circular Gauge ─────────────────────────────────────────────────────────
const Gauge = ({ pct, color, label, value }) => (
    <div className="text-center">
        <div className="relative w-14 h-14 mx-auto mb-1">
            <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="#e2e8f0" strokeWidth="3.5" />
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke={color} strokeWidth="3.5" strokeDasharray={`${pct * 100}, 100`} strokeLinecap="round" />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-gray-700">
                {Math.round(pct * 100)}%
            </span>
        </div>
        <p className="text-sm font-bold text-gray-800">{value}</p>
        <p className="text-xs text-gray-400">{label}</p>
    </div>
);

// ─── Agent Detail Slide-Over Panel ───────────────────────────────────────────
const AgentDetailPanel = ({ agent, onClose }) => {
    const [stats, setStats] = useState(null);
    const [calls, setCalls] = useState([]);
    const [risk, setRisk] = useState(null);
    const [busy, setBusy] = useState(true);
    const [tab, setTab] = useState('overview');

    useEffect(() => {
        const load = async () => {
            setBusy(true);
            try {
                const [statsRes, callsRes, riskRes] = await Promise.allSettled([
                    agentsAPI.getStats(agent.agent_id),
                    agentsAPI.getCalls(agent.agent_id, 20),
                    agentsAPI.getRisk(agent.agent_id),
                ]);
                if (statsRes.status === 'fulfilled') setStats(statsRes.value.data);
                if (callsRes.status === 'fulfilled') setCalls(callsRes.value.data);
                if (riskRes.status === 'fulfilled') setRisk(riskRes.value.data);
            } finally {
                setBusy(false);
            }
        };
        load();
    }, [agent.agent_id]);

    const radarData = TARGET_EMOTIONS.map(emo => ({
        emotion: emo.charAt(0).toUpperCase() + emo.slice(1),
        score: Math.round(((stats?.emotion_breakdown || {})[emo] || 0) * 100),
    }));

    const pieData = Object.entries(stats?.emotion_counts || {}).map(([name, value]) => ({ name, value }));

    const riskCfg = RISK_CONFIG[(risk?.risk_level || agent.risk_level || 'low').toLowerCase()] || RISK_CONFIG.low;
    const riskScore = risk?.risk_score ?? agent.risk_score ?? 0;

    const trendData = risk?.sentiment_history?.length > 0
        ? risk.sentiment_history
        : calls.slice().reverse().map((c, i) => ({ day: `#${i + 1}`, score: c.avg_sentiment || 0 }));

    return (
        <div className="fixed inset-0 z-50 flex">
            {/* Backdrop */}
            <div className="flex-1 bg-black/40 backdrop-blur-sm" onClick={onClose} />

            {/* Panel */}
            <div className="w-full max-w-2xl bg-white shadow-2xl flex flex-col overflow-hidden animate-slideInRight">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
                    <div>
                        <p className="text-xs opacity-80 uppercase tracking-wider font-medium">Agent Profile</p>
                        <h2 className="text-xl font-bold">{agent.agent_id}</h2>
                        <p className="text-xs opacity-70 mt-0.5">{stats?.total_calls ?? agent.call_count ?? 0} calls analyzed</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <RiskBadge level={risk?.risk_level || agent.risk_level || 'low'} />
                        {/* Export Buttons */}
                        <a
                            href={`http://localhost:8000/api/agents/${agent.agent_id}/export/csv`}
                            download
                            className="px-3 py-1.5 text-xs font-semibold bg-white/20 hover:bg-white/30 text-white rounded-lg transition flex items-center gap-1"
                            title="Download CSV report"
                        >
                            ⬇ CSV
                        </a>
                        <a
                            href={`http://localhost:8000/api/agents/${agent.agent_id}/export/pdf`}
                            download
                            className="px-3 py-1.5 text-xs font-semibold bg-white/20 hover:bg-white/30 text-white rounded-lg transition flex items-center gap-1"
                            title="Download PDF report"
                        >
                            ⬇ PDF
                        </a>
                        <button onClick={onClose} className="p-2 hover:bg-white/20 rounded-xl transition">
                            <X className="h-5 w-5" />
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-gray-100 bg-gray-50">
                    {[
                        { id: 'overview', label: 'Overview', icon: <Activity className="h-4 w-4" /> },
                        { id: 'emotions', label: 'Emotion Stats', icon: <Brain className="h-4 w-4" /> },
                        { id: 'risk', label: 'Risk Profile', icon: <AlertTriangle className="h-4 w-4" /> },
                        { id: 'calls', label: 'Call History', icon: <Phone className="h-4 w-4" /> },
                    ].map(t => (
                        <button key={t.id} onClick={() => setTab(t.id)}
                            className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-semibold transition-all ${tab === t.id ? 'bg-white text-indigo-600 border-b-2 border-indigo-500' : 'text-gray-500 hover:text-gray-700'
                                }`}>
                            {t.icon} {t.label}
                        </button>
                    ))}
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    {busy ? <LoadingSpinner /> : (
                        <>
                            {/** ── OVERVIEW TAB ── */}
                            {tab === 'overview' && (
                                <div className="space-y-6">
                                    {/* Risk Score Hero */}
                                    <div className={`rounded-2xl p-5 ${riskCfg.bg} ${riskCfg.border} border`}>
                                        <div className="flex items-center justify-between mb-3">
                                            <h3 className={`font-bold ${riskCfg.text}`}>Risk Assessment</h3>
                                            <RiskBadge level={risk?.risk_level || agent.risk_level || 'low'} />
                                        </div>
                                        <div className="flex items-end gap-4">
                                            <div>
                                                <span className={`text-5xl font-extrabold ${riskCfg.text}`}>{Math.round(riskScore * 100)}</span>
                                                <span className={`text-lg ${riskCfg.text} opacity-70`}>/100</span>
                                            </div>
                                            <div className="flex-1">
                                                <div className="w-full bg-white/70 rounded-full h-3">
                                                    <div className="h-3 rounded-full transition-all duration-1000" style={{ width: `${riskScore * 100}%`, background: riskCfg.badge.replace('bg-', '') === riskCfg.badge ? '#6366f1' : '' }} />
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Acoustic Stats */}
                                    <div className="bg-white border border-gray-100 rounded-2xl p-5">
                                        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                                            <Mic className="h-4 w-4 text-blue-500" /> Acoustic Averages
                                        </h3>
                                        <div className="grid grid-cols-3 gap-4">
                                            <Gauge pct={Math.min((stats?.avg_pitch || 0) / 350, 1)} color="#3b82f6"
                                                label="Avg Pitch" value={`${Math.round(stats?.avg_pitch || 0)} Hz`} />
                                            <Gauge pct={Math.min((stats?.avg_speech_rate || 0) / 200, 1)} color="#10b981"
                                                label="Speech Rate" value={`${Math.round(stats?.avg_speech_rate || 0)} WPM`} />
                                            <Gauge pct={stats?.avg_stress || 0} color={(stats?.avg_stress || 0) > 0.5 ? '#ef4444' : '#f59e0b'}
                                                label="Avg Stress" value={(stats?.avg_stress || 0).toFixed(2)} />
                                        </div>
                                    </div>

                                    {/* Trend */}
                                    {trendData.length > 0 && (
                                        <div className="bg-white border border-gray-100 rounded-2xl p-5">
                                            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                                                <TrendingUp className="h-4 w-4 text-green-500" /> Sentiment Trend
                                            </h3>
                                            <ResponsiveContainer width="100%" height={160}>
                                                <LineChart data={trendData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                                    <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                                                    <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                                                    <Tooltip />
                                                    <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1', r: 3 }} />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/** ── EMOTIONS TAB ── */}
                            {tab === 'emotions' && (
                                <div className="space-y-6">
                                    {stats ? (
                                        <>
                                            <div className="flex items-center gap-4 p-4 bg-indigo-50 rounded-2xl">
                                                <span className="text-3xl">{EMOTION_ICONS[stats.dominant_emotion] || '😐'}</span>
                                                <div>
                                                    <p className="text-xs text-indigo-500 font-bold uppercase tracking-wider">Dominant Emotion</p>
                                                    <p className="text-xl font-bold text-indigo-700 capitalize">{stats.dominant_emotion}</p>
                                                    <p className="text-xs text-indigo-400">{stats.total_calls} calls analyzed</p>
                                                </div>
                                            </div>

                                            {/* Radar */}
                                            <div className="bg-white border border-gray-100 rounded-2xl p-4">
                                                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Emotion Profile Radar</h3>
                                                <ResponsiveContainer width="100%" height={220}>
                                                    <RadarChart data={radarData}>
                                                        <PolarGrid stroke="#e2e8f0" />
                                                        <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 11, fill: '#64748b' }} />
                                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                                                        <Radar name="Frequency" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} strokeWidth={2} />
                                                    </RadarChart>
                                                </ResponsiveContainer>
                                            </div>

                                            {/* Emotion Breakdown Bars */}
                                            <div className="bg-white border border-gray-100 rounded-2xl p-4">
                                                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Emotion Distribution</h3>
                                                <div className="space-y-3">
                                                    {Object.entries(stats.emotion_counts || {}).sort((a, b) => b[1] - a[1]).map(([emo, count]) => {
                                                        const pct = (count / stats.total_calls) * 100;
                                                        return (
                                                            <div key={emo}>
                                                                <div className="flex justify-between text-sm mb-1">
                                                                    <span className="flex items-center gap-1.5 capitalize font-medium text-gray-700">
                                                                        {EMOTION_ICONS[emo]} {emo}
                                                                    </span>
                                                                    <span className="font-bold text-gray-600">{count} calls ({pct.toFixed(1)}%)</span>
                                                                </div>
                                                                <div className="w-full bg-gray-100 rounded-full h-2.5">
                                                                    <div className="h-2.5 rounded-full transition-all duration-700"
                                                                        style={{ width: `${pct}%`, background: EMOTION_COLORS[emo] || '#94a3b8' }} />
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        </>
                                    ) : (
                                        <p className="text-gray-400 italic text-center py-10">No emotion data available for this agent.</p>
                                    )}
                                </div>
                            )}

                            {/** ── RISK TAB ── */}
                            {tab === 'risk' && (
                                <div className="space-y-5">
                                    {risk ? (
                                        <>
                                            {/* Risk Factors */}
                                            <div>
                                                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Risk Factors</h3>
                                                {risk.risk_factors?.length > 0 ? (
                                                    <div className="space-y-3">
                                                        {risk.risk_factors.map((factor, i) => (
                                                            <div key={i} className="p-4 bg-red-50 border border-red-100 rounded-xl">
                                                                <div className="flex justify-between items-start mb-2">
                                                                    <div className="flex items-center gap-2">
                                                                        <AlertTriangle className="h-4 w-4 text-red-500 flex-shrink-0" />
                                                                        <p className="text-sm font-bold text-gray-900">{factor.factor}</p>
                                                                    </div>
                                                                    <span className="text-xs font-bold text-red-600 bg-red-100 px-2 py-0.5 rounded-full">
                                                                        {Math.round((factor.contribution || 0) * 100)}%
                                                                    </span>
                                                                </div>
                                                                <p className="text-xs text-gray-600 ml-6">{factor.description}</p>
                                                                <div className="mt-2 ml-6 w-full bg-red-200 rounded-full h-1.5">
                                                                    <div className="bg-red-500 h-1.5 rounded-full" style={{ width: `${(factor.contribution || 0) * 100}%` }} />
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-100 rounded-xl">
                                                        <CheckCircle className="h-5 w-5 text-green-500" />
                                                        <p className="text-sm text-green-700">No significant risk factors identified.</p>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Recommendations */}
                                            {risk.recommendations?.length > 0 && (
                                                <div>
                                                    <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Recommendations</h3>
                                                    <div className="space-y-2">
                                                        {risk.recommendations.map((rec, i) => (
                                                            <div key={i} className="flex items-start gap-3 p-3 bg-blue-50 border border-blue-100 rounded-xl">
                                                                <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                                                                <p className="text-sm text-blue-800">{rec}</p>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </>
                                    ) : (
                                        <div className="p-8 text-center text-gray-400">
                                            <Info className="h-8 w-8 mx-auto mb-3 text-gray-200" />
                                            <p className="font-medium">No risk profile found</p>
                                            <p className="text-sm mt-1">Run the marathon pipeline to generate risk scores.</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/** ── CALLS TAB ── */}
                            {tab === 'calls' && (
                                <div className="space-y-3">
                                    {calls.length > 0 ? calls.map((call) => {
                                        const emo = call.dominant_emotion || 'neutral';
                                        const color = EMOTION_COLORS[emo] || '#94a3b8';
                                        return (
                                            <Link to={`/call/${call.call_id}`} key={call.call_id}
                                                className="flex items-center gap-3 p-3 bg-gray-50 hover:bg-white hover:shadow-sm border border-transparent hover:border-gray-100 rounded-xl transition-all">
                                                <span className="text-xl">{EMOTION_ICONS[emo]}</span>
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-xs font-mono font-medium text-indigo-600 truncate">{call.call_id}</p>
                                                    <div className="flex items-center gap-3 mt-0.5">
                                                        <span className="text-xs font-bold capitalize" style={{ color }}>{emo}</span>
                                                        <span className="text-xs text-gray-400">{call.dataset}</span>
                                                    </div>
                                                </div>
                                                <div className="text-right flex-shrink-0">
                                                    <p className="text-xs text-gray-400">
                                                        {call.timestamp ? new Date(call.timestamp).toLocaleDateString() : '—'}
                                                    </p>
                                                    <ChevronRight className="h-4 w-4 text-gray-300 ml-auto mt-0.5" />
                                                </div>
                                            </Link>
                                        );
                                    }) : (
                                        <p className="text-center text-gray-400 italic py-10">No calls found for this agent.</p>
                                    )}
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

// ─── Agent Card ────────────────────────────────────────────────────────────
const AgentCard = ({ agent, onSelect }) => {
    const riskLevel = (agent.risk_level || 'low').toLowerCase();
    const riskCfg = RISK_CONFIG[riskLevel] || RISK_CONFIG.low;
    const riskScore = agent.risk_score ?? 0;

    return (
        <div
            onClick={() => onSelect(agent)}
            className={`bg-white rounded-2xl border-2 ${riskCfg.border} shadow-sm hover:shadow-lg cursor-pointer transition-all duration-200 hover:-translate-y-0.5 p-5`}
        >
            {/* Agent ID + Risk */}
            <div className="flex justify-between items-start mb-4">
                <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wider font-medium">Agent</p>
                    <p className="font-bold text-gray-900 text-sm truncate max-w-[140px]">{agent.agent_id}</p>
                </div>
                <RiskBadge level={riskLevel} />
            </div>

            {/* Risk Score Ring */}
            <div className="flex items-center gap-4 mb-4">
                <div className="relative w-16 h-16">
                    <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none" stroke="#e2e8f0" strokeWidth="3.5" />
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke={riskLevel === 'critical' ? '#ef4444' : riskLevel === 'high' ? '#f97316' : riskLevel === 'medium' ? '#eab308' : '#22c55e'}
                            strokeWidth="3.5" strokeDasharray={`${riskScore * 100}, 100`} strokeLinecap="round" />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-sm font-extrabold text-gray-800">
                        {Math.round(riskScore * 100)}
                    </span>
                </div>
                <div>
                    <p className="text-xs text-gray-400">Total Calls</p>
                    <p className="text-2xl font-bold text-gray-900">{agent.call_count || 0}</p>
                    <p className="text-xs text-gray-400 mt-0.5">Avg Sentiment: <span className="font-medium text-gray-600">{(agent.avg_sentiment || 0).toFixed(3)}</span></p>
                </div>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-400 border-t border-gray-50 pt-3">
                <span>Click for full profile</span>
                <ChevronRight className="h-4 w-4" />
            </div>
        </div>
    );
};

// ─── Main Page ───────────────────────────────────────────────────────────────
const AgentRiskPage = () => {
    const [agents, setAgents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedAgent, setSelectedAgent] = useState(null);
    const [search, setSearch] = useState('');
    const [riskFilter, setRiskFilter] = useState('');
    const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'table'

    const fetchAgents = useCallback(async () => {
        setLoading(true);
        try {
            const summary = (await agentsAPI.getAll()).data;
            // Enrich with risk scores in parallel (cap at 20 parallel requests)
            const enriched = await Promise.all(summary.map(async (agent) => {
                try {
                    const r = await agentsAPI.getRisk(agent.agent_id);
                    return { ...agent, ...r.data };
                } catch {
                    return { ...agent, risk_score: 0, risk_level: 'low' };
                }
            }));
            setAgents(enriched);
            setError(null);
        } catch {
            setError('Failed to load agents.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchAgents(); }, [fetchAgents]);

    const filtered = agents
        .filter(a => {
            const matchSearch = !search || a.agent_id.toLowerCase().includes(search.toLowerCase());
            const matchRisk = !riskFilter || (a.risk_level || 'low').toLowerCase() === riskFilter;
            return matchSearch && matchRisk;
        })
        .sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));

    // Summary stats
    const total = agents.length;
    const critical = agents.filter(a => (a.risk_level || '').toLowerCase() === 'critical').length;
    const highRisk = agents.filter(a => ['high', 'critical'].includes((a.risk_level || '').toLowerCase())).length;
    const avgScore = agents.length ? (agents.reduce((s, a) => s + (a.risk_score || 0), 0) / agents.length) : 0;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Agent Risk Assessment</h1>
                    <p className="text-gray-500 text-sm mt-1">Monitor agent performance and emotional stress levels</p>
                </div>
                <button onClick={fetchAgents}
                    className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 bg-white border border-gray-200 px-4 py-2 rounded-xl hover:bg-gray-50 transition">
                    <RefreshCw className="h-4 w-4" /> Refresh
                </button>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    { icon: <Users className="h-5 w-5 text-indigo-500" />, label: 'Total Agents', value: total, sub: 'tracked agents' },
                    { icon: <AlertTriangle className="h-5 w-5 text-red-500" />, label: 'Critical Risk', value: critical, sub: 'need immediate action' },
                    { icon: <Shield className="h-5 w-5 text-orange-500" />, label: 'High Risk', value: highRisk, sub: 'require monitoring' },
                    { icon: <Activity className="h-5 w-5 text-purple-500" />, label: 'Avg Risk Score', value: `${Math.round(avgScore * 100)}`, sub: 'across all agents' },
                ].map(s => (
                    <div key={s.label} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 flex items-center gap-3">
                        <div className="p-2 bg-gray-50 rounded-xl">{s.icon}</div>
                        <div>
                            <p className="text-xs text-gray-400 uppercase tracking-wide font-medium">{s.label}</p>
                            <p className="text-xl font-bold text-gray-800">{s.value}</p>
                            <p className="text-xs text-gray-400">{s.sub}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Filter Bar */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 flex flex-wrap items-center gap-3">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input type="text" placeholder="Search Agent ID..." value={search} onChange={e => setSearch(e.target.value)}
                        className="border border-gray-200 rounded-xl pl-9 pr-4 py-2 text-sm focus:ring-2 focus:ring-indigo-300 outline-none w-52" />
                </div>

                <select value={riskFilter} onChange={e => setRiskFilter(e.target.value)}
                    className="border border-gray-200 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-300 outline-none bg-white">
                    <option value="">All Risk Levels</option>
                    <option value="critical">🔴 Critical</option>
                    <option value="high">🟠 High</option>
                    <option value="medium">🟡 Medium</option>
                    <option value="low">🟢 Low</option>
                </select>

                {(search || riskFilter) && (
                    <button onClick={() => { setSearch(''); setRiskFilter(''); }}
                        className="text-xs text-indigo-600 hover:text-indigo-800 font-medium underline">Clear</button>
                )}

                <div className="ml-auto flex items-center gap-1 bg-gray-100 rounded-xl p-1">
                    {['grid', 'table'].map(mode => (
                        <button key={mode} onClick={() => setViewMode(mode)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${viewMode === mode ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}>
                            {mode === 'grid' ? '⊞ Grid' : '≡ Table'}
                        </button>
                    ))}
                </div>

                <span className="text-xs text-gray-400">{filtered.length} agents</span>
            </div>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            {loading ? <LoadingSpinner /> : (
                viewMode === 'grid' ? (
                    /* Grid View */
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {filtered.map(agent => (
                            <AgentCard key={agent.agent_id} agent={agent} onSelect={setSelectedAgent} />
                        ))}
                        {filtered.length === 0 && (
                            <div className="col-span-full p-16 text-center text-gray-400">
                                <Users className="h-10 w-10 mx-auto mb-3 text-gray-200" />
                                <p className="font-medium">No agents match your filters</p>
                            </div>
                        )}
                    </div>
                ) : (
                    /* Table View */
                    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
                        <table className="min-w-full divide-y divide-gray-100">
                            <thead className="bg-gray-50">
                                <tr>
                                    {['Agent ID', 'Calls', 'Avg Sentiment', 'Risk Score', 'Risk Level', ''].map(h => (
                                        <th key={h} className="px-5 py-3.5 text-left text-xs font-bold text-gray-400 uppercase tracking-wider">{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-50">
                                {filtered.map(agent => {
                                    const lvl = (agent.risk_level || 'low').toLowerCase();
                                    const cfg = RISK_CONFIG[lvl] || RISK_CONFIG.low;
                                    return (
                                        <tr key={agent.agent_id} onClick={() => setSelectedAgent(agent)}
                                            className="hover:bg-indigo-50/30 cursor-pointer transition-colors">
                                            <td className="px-5 py-3.5 text-sm font-mono font-medium text-gray-900">{agent.agent_id}</td>
                                            <td className="px-5 py-3.5 text-sm text-gray-600">{agent.call_count || 0}</td>
                                            <td className="px-5 py-3.5 text-sm">
                                                <span className={(agent.avg_sentiment || 0) >= 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                                                    {(agent.avg_sentiment || 0).toFixed(3)}
                                                </span>
                                            </td>
                                            <td className="px-5 py-3.5">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-24 bg-gray-100 rounded-full h-2">
                                                        <div className="h-2 rounded-full" style={{ width: `${(agent.risk_score || 0) * 100}%`, background: lvl === 'critical' ? '#ef4444' : lvl === 'high' ? '#f97316' : lvl === 'medium' ? '#eab308' : '#22c55e' }} />
                                                    </div>
                                                    <span className="text-sm font-bold text-gray-700">{Math.round((agent.risk_score || 0) * 100)}</span>
                                                </div>
                                            </td>
                                            <td className="px-5 py-3.5"><RiskBadge level={lvl} /></td>
                                            <td className="px-5 py-3.5 text-right">
                                                <span className="inline-flex items-center gap-1 text-xs font-bold text-indigo-600 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-lg transition">
                                                    Profile <ChevronRight className="h-3 w-3" />
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                        {filtered.length === 0 && (
                            <div className="p-12 text-center text-gray-400">
                                <MessageSquare className="h-8 w-8 mx-auto mb-2 text-gray-200" />
                                <p>No agents match your filters</p>
                            </div>
                        )}
                    </div>
                )
            )}

            {/* Agent Detail Panel */}
            {selectedAgent && (
                <AgentDetailPanel agent={selectedAgent} onClose={() => setSelectedAgent(null)} />
            )}
        </div>
    );
};

export default AgentRiskPage;
