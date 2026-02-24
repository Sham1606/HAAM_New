import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { callsAPI } from '../services/api';
import UploadModal from '../components/CallsList/UploadModal';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import {
    Plus, Filter, Search, Phone, TrendingUp, AlertTriangle,
    ChevronRight, Mic, MessageSquare, RefreshCw
} from 'lucide-react';

const EMOTION_COLORS = {
    neutral: { bg: 'bg-slate-100', text: 'text-slate-700', dot: '#94a3b8' },
    anger: { bg: 'bg-red-100', text: 'text-red-700', dot: '#ef4444' },
    disgust: { bg: 'bg-purple-100', text: 'text-purple-700', dot: '#8b5cf6' },
    fear: { bg: 'bg-amber-100', text: 'text-amber-700', dot: '#f59e0b' },
    sadness: { bg: 'bg-blue-100', text: 'text-blue-700', dot: '#3b82f6' },
};
const EMOTION_ICONS = { neutral: '😐', anger: '😠', disgust: '🤢', fear: '😨', sadness: '😢' };

const EmotionBadge = ({ emotion }) => {
    const e = emotion?.toLowerCase() || 'neutral';
    const c = EMOTION_COLORS[e] || EMOTION_COLORS.neutral;
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${c.bg} ${c.text}`}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: c.dot }} />
            {EMOTION_ICONS[e]} {e}
        </span>
    );
};

const StressIndicator = ({ score }) => {
    const s = score || 0;
    const color = s > 0.6 ? 'text-red-600' : s > 0.4 ? 'text-amber-600' : 'text-green-600';
    const label = s > 0.6 ? 'High' : s > 0.4 ? 'Med' : 'Low';
    return (
        <div className={`flex items-center gap-1 text-xs font-bold ${color}`}>
            <AlertTriangle className="h-3 w-3" />
            {label}
        </div>
    );
};

const CallsListPage = () => {
    const [calls, setCalls] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isUploadOpen, setIsUploadOpen] = useState(false);
    const [emotionFilter, setEmotionFilter] = useState('');
    const [searchTerm, setSearchTerm] = useState('');

    const fetchCalls = async () => {
        setLoading(true);
        try {
            const response = await callsAPI.getAll({ limit: 100 });
            setCalls(response.data);
            setError(null);
        } catch (err) {
            setError('Failed to load calls.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchCalls(); }, []);

    const filtered = calls.filter(c => {
        const matchEmotion = !emotionFilter || (c.dominant_emotion || '').toLowerCase() === emotionFilter;
        const matchSearch = !searchTerm || c.call_id.toLowerCase().includes(searchTerm.toLowerCase()) || (c.agent_id || '').toLowerCase().includes(searchTerm.toLowerCase());
        return matchEmotion && matchSearch;
    });

    // Stats
    const total = calls.length;
    const highStress = calls.filter(c => (c.agent_stress_score || c.overall_metrics?.agent_stress_score || 0) > 0.6).length;
    const emotionDist = calls.reduce((acc, c) => { const e = c.dominant_emotion || 'neutral'; acc[e] = (acc[e] || 0) + 1; return acc; }, {});
    const topEmotion = Object.entries(emotionDist).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Call Analysis Center</h1>
                    <p className="text-gray-500 text-sm mt-1">Review, analyze, and explore emotion-classified calls</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={fetchCalls} className="p-2.5 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition text-gray-500">
                        <RefreshCw className="h-4 w-4" />
                    </button>
                    <button
                        onClick={() => setIsUploadOpen(true)}
                        className="flex items-center gap-2 bg-indigo-600 text-white px-5 py-2.5 rounded-xl hover:bg-indigo-700 font-medium transition shadow-sm shadow-indigo-200"
                    >
                        <Plus className="h-4 w-4" /> Upload Call
                    </button>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    { icon: <Phone className="h-5 w-5 text-indigo-500" />, label: 'Total Calls', value: total, sub: 'in database' },
                    { icon: <AlertTriangle className="h-5 w-5 text-red-500" />, label: 'High Stress', value: highStress, sub: 'calls flagged' },
                    { icon: <TrendingUp className="h-5 w-5 text-green-500" />, label: 'Dominant', value: `${EMOTION_ICONS[topEmotion] || ''} ${topEmotion}`, sub: 'top emotion' },
                    { icon: <Mic className="h-5 w-5 text-blue-500" />, label: 'Filtered', value: filtered.length, sub: 'matching calls' },
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

            {/* Filters */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 flex flex-wrap items-center gap-3">
                <div className="flex items-center text-gray-400">
                    <Filter className="h-4 w-4 mr-1.5" />
                    <span className="text-sm font-medium">Filter:</span>
                </div>

                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search Call ID or Agent..."
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        className="border border-gray-200 rounded-xl pl-9 pr-4 py-2 text-sm focus:ring-2 focus:ring-indigo-300 outline-none w-56"
                    />
                </div>

                <select value={emotionFilter} onChange={e => setEmotionFilter(e.target.value)}
                    className="border border-gray-200 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-300 outline-none bg-white">
                    <option value="">All Emotions</option>
                    {['neutral', 'anger', 'disgust', 'fear', 'sadness'].map(e => (
                        <option key={e} value={e}>{EMOTION_ICONS[e]} {e}</option>
                    ))}
                </select>

                {(searchTerm || emotionFilter) && (
                    <button onClick={() => { setSearchTerm(''); setEmotionFilter(''); }}
                        className="text-xs text-indigo-600 hover:text-indigo-800 font-medium underline">
                        Clear
                    </button>
                )}
            </div>

            {/* Calls Table */}
            {loading ? <LoadingSpinner /> : (
                <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-100">
                        <thead className="bg-gray-50">
                            <tr>
                                {['Call ID', 'Agent', 'Emotion', 'Stress', 'Sentiment', 'Pitch (Hz)', 'Date', ''].map(h => (
                                    <th key={h} className="px-5 py-3.5 text-left text-xs font-bold text-gray-400 uppercase tracking-wider">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50">
                            {filtered.map(call => {
                                const stress = call.agent_stress_score || call.overall_metrics?.agent_stress_score || 0;
                                const pitch = call.avg_pitch || call.overall_metrics?.avg_pitch || call.acoustic_features?.pitch_mean || 0;
                                const sentiment = call.avg_sentiment || 0;
                                const isIemocap = call.call_id.startsWith('iemocap');

                                return (
                                    <tr key={call.call_id} className="hover:bg-indigo-50/30 transition-colors">
                                        <td className="px-5 py-3.5">
                                            <Link to={`/call/${call.call_id}`} className="text-sm font-mono font-medium text-indigo-600 hover:text-indigo-800 hover:underline max-w-[200px] truncate block">
                                                {call.call_id}
                                            </Link>
                                        </td>
                                        <td className="px-5 py-3.5 text-sm text-gray-600">{call.agent_id || '—'}</td>
                                        <td className="px-5 py-3.5">
                                            <EmotionBadge emotion={call.dominant_emotion} />
                                        </td>
                                        <td className="px-5 py-3.5">
                                            <StressIndicator score={stress} />
                                        </td>
                                        <td className="px-5 py-3.5 text-sm text-gray-600">
                                            <span className={sentiment >= 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                                                {sentiment.toFixed(2)}
                                            </span>
                                        </td>
                                        <td className="px-5 py-3.5 text-sm text-gray-600">
                                            {Math.round(pitch)} Hz
                                        </td>
                                        <td className="px-5 py-3.5 text-xs text-gray-400">
                                            {call.timestamp ? new Date(call.timestamp).toLocaleDateString() : '—'}
                                        </td>
                                        <td className="px-5 py-3.5 text-right">
                                            <Link to={`/call/${call.call_id}`}
                                                className="inline-flex items-center gap-1 text-xs font-bold text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-lg transition">
                                                Analyse <ChevronRight className="h-3 w-3" />
                                            </Link>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>

                    {filtered.length === 0 && (
                        <div className="p-16 text-center">
                            <MessageSquare className="h-10 w-10 text-gray-200 mx-auto mb-3" />
                            <p className="text-gray-400 font-medium">No calls match your filters</p>
                            <p className="text-gray-300 text-sm mt-1">Try clearing the filters or uploading a new call</p>
                        </div>
                    )}
                </div>
            )}

            {isUploadOpen && (
                <UploadModal onClose={() => setIsUploadOpen(false)} onSuccess={() => { fetchCalls(); setIsUploadOpen(false); }} />
            )}
            {error && <ErrorToast message={error} onClose={() => setError(null)} />}
        </div>
    );
};

export default CallsListPage;
