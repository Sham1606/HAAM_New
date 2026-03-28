/**
 * AgentDashboard: Personal dashboard for agents.
 * Shows their own stats, recent calls, emotional summary, and quick access to live analysis.
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../services/AuthContext';
import { Link } from 'react-router-dom';
import {
    Mic, Activity, TrendingUp, TrendingDown, Minus,
    Phone, AlertTriangle, Brain, BarChart3, ArrowRight
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const EMOTION_CONFIG = {
    neutral:  { icon: '😐', color: '#94a3b8', bg: 'bg-slate-50',   text: 'text-slate-700'  },
    anger:    { icon: '😠', color: '#ef4444', bg: 'bg-red-50',     text: 'text-red-700'    },
    disgust:  { icon: '🤢', color: '#8b5cf6', bg: 'bg-purple-50',  text: 'text-purple-700' },
    fear:     { icon: '😨', color: '#f59e0b', bg: 'bg-amber-50',   text: 'text-amber-700'  },
    sadness:  { icon: '😢', color: '#3b82f6', bg: 'bg-blue-50',    text: 'text-blue-700'   },
};

const AgentDashboardPage = () => {
    const { user } = useAuth();
    const [stats, setStats] = useState(null);
    const [recentCalls, setRecentCalls] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!user?.id) return;
        setLoading(true);

        Promise.all([
            axios.get(`${API_BASE}/api/agents/${user.id}/stats`).catch(() => null),
            axios.get(`${API_BASE}/api/agents/${user.id}/calls`, { params: { limit: 5 } }).catch(() => null),
        ]).then(([statsRes, callsRes]) => {
            if (statsRes) setStats(statsRes.data);
            if (callsRes) setRecentCalls(callsRes.data);
            setLoading(false);
        });
    }, [user]);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Activity className="h-8 w-8 text-indigo-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6 pb-10">
            {/* Welcome Header */}
            <div className="bg-gradient-to-r from-indigo-600 to-blue-600 rounded-2xl p-6 text-white shadow-lg">
                <h1 className="text-2xl font-bold">Welcome back, {user?.display_name || user?.username} 👋</h1>
                <p className="text-indigo-200 mt-1 text-sm">Here's your performance overview</p>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    { label: 'Total Calls', value: stats?.total_calls || 0, icon: <Phone className="h-5 w-5" />, color: 'text-indigo-600 bg-indigo-50' },
                    { label: 'Avg Stress', value: `${Math.round((stats?.avg_stress || 0) * 100)}%`, icon: <AlertTriangle className="h-5 w-5" />, color: (stats?.avg_stress || 0) > 0.5 ? 'text-red-600 bg-red-50' : 'text-green-600 bg-green-50' },
                    { label: 'Dominant Emotion', value: stats?.dominant_emotion || 'neutral', icon: <Brain className="h-5 w-5" />, color: 'text-purple-600 bg-purple-50' },
                    { label: 'Avg Sentiment', value: (stats?.avg_sentiment || 0).toFixed(2), icon: <BarChart3 className="h-5 w-5" />, color: 'text-blue-600 bg-blue-50' },
                ].map(s => (
                    <div key={s.label} className="bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${s.color}`}>{s.icon}</div>
                        <p className="text-2xl font-bold text-gray-800 capitalize">{s.value}</p>
                        <p className="text-xs text-gray-400 mt-1">{s.label}</p>
                    </div>
                ))}
            </div>

            {/* Emotion Breakdown */}
            {stats?.emotion_breakdown && (
                <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider mb-4">Your Emotion Profile</h3>
                    <div className="space-y-2">
                        {Object.entries(stats.emotion_breakdown)
                            .sort((a, b) => b[1] - a[1])
                            .map(([emo, pct]) => {
                                const cfg = EMOTION_CONFIG[emo] || EMOTION_CONFIG.neutral;
                                return (
                                    <div key={emo} className="flex items-center gap-3">
                                        <span className="w-6 text-center">{cfg.icon}</span>
                                        <span className="text-xs text-gray-600 w-16 capitalize">{emo}</span>
                                        <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                                            <div className="h-3 rounded-full transition-all duration-500"
                                                style={{ width: `${Math.round(pct * 100)}%`, background: cfg.color }} />
                                        </div>
                                        <span className="text-xs font-semibold text-gray-500 w-10 text-right">{Math.round(pct * 100)}%</span>
                                    </div>
                                );
                            })}
                    </div>
                </div>
            )}

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link to="/live"
                    className="flex items-center justify-between bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-2xl p-5 shadow-lg hover:shadow-xl transition group">
                    <div className="flex items-center gap-3">
                        <Mic className="h-6 w-6" />
                        <div>
                            <p className="font-bold">Start Live Analysis</p>
                            <p className="text-indigo-200 text-xs">Begin real-time emotion detection</p>
                        </div>
                    </div>
                    <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition" />
                </Link>
                <Link to="/my-calls"
                    className="flex items-center justify-between bg-white border border-gray-200 text-gray-800 rounded-2xl p-5 shadow-sm hover:shadow-md transition group">
                    <div className="flex items-center gap-3">
                        <Phone className="h-6 w-6 text-indigo-500" />
                        <div>
                            <p className="font-bold">View My Calls</p>
                            <p className="text-gray-400 text-xs">See all your processed calls</p>
                        </div>
                    </div>
                    <ArrowRight className="h-5 w-5 text-gray-400 group-hover:translate-x-1 transition" />
                </Link>
            </div>

            {/* Recent Calls */}
            {recentCalls.length > 0 && (
                <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider mb-4">Recent Calls</h3>
                    <div className="space-y-2">
                        {recentCalls.map(call => {
                            const cfg = EMOTION_CONFIG[call.dominant_emotion] || EMOTION_CONFIG.neutral;
                            return (
                                <div key={call.call_id} className={`flex items-center justify-between p-3 rounded-xl ${cfg.bg}`}>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xl">{cfg.icon}</span>
                                        <div>
                                            <p className="text-sm font-medium text-gray-800">{call.call_id}</p>
                                            <p className="text-xs text-gray-400">{call.timestamp?.split('T')[0]}</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className={`text-xs font-semibold capitalize ${cfg.text}`}>{call.dominant_emotion}</p>
                                        <p className="text-xs text-gray-400">Stress: {Math.round((call.agent_stress_score || 0) * 100)}%</p>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AgentDashboardPage;
