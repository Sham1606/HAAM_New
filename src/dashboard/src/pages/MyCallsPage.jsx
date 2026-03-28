/**
 * MyCallsPage: Shows the agent's own processed calls.
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../services/AuthContext';
import { Link } from 'react-router-dom';
import { Phone, ArrowRight, Activity } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const EMOTION_CONFIG = {
    neutral: { icon: '😐', bg: 'bg-slate-50', text: 'text-slate-700' },
    anger: { icon: '😠', bg: 'bg-red-50', text: 'text-red-700' },
    disgust: { icon: '🤢', bg: 'bg-purple-50', text: 'text-purple-700' },
    fear: { icon: '😨', bg: 'bg-amber-50', text: 'text-amber-700' },
    sadness: { icon: '😢', bg: 'bg-blue-50', text: 'text-blue-700' },
};

const MyCallsPage = () => {
    const { user } = useAuth();
    const [calls, setCalls] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!user?.id) return;
        axios.get(`${API_BASE}/api/agents/${user.id}/calls`, { params: { limit: 100 } })
            .then(res => { setCalls(res.data); setLoading(false); })
            .catch(() => setLoading(false));
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
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
                <h1 className="text-2xl font-bold flex items-center gap-3">
                    <Phone className="h-7 w-7" /> My Calls
                </h1>
                <p className="text-blue-200 mt-1 text-sm">
                    {calls.length} processed calls for {user?.display_name || user?.username}
                </p>
            </div>

            {calls.length === 0 ? (
                <div className="text-center py-20 text-gray-400">
                    <Phone className="h-16 w-16 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-semibold">No calls yet</p>
                    <p className="text-sm mt-1">Start a live analysis to process your first call</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {calls.map(call => {
                        const cfg = EMOTION_CONFIG[call.dominant_emotion] || EMOTION_CONFIG.neutral;
                        return (
                            <Link to={`/call/${call.call_id}`} key={call.call_id}
                                className={`flex items-center justify-between p-4 rounded-xl border ${cfg.bg} border-transparent hover:shadow-md transition group`}>
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{cfg.icon}</span>
                                    <div>
                                        <p className="text-sm font-semibold text-gray-800">{call.call_id}</p>
                                        <p className="text-xs text-gray-400">{call.timestamp?.replace('T', ' ').slice(0, 19)}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="text-right">
                                        <p className={`text-xs font-bold capitalize ${cfg.text}`}>{call.dominant_emotion}</p>
                                        <p className="text-xs text-gray-400">Stress: {Math.round((call.agent_stress_score || 0) * 100)}%</p>
                                    </div>
                                    <ArrowRight className="h-4 w-4 text-gray-300 group-hover:text-indigo-500 group-hover:translate-x-1 transition" />
                                </div>
                            </Link>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default MyCallsPage;
