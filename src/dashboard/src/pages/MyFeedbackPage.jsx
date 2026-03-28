/**
 * MyFeedbackPage: Shows AI-generated coaching feedback based on the agent's
 * recent emotional patterns and stress data.
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../services/AuthContext';
import { MessageSquare, Activity, TrendingUp, TrendingDown, Minus, Brain, Heart } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const EMOTION_CONFIG = {
    neutral: { icon: '😐', color: '#94a3b8' },
    anger: { icon: '😠', color: '#ef4444' },
    disgust: { icon: '🤢', color: '#8b5cf6' },
    fear: { icon: '😨', color: '#f59e0b' },
    sadness: { icon: '😢', color: '#3b82f6' },
};

const MyFeedbackPage = () => {
    const { user } = useAuth();
    const [calls, setCalls] = useState([]);
    const [feedbackList, setFeedbackList] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!user?.id) return;

        axios.get(`${API_BASE}/api/agents/${user.id}/calls`, { params: { limit: 20 } })
            .then(async (res) => {
                const callData = res.data || [];
                setCalls(callData);

                // Generate feedback for each recent call
                const fbPromises = callData.slice(0, 10).map(async (call) => {
                    try {
                        const fbRes = await axios.post(`${API_BASE}/api/feedback/predict`, {
                            emotion: call.dominant_emotion || 'neutral',
                            stress_score: call.agent_stress_score || 0,
                            confidence: 0.7,
                        });
                        return {
                            call_id: call.call_id,
                            emotion: call.dominant_emotion,
                            stress: call.agent_stress_score,
                            timestamp: call.timestamp,
                            feedback: fbRes.data.feedback,
                        };
                    } catch {
                        return null;
                    }
                });

                const results = (await Promise.all(fbPromises)).filter(Boolean);
                setFeedbackList(results);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, [user]);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Activity className="h-8 w-8 text-indigo-500 animate-spin" />
            </div>
        );
    }

    // Overall stats
    const avgStress = calls.length > 0
        ? calls.reduce((s, c) => s + (c.agent_stress_score || 0), 0) / calls.length : 0;

    const emotionCounts = {};
    calls.forEach(c => {
        const emo = c.dominant_emotion || 'neutral';
        emotionCounts[emo] = (emotionCounts[emo] || 0) + 1;
    });
    const dominantEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'neutral';
    const dominantCfg = EMOTION_CONFIG[dominantEmotion] || EMOTION_CONFIG.neutral;

    return (
        <div className="space-y-6 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-2xl p-6 text-white shadow-lg">
                <h1 className="text-2xl font-bold flex items-center gap-3">
                    <MessageSquare className="h-7 w-7" /> My Feedback
                </h1>
                <p className="text-emerald-200 mt-1 text-sm">
                    AI-powered coaching based on your emotional patterns
                </p>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm text-center">
                    <Heart className="h-8 w-8 text-rose-400 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-gray-800">{Math.round(avgStress * 100)}%</p>
                    <p className="text-xs text-gray-400">Average Stress</p>
                </div>
                <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm text-center">
                    <span className="text-4xl block mb-1">{dominantCfg.icon}</span>
                    <p className="text-lg font-bold text-gray-800 capitalize">{dominantEmotion}</p>
                    <p className="text-xs text-gray-400">Most Common Emotion</p>
                </div>
                <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm text-center">
                    <Brain className="h-8 w-8 text-indigo-400 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-gray-800">{calls.length}</p>
                    <p className="text-xs text-gray-400">Calls Analyzed</p>
                </div>
            </div>

            {/* Feedback Cards */}
            {feedbackList.length === 0 ? (
                <div className="text-center py-16 text-gray-400">
                    <MessageSquare className="h-16 w-16 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-semibold">No feedback yet</p>
                    <p className="text-sm mt-1">Process some calls to receive AI coaching</p>
                </div>
            ) : (
                <div className="space-y-3">
                    <h3 className="text-sm font-bold text-gray-600 uppercase tracking-wider">Recent Coaching</h3>
                    {feedbackList.map((fb, i) => {
                        const cfg = EMOTION_CONFIG[fb.emotion] || EMOTION_CONFIG.neutral;
                        return (
                            <div key={fb.call_id}
                                className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm hover:shadow-md transition">
                                <div className="flex items-start gap-4">
                                    <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center text-2xl"
                                        style={{ background: cfg.color + '15' }}>
                                        {cfg.icon}
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-xs font-semibold text-gray-500">{fb.call_id}</span>
                                            <span className="text-xs text-gray-300">·</span>
                                            <span className="text-xs text-gray-400">{fb.timestamp?.split('T')[0]}</span>
                                        </div>
                                        <p className="text-sm text-gray-700 leading-relaxed">{fb.feedback}</p>
                                        <div className="flex items-center gap-3 mt-2">
                                            <span className={`text-xs font-medium capitalize px-2 py-0.5 rounded-full`}
                                                style={{ background: cfg.color + '15', color: cfg.color }}>
                                                {fb.emotion}
                                            </span>
                                            <span className="text-xs text-gray-400">
                                                Stress: {Math.round((fb.stress || 0) * 100)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default MyFeedbackPage;
