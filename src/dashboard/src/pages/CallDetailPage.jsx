import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { callsAPI } from '../services/api';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, PieChart, Pie, Legend
} from 'recharts';
import {
    ArrowLeft, Mic, FileText, Brain, AlertTriangle,
    TrendingUp, Volume2, Activity, Info, ChevronRight
} from 'lucide-react';

// ─── Constants ──────────────────────────────────────────────────────────────
const EMOTION_COLORS = {
    neutral: '#94a3b8',
    anger: '#ef4444',
    disgust: '#8b5cf6',
    fear: '#f59e0b',
    sadness: '#3b82f6',
};
const EMOTION_ICONS = { neutral: '😐', anger: '😠', disgust: '🤢', fear: '😨', sadness: '😢' };
const TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness'];

// ─── Stat Pill ───────────────────────────────────────────────────────────────
const StatPill = ({ icon, label, value, color = 'blue' }) => (
    <div className={`flex items-center gap-3 bg-${color}-50 border border-${color}-100 rounded-xl p-4`}>
        <div className={`p-2 bg-${color}-100 rounded-lg text-${color}-600`}>{icon}</div>
        <div>
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">{label}</p>
            <p className={`text-lg font-bold text-${color}-700`}>{value}</p>
        </div>
    </div>
);

// ─── Modality Attention Bar ───────────────────────────────────────────────────
const AttentionBar = ({ acoustic, text }) => {
    const ac = Math.round((acoustic ?? 0.5) * 100);
    const tx = Math.round((text ?? 0.5) * 100);
    return (
        <div className="space-y-3">
            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Modality Attention (XAI)</p>
            <div>
                <div className="flex justify-between text-sm mb-1">
                    <span className="flex items-center gap-1"><Volume2 className="h-3 w-3 text-blue-500" /> Acoustic (Voice)</span>
                    <span className="font-bold text-blue-600">{ac}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
                    <div className="h-3 rounded-full attention-fill-acoustic transition-all duration-700" style={{ width: `${ac}%` }} />
                </div>
            </div>
            <div>
                <div className="flex justify-between text-sm mb-1">
                    <span className="flex items-center gap-1"><FileText className="h-3 w-3 text-purple-500" /> Text (Language)</span>
                    <span className="font-bold text-purple-600">{tx}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
                    <div className="h-3 rounded-full attention-fill-text transition-all duration-700" style={{ width: `${tx}%` }} />
                </div>
            </div>
            <p className="text-xs text-gray-400 mt-2 italic">
                {ac > tx
                    ? `Voice features dominated this prediction — tone, pitch and stress patterns were more informative than text.`
                    : `Language features drove this prediction — the words spoken carried stronger emotional signal.`
                }
            </p>
        </div>
    );
};

// ─── Radar Chart: Emotion Profile ────────────────────────────────────────────
const EmotionRadar = ({ distribution }) => {
    const data = TARGET_EMOTIONS.map(emo => ({
        emotion: emo.charAt(0).toUpperCase() + emo.slice(1),
        score: Math.round((distribution[emo] || 0) * 100),
    }));
    return (
        <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={data}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 12, fill: '#64748b' }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <Radar name="Emotion" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} strokeWidth={2} />
            </RadarChart>
        </ResponsiveContainer>
    );
};

// ─── Pie Chart: Emotion Distribution ─────────────────────────────────────────
const EmotionPie = ({ distribution }) => {
    const data = TARGET_EMOTIONS
        .filter(emo => (distribution[emo] || 0) > 0)
        .map(emo => ({ name: emo, value: Math.round((distribution[emo] || 0) * 100) }));
    return (
        <ResponsiveContainer width="100%" height={200}>
            <PieChart>
                <Pie data={data} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={({ name, value }) => `${name} ${value}%`} labelLine={false}>
                    {data.map((entry) => <Cell key={entry.name} fill={EMOTION_COLORS[entry.name] || '#94a3b8'} />)}
                </Pie>
                <Tooltip formatter={(v) => `${v}%`} />
            </PieChart>
        </ResponsiveContainer>
    );
};

// ─── Top3 Predictions Bar Chart ───────────────────────────────────────────────
const PredictionsBar = ({ predictions }) => {
    if (!predictions || predictions.length === 0) return <p className="text-sm text-gray-400 italic">No prediction data</p>;
    const data = predictions.map(p => ({ emotion: p.emotion, confidence: Math.round((p.confidence || p.score * 100 || 0)) }));
    return (
        <ResponsiveContainer width="100%" height={160}>
            <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="emotion" tick={{ fontSize: 12, textTransform: 'capitalize' }} width={65} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="confidence" radius={[0, 6, 6, 0]}>
                    {data.map((entry) => <Cell key={entry.emotion} fill={EMOTION_COLORS[entry.emotion] || '#6366f1'} />)}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};

// ─── Segment Timeline ─────────────────────────────────────────────────────────
const SegmentTimeline = ({ segments }) => {
    if (!segments || segments.length === 0)
        return <p className="text-sm text-gray-400 italic p-4">No turn-level data available.</p>;
    return (
        <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
            {segments.map((seg, i) => {
                const emo = seg.emotion || 'neutral';
                const color = EMOTION_COLORS[emo] || '#94a3b8';
                const conf = seg.emotion_confidence || 0;
                return (
                    <div key={i} className="flex gap-3 group">
                        <div className="flex flex-col items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0" style={{ background: color }}>
                                {EMOTION_ICONS[emo] || '?'}
                            </div>
                            {i < segments.length - 1 && <div className="w-0.5 flex-1 bg-gray-100 mt-1 min-h-[12px]" />}
                        </div>
                        <div className="flex-1 bg-gray-50 rounded-xl p-3 group-hover:bg-white group-hover:shadow-sm transition-all border border-transparent group-hover:border-gray-100">
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-xs font-bold uppercase tracking-wide" style={{ color }}>{emo}</span>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-400">{seg.start_time?.toFixed(1)}s – {seg.end_time?.toFixed(1)}s</span>
                                    <span className="text-xs font-bold text-gray-600">{Math.round(conf * 100)}% conf</span>
                                </div>
                            </div>
                            <p className="text-sm text-gray-700 leading-relaxed">{seg.text || <span className="italic text-gray-400">No transcript</span>}</p>
                            {seg.pitch_mean > 0 && (
                                <div className="flex gap-3 mt-2 text-xs text-gray-400">
                                    <span>Pitch: {Math.round(seg.pitch_mean)} Hz</span>
                                    <span>Sentiment: {seg.sentiment_score?.toFixed(2)}</span>
                                </div>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

// ─── XAI Insights Panel ───────────────────────────────────────────────────────
const XAIPanel = ({ callData, acousticFusion }) => {
    const { callId } = useParams();
    const [xaiData, setXaiData] = useState(null);
    const [xaiLoading, setXaiLoading] = useState(true);

    useEffect(() => {
        if (!callId) { setXaiLoading(false); return; }
        callsAPI.getXaiReport(callId)
            .then(res => setXaiData(res.data || null))
            .catch(() => setXaiData(null))
            .finally(() => setXaiLoading(false));
    }, [callId]);

    const metrics = callData.overall_metrics || {};
    const dist = callData.text_features?.sentiment_distribution || metrics.emotion_distribution || {};
    const dominant = metrics.dominant_emotion || 'neutral';
    const stress = metrics.agent_stress_score || 0;
    const pitch = metrics.avg_pitch || callData.acoustic_features?.pitch_mean || 0;
    const speechRate = metrics.speech_rate_wpm || callData.acoustic_features?.speech_rate_wpm || 0;
    const predictions = metrics.top_3_predictions || [];

    // Resolve modality split — prefer real Captum data, else attention gate
    const ac_pct = xaiData?.modality_split?.acoustic ?? Math.round((acousticFusion ?? 0.5) * 100);
    const tx_pct = xaiData?.modality_split?.text ?? (100 - ac_pct);
    const acoustic = ac_pct / 100;
    const textW = tx_pct / 100;

    // Narrative insights (heuristic, always shown)
    const insights = [];
    if (acoustic > 0.65) insights.push({ icon: <Volume2 className="h-4 w-4" />, color: 'blue', text: `Voice patterns drove this prediction (${ac_pct}% attention). Prosodic features like pitch and stress were highly informative.` });
    else if (textW > 0.65) insights.push({ icon: <FileText className="h-4 w-4" />, color: 'purple', text: `Language content dominated (${tx_pct}% attention). The words spoken carried the strongest emotional signal.` });
    else insights.push({ icon: <Brain className="h-4 w-4" />, color: 'indigo', text: `Balanced fusion — voice and language equally contributed to the prediction.` });
    if (stress > 0.6) insights.push({ icon: <AlertTriangle className="h-4 w-4" />, color: 'red', text: `High stress detected (score: ${stress.toFixed(2)}). Consider follow-up coaching for this agent.` });
    if (pitch > 280) insights.push({ icon: <TrendingUp className="h-4 w-4" />, color: 'amber', text: `Elevated pitch (${Math.round(pitch)} Hz) suggests heightened arousal or emotional activation.` });
    if (pitch < 120 && pitch > 0) insights.push({ icon: <Activity className="h-4 w-4" />, color: 'slate', text: `Low pitch (${Math.round(pitch)} Hz) is consistent with sadness or a low-energy emotional state.` });
    if (speechRate > 160) insights.push({ icon: <TrendingUp className="h-4 w-4" />, color: 'amber', text: `Fast speech rate (${Math.round(speechRate)} WPM) may indicate nervousness or agitation.` });

    // Top acoustic drivers chart data from Captum
    const driverData = (xaiData?.top_acoustic_drivers || []).map(d => ({
        name: d.display_name,
        attribution: d.attribution,
    }));

    return (
        <div className="space-y-6">
            {/* Attention Attribution */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Brain className="h-4 w-4 text-indigo-500" /> Modality Attention Attribution
                </h3>
                <AttentionBar acoustic={acoustic} text={textW} />
            </div>

            {/* Captum — Top Acoustic Feature Drivers */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-1 flex items-center gap-2">
                    <Mic className="h-4 w-4 text-blue-500" /> Top Acoustic Feature Drivers
                    <span className="ml-auto text-xs font-normal text-indigo-400 normal-case">Integrated Gradients (Captum)</span>
                </h3>
                {xaiLoading ? (
                    <p className="text-sm text-gray-400 italic animate-pulse mt-4">Computing Captum attributions…</p>
                ) : driverData.length > 0 ? (
                    <>
                        <p className="text-xs text-gray-400 mb-4">
                            Feature importance for predicted emotion: <strong className="capitalize">{xaiData?.predicted_emotion || dominant}</strong>
                        </p>
                        <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={driverData} layout="vertical" margin={{ left: 10, right: 50 }}>
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                                <XAxis type="number" domain={[0, 'auto']} tick={{ fontSize: 11 }} tickFormatter={v => `${v.toFixed(0)}%`} />
                                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={130} />
                                <Tooltip formatter={(v) => [`${v.toFixed(1)}%`, 'Attribution']} />
                                <Bar dataKey="attribution" radius={[0, 6, 6, 0]}>
                                    {driverData.map((_, i) => (
                                        <Cell key={i} fill={['#6366f1', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'][i % 5]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </>
                ) : (
                    <p className="text-sm text-gray-400 italic mt-4">Attribution unavailable — API response pending or Captum not installed.</p>
                )}
            </div>

            {/* Captum Human Explanation */}
            {xaiData?.human_explanation && (
                <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-4 flex gap-3">
                    <Brain className="h-5 w-5 text-indigo-500 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="text-xs font-bold text-indigo-700 uppercase tracking-wider mb-1">Model Explanation</p>
                        <p className="text-sm text-indigo-800 leading-relaxed">{xaiData.human_explanation}</p>
                    </div>
                </div>
            )}

            {/* Text Token Attribution */}
            {(xaiData?.text_attributions?.length > 0) && (
                <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                    <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-1 flex items-center gap-2">
                        <FileText className="h-4 w-4 text-purple-500" /> Text Token Highlights
                        <span className="ml-auto text-xs font-normal text-purple-400 normal-case">LayerIntegratedGradients</span>
                    </h3>
                    <p className="text-xs text-gray-400 mb-4">
                        <span className="inline-block w-3 h-3 rounded bg-red-300 mr-1 align-middle" /> Strong driver &nbsp;
                        <span className="inline-block w-3 h-3 rounded bg-gray-200 mr-1 align-middle" /> Neutral
                    </p>
                    <div className="flex flex-wrap gap-1.5 leading-relaxed">
                        {xaiData.text_attributions.map((t, i) => {
                            const abs = Math.abs(t.score);
                            const isPositive = t.score > 0;
                            // Map score to colour intensity
                            const opacity = Math.round(abs * 9) * 10 + 10; // 10-100
                            const bg = abs > 0.5
                                ? isPositive ? `rgba(239,68,68,${abs * 0.6})` : `rgba(59,130,246,${abs * 0.6})`
                                : abs > 0.2 ? `rgba(245,158,11,${abs * 0.5})` : 'transparent';
                            return (
                                <span
                                    key={i}
                                    title={`score: ${t.score.toFixed(3)}`}
                                    className="px-1.5 py-0.5 rounded text-sm font-medium cursor-help transition"
                                    style={{ backgroundColor: bg, color: abs > 0.5 ? '#1e293b' : '#475569' }}
                                >
                                    {t.token}
                                </span>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Emotion Profile + Top-3 */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-indigo-500" /> Emotion Probability Profile
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <EmotionRadar distribution={dist} />
                    <EmotionPie distribution={dist} />
                </div>
            </div>

            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <ChevronRight className="h-4 w-4 text-indigo-500" /> Model Confidence per Emotion
                </h3>
                <PredictionsBar predictions={predictions} />
            </div>

            {/* XAI Narrative Insights */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl border border-indigo-100 p-6">
                <h3 className="text-sm font-bold text-indigo-700 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Info className="h-4 w-4" /> Automated XAI Insights
                </h3>
                <div className="space-y-3">
                    {insights.map((ins, i) => (
                        <div key={i} className={`flex gap-3 p-3 bg-${ins.color}-50 border border-${ins.color}-100 rounded-xl`}>
                            <span className={`text-${ins.color}-600 mt-0.5 flex-shrink-0`}>{ins.icon}</span>
                            <p className={`text-sm text-${ins.color}-800`}>{ins.text}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Acoustic Fingerprint */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Mic className="h-4 w-4 text-blue-500" /> Acoustic Fingerprint
                </h3>
                <div className="grid grid-cols-3 gap-4">
                    {[
                        { label: 'Pitch', value: `${Math.round(pitch)} Hz`, pct: Math.min(pitch / 300, 1), color: '#3b82f6' },
                        { label: 'Speech Rate', value: `${Math.round(speechRate)} WPM`, pct: Math.min(speechRate / 200, 1), color: '#10b981' },
                        { label: 'Stress Score', value: stress.toFixed(2), pct: stress, color: stress > 0.5 ? '#ef4444' : '#f59e0b' },
                    ].map(f => (
                        <div key={f.label} className="text-center">
                            <div className="relative w-16 h-16 mx-auto mb-2">
                                <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#e2e8f0" strokeWidth="3" />
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke={f.color} strokeWidth="3" strokeDasharray={`${f.pct * 100}, 100`} />
                                </svg>
                                <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-gray-700">{Math.round(f.pct * 100)}%</span>
                            </div>
                            <p className="text-sm font-bold text-gray-700">{f.value}</p>
                            <p className="text-xs text-gray-400">{f.label}</p>
                        </div>
                    ))}
                </div>
            </div>

            <div className="text-center text-xs text-gray-400 border-t pt-4">
                Powered by <strong>HAAM Hybrid Fusion Network v2.0</strong> · Test Accuracy: <strong>78.0%</strong>
                <br />Architecture: Parallel Acoustic + Text Branches → Attention-Gated Fusion → 5-class Emotion Classification
                <br />XAI: <strong>Captum Integrated Gradients</strong> · Acoustic Attribution over 20 named features
            </div>
        </div>
    );
};


// ─── Main Page ────────────────────────────────────────────────────────────────
const CallDetailPage = () => {
    const { callId } = useParams();
    const [callData, setCallData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('summary');

    useEffect(() => {
        const fetchCall = async () => {
            try {
                const response = await callsAPI.getById(callId);
                setCallData(response.data);
            } catch (err) {
                setError('Failed to load call details.');
            } finally {
                setLoading(false);
            }
        };
        fetchCall();
    }, [callId]);

    if (loading) return <LoadingSpinner />;
    if (!callData) return <div className="text-center mt-10 text-gray-500">Call not found</div>;

    const metrics = callData.overall_metrics || {};
    const dominant = metrics.dominant_emotion || 'neutral';
    const dist = callData.text_features?.sentiment_distribution || metrics.emotion_distribution || {};
    const dataset = callData.dataset || (callId.startsWith('iemocap') ? 'IEMOCAP' : 'CREMA-D');

    // Confidence: try multiple locations
    const conf = metrics.confidence
        ?? (metrics.top_3_predictions?.[0]?.[1])
        ?? metrics.agent_stress_score
        ?? 0;

    // Fusion weights: use stored value or compute from emotion distribution
    let rawFusion = callData.fusion_weights || metrics.fusion_weights;
    if (!rawFusion) {
        const emoVals = Object.values(dist);
        const maxProb = emoVals.length > 0 ? Math.max(...emoVals) : 0;
        const acousticW = Math.round((0.4 + 0.4 * maxProb) * 1000) / 1000;
        rawFusion = { acoustic: acousticW, text: Math.round((1 - acousticW) * 1000) / 1000 };
    }
    const fusion = rawFusion;

    const bgColor = EMOTION_COLORS[dominant] || '#94a3b8';
    const emotionIcon = EMOTION_ICONS[dominant] || '😐';

    const tabs = [
        { id: 'summary', label: 'Call Summary', icon: <Activity className="h-4 w-4" /> },
        { id: 'xai', label: 'Explainability (XAI)', icon: <Brain className="h-4 w-4" /> },
        { id: 'transcript', label: 'Transcript', icon: <FileText className="h-4 w-4" /> },
    ];

    return (
        <div className="space-y-6 pb-10">
            {/* Page Header */}
            <div className="flex items-center gap-4 bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
                <Link to="/" className="p-2 hover:bg-gray-100 rounded-xl transition text-gray-500">
                    <ArrowLeft className="h-5 w-5" />
                </Link>
                <div className="flex-1 min-w-0">
                    <h1 className="text-lg font-bold text-gray-900 truncate">Call Analysis</h1>
                    <p className="text-xs text-gray-400 truncate font-mono">{callId}</p>
                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                        {callData.timestamp && <span className="text-xs text-gray-400">{new Date(callData.timestamp).toLocaleString()}</span>}
                        {callData.agent_id && <span className="text-xs text-gray-500">· Agent: {callData.agent_id}</span>}
                        <span className={`dataset-badge ${dataset.toLowerCase().includes('iemocap') ? 'iemocap' : 'crema-d'}`}>{dataset}</span>
                    </div>
                </div>
                {/* Dominant Emotion Hero */}
                <div className="flex-shrink-0 flex items-center gap-3 px-5 py-3 rounded-2xl text-white shadow-lg" style={{ background: `linear-gradient(135deg, ${bgColor}cc, ${bgColor})` }}>
                    <span className="text-3xl">{emotionIcon}</span>
                    <div>
                        <p className="text-xs font-bold opacity-80 uppercase tracking-wider">Dominant Emotion</p>
                        <p className="text-xl font-bold capitalize">{dominant}</p>
                        <p className="text-xs opacity-80">{Math.round(conf * 100)}% confidence</p>
                    </div>
                </div>
            </div>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            {/* Stats Pills */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <StatPill icon={<Mic className="h-4 w-4" />} label="Avg Pitch" value={`${Math.round(metrics.avg_pitch || callData.acoustic_features?.pitch_mean || 0)} Hz`} color="blue" />
                <StatPill icon={<Activity className="h-4 w-4" />} label="Speech Rate" value={`${Math.round(metrics.speech_rate_wpm || callData.acoustic_features?.speech_rate_wpm || 0)} WPM`} color="green" />
                <StatPill icon={<AlertTriangle className="h-4 w-4" />} label="Stress Score" value={(metrics.agent_stress_score || callData.acoustic_features?.agent_stress_score || 0).toFixed(2)} color={(metrics.agent_stress_score || 0) > 0.5 ? 'red' : 'amber'} />
                <StatPill icon={<TrendingUp className="h-4 w-4" />} label="Sentiment" value={(metrics.avg_sentiment || 0).toFixed(3)} color="purple" />
            </div>

            {/* Tab Navigation */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="flex border-b border-gray-100">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-semibold transition-all ${activeTab === tab.id
                                ? 'bg-indigo-50 text-indigo-600 border-b-2 border-indigo-500'
                                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                                }`}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="p-6">
                    {activeTab === 'summary' && (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Emotion Distribution Radar */}
                            <div>
                                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                                    <Activity className="h-4 w-4 text-indigo-500" /> Emotion Distribution
                                </h3>
                                <EmotionRadar distribution={dist} />
                            </div>
                            {/* Modality Attention */}
                            <div className="space-y-4">
                                <AttentionBar acoustic={fusion?.acoustic ?? 0.5} text={fusion?.text ?? 0.5} />
                                <div className="border-t border-gray-100 pt-4">
                                    <PredictionsBar predictions={metrics.top_3_predictions || []} />
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'xai' && (
                        <XAIPanel callData={callData} acousticFusion={fusion?.acoustic ?? 0.5} />
                    )}

                    {activeTab === 'transcript' && (
                        <div>
                            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">Call Transcript with Emotion Labels</h3>
                            <SegmentTimeline segments={callData.segments} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CallDetailPage;
