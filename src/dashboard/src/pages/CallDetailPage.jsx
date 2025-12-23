
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { callsAPI } from '../services/api';
import TranscriptView from '../components/CallDetail/TranscriptView';
import EmotionTimeline from '../components/CallDetail/EmotionTimeline';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import { ArrowLeft, Activity, Mic, BarChart2 } from 'lucide-react';

const CallDetailPage = () => {
    const { callId } = useParams();
    const [callData, setCallData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

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

    if (!callData) return <div className="text-center mt-10">Call not found</div>;

    // Helper to safely access metrics which might be missing in old data
    const metrics = callData.overall_metrics || {};

    // Schema Resilience (v1 vs v2 fallback)
    const acoustic = {
        pitch_mean: callData.acoustic_features?.pitch_mean || metrics.avg_pitch || 0,
        speech_rate_wpm: callData.acoustic_features?.speech_rate_wpm || metrics.speech_rate_wpm || 0,
        agent_stress_score: callData.acoustic_features?.agent_stress_score || metrics.agent_stress_score || 0
    };

    // Emotion Distribution - check v1/v2 locations
    const sentiment = callData.text_features?.sentiment_distribution || metrics.emotion_distribution || {};

    const predictions = metrics.top_3_predictions || [];
    const fusion = callData.fusion_weights || metrics.fusion_weights || null;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div className="flex items-center">
                    <Link to="/" className="text-gray-500 hover:text-gray-700 mr-4">
                        <ArrowLeft className="h-6 w-6" />
                    </Link>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Call Analysis: {callId}</h1>
                        <div className="text-sm text-gray-500">
                            {callData.timestamp && new Date(callData.timestamp).toLocaleString()}
                            {callData.agent_id && <span className="ml-2">• Agent: {callData.agent_id}</span>}
                            <span className={`dataset-badge ${callData.dataset?.toLowerCase().includes('iemocap') ? 'iemocap' : 'crema-d'} ml-2`}>
                                {callData.dataset || 'CREMA-D'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column: Predictions & Features */}
                <div className="space-y-6">

                    {/* Prediction Summary */}
                    <div className="bg-white p-6 rounded-lg shadow-sm border-t-4 border-blue-500">
                        <h3 className="text-gray-500 text-sm font-bold uppercase mb-2">Predicted Emotion</h3>
                        <div className="flex items-center justify-between mb-4">
                            <span className={`emotion-tag ${metrics.dominant_emotion || 'neutral'} text-2xl px-4 py-2`}>
                                {(metrics.dominant_emotion || 'Neutral').toUpperCase()}
                            </span>
                        </div>

                        <div className="mb-4">
                            <div className="flex justify-between text-sm mb-1">
                                <span className="font-semibold text-gray-700">Confidence</span>
                                <span>{Math.round((metrics.confidence || 0) * 100)}%</span>
                            </div>
                            <div className="confidence-bar">
                                <div className="bar-container">
                                    <div
                                        className="bar-fill"
                                        style={{ width: `${(metrics.confidence || 0) * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>

                        {predictions.length > 0 && (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                                <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Top Predictions</h4>
                                {predictions.map(([emo, score]) => (
                                    <div key={emo} className="flex items-center justify-between text-sm mb-2">
                                        <span className="capitalize">{emo}</span>
                                        <div className="flex items-center w-2/3">
                                            <div className="w-full bg-gray-100 rounded-full h-1.5 mr-2">
                                                <div className="bg-gray-400 h-1.5 rounded-full" style={{ width: `${score * 100}%` }}></div>
                                            </div>
                                            <span className="text-xs w-8 text-right">{Math.round(score * 100)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {fusion && (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                                <h4 className="text-xs font-bold text-gray-400 uppercase mb-3">Modality Attention</h4>
                                <div className="space-y-3">
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span>Acoustic</span>
                                            <span className="font-bold">{Math.round(fusion.acoustic * 100)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-100 rounded-full h-1.5">
                                            <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${fusion.acoustic * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span>Text</span>
                                            <span className="font-bold">{Math.round(fusion.text * 100)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-100 rounded-full h-1.5">
                                            <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${fusion.text * 100}%` }}></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Acoustic Features */}
                    <div className="bg-white p-6 rounded-lg shadow-sm">
                        <div className="flex items-center mb-4">
                            <Mic className="h-5 w-5 text-gray-500 mr-2" />
                            <h3 className="text-lg font-bold text-gray-900">Acoustic Features</h3>
                        </div>

                        <div className="space-y-6">
                            {/* Pitch */}
                            <div>
                                <div className="flex justify-between mb-1">
                                    <span className="text-sm font-medium text-gray-600">Pitch (Hz)</span>
                                    <span className="font-bold">{Math.round(acoustic.pitch_mean || 0)} Hz</span>
                                </div>
                                <div className="feature-bar">
                                    <div className="bar-fill" style={{ width: `${Math.min((acoustic.pitch_mean || 0) / 300 * 100, 100)}%` }}></div>
                                </div>
                            </div>

                            {/* Rate */}
                            <div>
                                <div className="flex justify-between mb-1">
                                    <span className="text-sm font-medium text-gray-600">Speech Rate</span>
                                    <span className="font-bold">{Math.round(acoustic.speech_rate_wpm || 0)} WPM</span>
                                </div>
                                <div className="feature-bar">
                                    <div className="bar-fill" style={{ width: `${Math.min((acoustic.speech_rate_wpm || 0) / 200 * 100, 100)}%` }}></div>
                                </div>
                            </div>

                            {/* Stress */}
                            <div>
                                <div className="flex justify-between mb-1">
                                    <span className="text-sm font-medium text-gray-600">Stress Score</span>
                                    <span className={`font-bold ${(acoustic.agent_stress_score || 0) > 0.5 ? 'text-red-500' : 'text-green-500'}`}>
                                        {(acoustic.agent_stress_score || 0).toFixed(2)}
                                    </span>
                                </div>
                                <div className="feature-bar stress">
                                    <div className="bar-fill" style={{ width: `${Math.min((acoustic.agent_stress_score || 0) * 100, 100)}%` }}></div>
                                </div>
                                {(acoustic.agent_stress_score || 0) > 0.5 && (
                                    <p className="text-xs text-red-500 mt-1">⚠️ High stress detected</p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Text Sentiment */}
                    <div className="bg-white p-6 rounded-lg shadow-sm">
                        <div className="flex items-center mb-4">
                            <BarChart2 className="h-5 w-5 text-gray-500 mr-2" />
                            <h3 className="text-lg font-bold text-gray-900">Text Sentiment</h3>
                        </div>

                        <div className="sentiment-distribution">
                            {Object.entries(sentiment).map(([emo, score]) => (
                                <div key={emo} className="sentiment-item">
                                    <div className="emotion-label text-sm">{emo}</div>
                                    <div className="sentiment-bar">
                                        <div className={`bar-fill ${emo}`} style={{ width: `${score * 100}%` }}></div>
                                    </div>
                                    <div className="text-xs font-bold w-10 text-right">{Math.round(score * 100)}%</div>
                                </div>
                            ))}
                            {Object.keys(sentiment).length === 0 && <p className="text-sm text-gray-400">No text sentiment available</p>}
                        </div>
                    </div>
                </div>

                {/* Right Column: Transcript & Timeline */}
                <div className="lg:col-span-2 space-y-6">
                    <EmotionTimeline segments={callData.segments} />
                    <TranscriptView segments={callData.segments} />

                    <div className="bg-gray-50 p-4 rounded text-center text-sm text-gray-500">
                        Analysis powered by <strong>HAAM Hybrid Fusion Network v2.0</strong>
                        <br />
                        Test Accuracy: 50.0% (Current v2 Benchmark)
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CallDetailPage;
