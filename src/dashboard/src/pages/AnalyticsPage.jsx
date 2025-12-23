
import React, { useState, useEffect } from 'react';
import { analyticsAPI, modelAPI } from '../services/api';
import SummaryCards from '../components/Analytics/SummaryCards';
import EmotionPieChart from '../components/Analytics/EmotionPieChart';
import SentimentTrendChart from '../components/Analytics/SentimentTrendChart';
import CallVolumeChart from '../components/Analytics/CallVolumeChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';

const AnalyticsPage = () => {
    const [data, setData] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [analyticsRes, modelRes] = await Promise.all([
                    analyticsAPI.getOverview(),
                    modelAPI.getInfo()
                ]);
                setData(analyticsRes.data);
                setModelInfo(modelRes.data);
            } catch (err) {
                console.error(err);
                setError('Failed to load analytics.');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <LoadingSpinner />;

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-900">Analytics Overview</h1>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            {/* Hybrid Model Performance Card */}
            {modelInfo && (
                <div className="hybrid-model">
                    <div className="flex justify-between items-start">
                        <div>
                            <h2 className="text-2xl font-bold">{modelInfo.model_name}</h2>
                            <p className="opacity-90">{modelInfo.architecture}</p>
                        </div>
                        <div className="text-right">
                            <div className="text-3xl font-bold">{Math.round(modelInfo.test_accuracy * 1000) / 10}%</div>
                            <div className="text-sm opacity-90">Test Accuracy</div>
                        </div>
                    </div>

                    <div className="model-stats">
                        <div className="bg-white/10 p-4 rounded-lg">
                            <div className="text-2xl font-bold">{modelInfo.training_samples.toLocaleString()}</div>
                            <div className="text-sm opacity-80">Training Samples</div>
                        </div>
                        <div className="bg-white/10 p-4 rounded-lg">
                            <div className="text-2xl font-bold">{modelInfo.datasets.length}</div>
                            <div className="text-sm opacity-80">Datasets ({modelInfo.datasets.join(' + ')})</div>
                        </div>
                        <div className="bg-white/10 p-4 rounded-lg">
                            <div className="text-2xl font-bold">{modelInfo.emotions.length}</div>
                            <div className="text-sm opacity-80">Target Emotions</div>
                        </div>
                    </div>

                    <div className="mt-4">
                        <span className="text-sm opacity-90 mr-2">Supported Emotions:</span>
                        {modelInfo.emotions.map(emo => (
                            <span key={emo} className={`emotion-tag ${emo} border border-white/20`}>
                                {emo}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {data && (
                <>
                    {/* Dataset Distribution & Validation Metrics */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Dataset Stats */}
                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Training Data Distribution</h3>
                            <div className="space-y-4">
                                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border-l-4 border-blue-500">
                                    <div>
                                        <div className="font-bold text-gray-800">CREMA-D</div>
                                        <div className="text-sm text-gray-500">Acted Speech</div>
                                    </div>
                                    <div className="text-xl font-bold text-blue-600">
                                        {data.dataset_breakdown?.['CREMA-D']?.toLocaleString() || 0}
                                    </div>
                                </div>
                                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border-l-4 border-green-500">
                                    <div>
                                        <div className="font-bold text-gray-800">IEMOCAP</div>
                                        <div className="text-sm text-gray-500">Conversational Speech</div>
                                    </div>
                                    <div className="text-xl font-bold text-green-600">
                                        {data.dataset_breakdown?.['IEMOCAP']?.toLocaleString() || 0}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Validation Metrics */}
                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Validation Metrics</h3>
                            <div className="space-y-6">
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="font-medium">CREMA-D Only (Clean)</span>
                                        <span className="font-bold">{data.validation_metrics?.crema_d_accuracy}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                        <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${data.validation_metrics?.crema_d_accuracy}%` }}></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="font-medium">IEMOCAP Only (Noisy)</span>
                                        <span className="font-bold">{data.validation_metrics?.iemocap_accuracy}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                        <div className="bg-green-500 h-2.5 rounded-full" style={{ width: `${data.validation_metrics?.iemocap_accuracy}%` }}></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="font-medium text-purple-700">Hybrid Model (Combined)</span>
                                        <span className="font-bold text-purple-700">{data.validation_metrics?.combined_accuracy}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                        <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${data.validation_metrics?.combined_accuracy}%` }}></div>
                                    </div>
                                    <p className="text-xs text-gray-500 mt-2">
                                        * Hybrid model balances performance across diverse audio conditions.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <SummaryCards
                        totalCalls={data.total_calls}
                        totalAgents={data.total_agents}
                        avgSentiment={data.avg_sentiment}
                        highRiskAgents={data.high_risk_agents}
                    />

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Emotion Distribution</h3>
                            <EmotionPieChart distribution={data.emotion_distribution} />
                        </div>

                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Call Volume (30 Days)</h3>
                            <CallVolumeChart />
                        </div>

                        <div className="bg-white p-6 rounded-lg shadow-sm lg:col-span-2">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Sentiment Trend</h3>
                            <SentimentTrendChart />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default AnalyticsPage;
