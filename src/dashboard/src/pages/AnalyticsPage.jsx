
import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../services/api';
import SummaryCards from '../components/Analytics/SummaryCards';
import EmotionPieChart from '../components/Analytics/EmotionPieChart';
import SentimentTrendChart from '../components/Analytics/SentimentTrendChart';
import CallVolumeChart from '../components/Analytics/CallVolumeChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';

const AnalyticsPage = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await analyticsAPI.getOverview();
                setData(response.data);
            } catch (err) {
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

            {data && (
                <>
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
