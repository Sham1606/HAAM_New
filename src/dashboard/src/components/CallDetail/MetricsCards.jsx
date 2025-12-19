
import React from 'react';
import { Activity, AlertTriangle, Smile } from 'lucide-react';

const MetricsCards = ({ metrics }) => {
    const getSentimentColor = (val) => {
        if (val > 0.3) return 'text-green-600';
        if (val < 0) return 'text-red-600';
        return 'text-yellow-600';
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
            {/* Sentiment Card */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-gray-500 text-sm font-medium">Avg Sentiment</h3>
                    <Smile className="h-5 w-5 text-gray-400" />
                </div>
                <p className={`text-3xl font-bold ${getSentimentColor(metrics.avg_sentiment)}`}>
                    {metrics.avg_sentiment.toFixed(2)}
                </p>
            </div>

            {/* Emotion Card */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-gray-500 text-sm font-medium">Dominant Emotion</h3>
                    <Activity className="h-5 w-5 text-gray-400" />
                </div>
                <p className="text-3xl font-bold text-gray-900 capitalize">
                    {metrics.dominant_emotion || 'Neutral'}
                </p>
            </div>

            {/* Escalation Card */}
            <div className={`p-6 rounded-lg shadow-sm border-l-4 ${metrics.escalation_flag ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'}`}>
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-gray-700 text-sm font-medium">Escalation Status</h3>
                    <AlertTriangle className={`h-5 w-5 ${metrics.escalation_flag ? 'text-red-500' : 'text-green-500'}`} />
                </div>
                <p className={`text-xl font-bold ${metrics.escalation_flag ? 'text-red-700' : 'text-green-700'}`}>
                    {metrics.escalation_flag ? 'Escalated' : 'Normal'}
                </p>
            </div>

            {/* Stress Score Card */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-gray-500 text-sm font-medium mb-4">Agent Stress Score</h3>
                <div className="relative pt-1">
                    <div className="flex mb-2 items-center justify-between">
                        <div>
                            <span className="text-xs font-semibold inline-block text-primary">
                                {metrics.agent_stress_score.toFixed(1)} / 10
                            </span>
                        </div>
                    </div>
                    <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
                        <div
                            style={{ width: `${(metrics.agent_stress_score / 10) * 100}%` }}
                            className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${metrics.agent_stress_score > 7 ? 'bg-red-500' : 'bg-blue-500'
                                }`}
                        ></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MetricsCards;
