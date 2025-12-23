
import React from 'react';
import { X, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';
import RiskBadge from './RiskBadge';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const RiskModal = ({ agent, onClose }) => {
    // Use real trend data if available, otherwise fallback to a more neutral placeholder
    const trendData = agent.sentiment_history && agent.sentiment_history.length > 0
        ? agent.sentiment_history.map((h, i) => ({ day: h.day || i.toString(), score: h.score }))
        : [
            { day: '1', score: 0.1 }, { day: '5', score: 0.1 }, { day: '10', score: 0.1 },
            { day: '15', score: 0.1 }, { day: '20', score: 0.1 }, { day: '25', score: 0.1 }
        ];

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75 overflow-y-auto h-full w-full z-50 flex items-center justify-center p-4">
            <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex justify-between items-center p-6 border-b">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-900">Risk Profile: {agent.agent_id}</h2>
                        <p className="text-sm text-gray-500">Last updated: Today</p>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-500">
                        <X className="h-6 w-6" />
                    </button>
                </div>

                <div className="p-6 space-y-8">
                    {/* Score Section */}
                    <div className="flex items-center space-x-6">
                        <div className="text-center">
                            <span className="block text-4xl font-extrabold text-gray-900">{(agent.risk_score * 100).toFixed(0)}</span>
                            <span className="text-xs text-gray-500">Risk Score</span>
                        </div>
                        <RiskBadge level={agent.risk_level} />
                        <div className="flex-1 bg-gray-50 p-4 rounded-md">
                            <div className="flex items-center text-sm text-gray-700">
                                <TrendingUp className="h-4 w-4 mr-2 text-gray-400" />
                                Trend: <span className="font-semibold ml-1 capitalize text-red-600">{agent.trend_direction || 'Stable'}</span>
                            </div>
                        </div>
                    </div>

                    {/* Risk Factors */}
                    <div>
                        <h3 className="text-lg font-bold text-gray-900 mb-4">Risk Factors</h3>
                        <div className="space-y-3">
                            {agent.risk_factors && agent.risk_factors.length > 0 ? (
                                agent.risk_factors.map((factor, idx) => (
                                    <div key={idx} className="flex items-start bg-red-50 p-3 rounded-lg">
                                        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                                        <div className="flex-1">
                                            <p className="text-sm font-bold text-gray-900">{factor.factor}</p>
                                            <p className="text-xs text-gray-600 mt-1">{factor.description}</p>
                                            <div className="mt-2 w-full bg-red-200 rounded-full h-1.5">
                                                <div
                                                    className="bg-red-500 h-1.5 rounded-full"
                                                    style={{ width: `${factor.contribution * 100}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                        <span className="text-xs font-bold text-red-700 ml-3">
                                            {(factor.contribution * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                ))
                            ) : (
                                <p className="text-gray-500 italic">No significant risk factors identified.</p>
                            )}
                        </div>
                    </div>

                    {/* Recommendations */}
                    <div>
                        <h3 className="text-lg font-bold text-gray-900 mb-4">Recommendations</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {agent.recommendations && agent.recommendations.map((rec, idx) => (
                                <div key={idx} className="flex items-start p-3 border border-green-100 rounded-lg bg-green-50">
                                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                                    <p className="text-sm text-green-800">{rec}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* 30 Day Trend */}
                    <div>
                        <h3 className="text-lg font-bold text-gray-900 mb-4">30-Day Sentiment Trend</h3>
                        <div className="h-48 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={trendData}>
                                    <XAxis dataKey="day" hide />
                                    <YAxis />
                                    <Tooltip />
                                    <Line type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RiskModal;
