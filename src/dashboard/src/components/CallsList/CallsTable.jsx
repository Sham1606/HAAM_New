
import React from 'react';
import { Link } from 'react-router-dom';
import { Clock, User } from 'lucide-react';

const CallsTable = ({ calls }) => {
    if (!calls.length) {
        return (
            <div className="text-center py-10 bg-white rounded-lg shadow">
                <p className="text-gray-500">No calls found.</p>
            </div>
        );
    }

    const getSentimentColor = (score) => {
        if (score > 0.3) return 'bg-green-500';
        if (score < 0) return 'bg-red-500';
        return 'bg-yellow-500';
    };

    const getEmotionBadge = (emotion) => {
        const styles = {
            joy: 'bg-green-100 text-green-800',
            anger: 'bg-red-100 text-red-800',
            sadness: 'bg-blue-100 text-blue-800',
            neutral: 'bg-gray-100 text-gray-800',
        };
        const style = styles[emotion?.toLowerCase()] || styles.neutral;

        return (
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${style}`}>
                {emotion || 'Unknown'}
            </span>
        );
    };

    return (
        <div className="bg-white shadow overflow-hidden rounded-lg">
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Call ID</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Emotion</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sentiment</th>
                            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {calls.map((call) => (
                            <tr key={call.call_id} className="hover:bg-gray-50">
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-primary">
                                    <Link to={`/call/${call.call_id}`}>{call.call_id}</Link>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 flex items-center">
                                    <User className="h-4 w-4 mr-1 text-gray-400" />
                                    {call.agent_id}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                    {new Date(call.timestamp).toLocaleString()}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap">
                                    {getEmotionBadge(call.dominant_emotion)}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${getSentimentColor(call.avg_sentiment)}`}
                                            style={{ width: `${Math.min(Math.abs(call.avg_sentiment) * 100, 100)}%` }}
                                        />
                                    </div>
                                    <span className="text-xs text-gray-400 ml-1">{call.avg_sentiment.toFixed(2)}</span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                    <Link to={`/call/${call.call_id}`} className="text-primary hover:text-blue-900">
                                        View
                                    </Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default CallsTable;
