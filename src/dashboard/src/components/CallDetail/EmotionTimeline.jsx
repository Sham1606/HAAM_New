
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const EmotionTimeline = ({ segments }) => {
    // Map segments to timeline points
    // We take the mid-point of each segment
    const data = segments.map(seg => ({
        time: (seg.start_time + seg.end_time) / 2,
        sentiment: seg.sentiment_score,
        text: seg.text.substring(0, 30) + '...',
        emotion: seg.emotion
    }));

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const point = payload[0].payload;
            return (
                <div className="bg-white p-3 border border-gray-200 shadow-lg rounded text-sm">
                    <p className="font-semibold text-gray-700">{Math.round(point.time)}s</p>
                    <p className="text-primary">Sentiment: {point.sentiment.toFixed(2)}</p>
                    <p className="text-gray-500 italic">"{point.text}"</p>
                    <p className="text-xs text-gray-400 uppercase mt-1">{point.emotion}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Sentiment Timeline</h2>
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                            dataKey="time"
                            type="number"
                            tickFormatter={(val) => `${Math.round(val)}s`}
                            label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }}
                        />
                        <YAxis domain={[-1, 1]} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
                        <Line
                            type="monotone"
                            dataKey="sentiment"
                            stroke="#2563eb"
                            strokeWidth={2}
                            dot={{ r: 4 }}
                            activeDot={{ r: 6 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default EmotionTimeline;
