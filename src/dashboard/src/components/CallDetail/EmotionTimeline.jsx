
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const EmotionTimeline = ({ segments }) => {
    // Transform segments into a time series that represents duration
    // For each segment, we add a start point and an end point with the same sentiment
    // This creates a "step" or continuous flow effect
    const data = [];
    segments.forEach(seg => {
        data.push({
            time: seg.start_time,
            sentiment: seg.sentiment_score,
            text: seg.text.substring(0, 30) + '...',
            emotion: seg.emotion
        });
        data.push({
            time: seg.end_time,
            sentiment: seg.sentiment_score,
            text: seg.text.substring(0, 30) + '...',
            emotion: seg.emotion
        });
    });

    // Calculate domain to ensure chart renders even with single point
    const xDomain = data.length === 1
        ? [Math.max(0, data[0].time - 2), data[0].time + 2]
        : ['dataMin', 'dataMax'];

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
                    <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                            dataKey="time"
                            type="number"
                            domain={xDomain}
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
