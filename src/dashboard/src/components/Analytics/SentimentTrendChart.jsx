
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SentimentTrendChart = ({ data }) => {
    // Mock data as API doesn't provide historical aggregate trend yet (fallback)
    const fallbackData = Array.from({ length: 30 }, (_, i) => ({
        date: `Day ${i + 1}`,
        sentiment: Math.sin(i / 5) * 0.5
    }));

    const chartData = data && data.length > 0 ? data : fallbackData;

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                        dataKey="date"
                        label={{ value: 'Date', position: 'bottom', offset: 0 }}
                    />
                    <YAxis
                        domain={[-1, 1]}
                        label={{ value: 'Avg Sentiment', angle: -90, position: 'insideLeft', offset: 10 }}
                    />
                    <Tooltip
                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                    />
                    <Line
                        type="monotone"
                        dataKey="sentiment"
                        stroke="#2563eb"
                        strokeWidth={2}
                        dot={true}
                        fill="url(#colorSentiment)"
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SentimentTrendChart;
