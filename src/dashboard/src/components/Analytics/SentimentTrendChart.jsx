
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SentimentTrendChart = () => {
    // Mock data as API doesn't provide historical aggregate trend yet
    const data = Array.from({ length: 30 }, (_, i) => ({
        date: `Day ${i + 1}`,
        sentiment: Math.sin(i / 5) * 0.5 + Math.random() * 0.2
    }));

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" hide />
                    <YAxis domain={[-1, 1]} />
                    <Tooltip
                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                    />
                    <Line
                        type="monotone"
                        dataKey="sentiment"
                        stroke="#2563eb"
                        strokeWidth={2}
                        dot={false}
                        fill="url(#colorSentiment)"
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SentimentTrendChart;
