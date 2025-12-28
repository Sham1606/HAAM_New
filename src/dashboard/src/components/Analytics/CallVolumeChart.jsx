
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CallVolumeChart = ({ data }) => {
    // Data expected: [{date: '2024-12-10', calls: 50}, ...]
    // If no data, fallback
    const chartData = data && data.length > 0 ? data : [];

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                        dataKey="date"
                        label={{ value: 'Date', position: 'bottom', offset: 0 }}
                        tick={{ fontSize: 12 }}
                    />
                    <YAxis
                        label={{ value: 'Calls', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip cursor={{ fill: '#f3f4f6' }} />
                    <Bar dataKey="calls" name="Call Volume" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default CallVolumeChart;
