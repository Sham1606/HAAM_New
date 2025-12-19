
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CallVolumeChart = () => {
    // Mock data
    const data = Array.from({ length: 30 }, (_, i) => ({
        date: `Day ${i + 1}`,
        calls: Math.floor(Math.random() * 50) + 20
    }));

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip cursor={{ fill: '#f3f4f6' }} />
                    <Bar dataKey="calls" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default CallVolumeChart;
