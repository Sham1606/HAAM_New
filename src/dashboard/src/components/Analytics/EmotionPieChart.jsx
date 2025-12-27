
import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const EmotionPieChart = ({ distribution = {} }) => {
    const data = Object.keys(distribution || {}).map(key => ({
        name: key.charAt(0).toUpperCase() + key.slice(1),
        value: distribution[key]
    }));

    const COLORS = {
        Joy: '#10b981',
        Anger: '#ef4444',
        Sadness: '#3b82f6',
        Neutral: '#9ca3af',
        Surprise: '#f59e0b',
        Fear: '#7c3aed',
        Disgust: '#8b5cf6'
    };

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#ccc'} />
                        ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                </PieChart>
            </ResponsiveContainer>
        </div>
    );
};

export default EmotionPieChart;
