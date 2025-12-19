
import React from 'react';
import { Phone, Users, TrendingUp, AlertTriangle } from 'lucide-react';

const SummaryCard = ({ title, value, icon: Icon, color, subtext }) => (
    <div className="bg-white p-6 rounded-lg shadow-sm flex items-center">
        <div className={`p-3 rounded-full mr-4 ${color}`}>
            <Icon className="h-6 w-6 text-white" />
        </div>
        <div>
            <p className="text-sm font-medium text-gray-500">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {subtext && <p className="text-xs text-gray-400 mt-1">{subtext}</p>}
        </div>
    </div>
);

const SummaryCards = ({ totalCalls, totalAgents, avgSentiment, highRiskAgents }) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <SummaryCard
                title="Total Calls"
                value={totalCalls}
                icon={Phone}
                color="bg-primary"
            />
            <SummaryCard
                title="Active Agents"
                value={totalAgents}
                icon={Users}
                color="bg-purple-500"
            />
            <SummaryCard
                title="Avg Sentiment"
                value={avgSentiment.toFixed(2)}
                icon={TrendingUp}
                color={avgSentiment > 0 ? "bg-green-500" : "bg-red-500"}
            />
            <SummaryCard
                title="High Risk Agents"
                value={highRiskAgents}
                icon={AlertTriangle}
                color={highRiskAgents > 0 ? "bg-red-600" : "bg-green-500"}
                subtext="Requires immediate attention"
            />
        </div>
    );
};

export default SummaryCards;
