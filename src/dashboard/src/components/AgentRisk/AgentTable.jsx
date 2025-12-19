
import React from 'react';
import RiskBadge from './RiskBadge';
import { ChevronRight } from 'lucide-react';

const AgentTable = ({ agents, onSelectAgent }) => {
    return (
        <div className="bg-white shadow overflow-hidden rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                    <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent ID</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Calls</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Sentiment</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Score</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Level</th>
                        <th className="px-6 py-3 relative">
                            <span className="sr-only">Details</span>
                        </th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {agents.map((agent) => (
                        <tr
                            key={agent.agent_id}
                            onClick={() => onSelectAgent(agent)}
                            className="hover:bg-gray-50 cursor-pointer transition-colors"
                        >
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                {agent.agent_id}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {agent.call_count}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {agent.avg_sentiment?.toFixed(3)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-700">
                                {(agent.risk_score * 100).toFixed(0)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                <RiskBadge level={agent.risk_level} />
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                <ChevronRight className="h-5 w-5 text-gray-400" />
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default AgentTable;
