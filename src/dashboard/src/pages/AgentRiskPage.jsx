
import React, { useState, useEffect } from 'react';
import { agentsAPI } from '../services/api';
import AgentTable from '../components/AgentRisk/AgentTable';
import RiskModal from '../components/AgentRisk/RiskModal';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';

const AgentRiskPage = () => {
    const [agents, setAgents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedAgent, setSelectedAgent] = useState(null);
    const [showHighRiskOnly, setShowHighRiskOnly] = useState(false);

    useEffect(() => {
        const fetchAgents = async () => {
            try {
                const response = await agentsAPI.getAll();
                // Since getall returns summary stats, we might need enriching with risk score.
                // Or we rely on the summary to have basic stats, then fetch risk on click.
                // Wait, the getAll endpoint in app.py only returns summary.
                // We probably need to fetch risk score for ALL agents to sort them?
                // Efficiency issue. Ideally backend should provide risk summary in list.
                // For now, let's fetch individual risk for each agent in parallel (limited) or just simple listing.
                // Given prompt requirements: "Columns: Risk Score, Risk Level".
                // I will update the frontend to fetch risk for each agent or modify backend?
                // Modifying backend is out of scope for this task (impl plan says dash).
                // I will fetch risk for each agent client-side.

                const summary = response.data;
                const enriched = await Promise.all(summary.map(async (agent) => {
                    try {
                        // Fallback to default if risk not computed
                        const riskRes = await agentsAPI.getRisk(agent.agent_id);
                        return { ...agent, ...riskRes.data };
                    } catch {
                        return { ...agent, risk_score: 0, risk_level: 'low' };
                    }
                }));

                setAgents(enriched);
            } catch (err) {
                setError('Failed to load agents.');
            } finally {
                setLoading(false);
            }
        };
        fetchAgents();
    }, []);

    const filteredAgents = showHighRiskOnly
        ? agents.filter(a => ['high', 'critical'].includes(a.risk_level))
        : agents;

    const sortedAgents = [...filteredAgents].sort((a, b) => b.risk_score - a.risk_score);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold text-gray-900">Agent Risk Assessment</h1>
                <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-700">Show only High Risk</span>
                    <button
                        onClick={() => setShowHighRiskOnly(!showHighRiskOnly)}
                        className={`relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary ${showHighRiskOnly ? 'bg-primary' : 'bg-gray-200'}`}
                    >
                        <span className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200 ${showHighRiskOnly ? 'translate-x-5' : 'translate-x-0'}`} />
                    </button>
                </div>
            </div>

            {loading ? (
                <LoadingSpinner />
            ) : (
                <AgentTable agents={sortedAgents} onSelectAgent={setSelectedAgent} />
            )}

            {selectedAgent && (
                <RiskModal
                    agent={selectedAgent}
                    onClose={() => setSelectedAgent(null)}
                />
            )}

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}
        </div>
    );
};

export default AgentRiskPage;
