import React, { useState, useEffect } from 'react';
import { callsAPI } from '../../services/api';
import { Info, AlertTriangle, Zap, MessageSquare } from 'lucide-react';

const ExplainabilityView = ({ callId }) => {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                const response = await callsAPI.getXaiReport(callId);
                setReport(response.data.content);
            } catch (err) {
                console.error("Failed to load XAI report", err);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, [callId]);

    const plots = [
        { id: 'trajectory', title: 'Emotion Trajectory', icon: <Zap className="h-4 w-4" /> },
        { id: 'flow', title: 'Sentiment Flow', icon: <MessageSquare className="h-4 w-4" /> },
        { id: 'importance', title: 'Modality Importance', icon: <Info className="h-4 w-4" /> }
    ];

    if (loading) return <div className="p-10 text-center text-gray-400">Loading explainability insights...</div>;

    return (
        <div className="space-y-8 animate-fadeIn">
            {/* Visual Analytics Grid */}
            <div className="grid grid-cols-1 gap-6">
                {plots.map(plot => (
                    <div key={plot.id} className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                        <div className="px-4 py-3 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
                            <span className="flex items-center text-sm font-bold text-gray-700">
                                {React.cloneElement(plot.icon, { className: 'mr-2 text-blue-500' })}
                                {plot.title}
                            </span>
                        </div>
                        <div className="p-2 bg-gray-900 flex justify-center">
                            <img
                                src={callsAPI.getXaiPlotURL(callId, plot.id)}
                                alt={plot.title}
                                className="max-h-[400px] w-auto object-contain"
                                onError={(e) => {
                                    e.target.style.display = 'none';
                                    e.target.parentNode.innerHTML = '<div class="p-10 text-gray-500 italic">Visualization not yet generated for this call. Run generate_xai_reports.py to populate.</div>';
                                }}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Automated Summary Report */}
            {report && (
                <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-r-xl">
                    <div className="flex items-center mb-4">
                        <Info className="h-5 w-5 text-blue-600 mr-2" />
                        <h3 className="text-lg font-bold text-blue-900">Automated XAI Insights</h3>
                    </div>
                    <div className="prose prose-blue max-w-none text-blue-800 text-sm whitespace-pre-wrap">
                        {/* Basic MD rendering or plain text fallback */}
                        {report.split('## 📊 Critical Insights')[1] || "Full report content available in results/xai_reports."}
                    </div>
                </div>
            )}

            {!report && (
                <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded flex items-start">
                    <AlertTriangle className="h-5 w-5 text-amber-500 mr-3 mt-0.5" />
                    <div>
                        <p className="text-sm text-amber-800 font-bold">In-Depth Analysis Pending</p>
                        <p className="text-xs text-amber-700">The detailed explainability report for this call hasn't been generated yet. Please trigger the batch reporting script.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ExplainabilityView;
