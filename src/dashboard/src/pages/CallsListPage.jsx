
import React, { useState, useEffect } from 'react';
import { callsAPI } from '../services/api';
import CallsTable from '../components/CallsList/CallsTable';
import UploadModal from '../components/CallsList/UploadModal';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import { Plus, Filter } from 'lucide-react';

const CallsListPage = () => {
    const [calls, setCalls] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isUploadOpen, setIsUploadOpen] = useState(false);

    // Filters
    const [datasetFilter, setDatasetFilter] = useState('');
    const [agentFilter, setAgentFilter] = useState('');

    const fetchCalls = async () => {
        setLoading(true);
        try {
            const params = { limit: 100 };
            if (datasetFilter) params.dataset = datasetFilter;
            if (agentFilter) params.agent_id = agentFilter;

            const response = await callsAPI.getAll(params);
            setCalls(response.data);
            setError(null);
        } catch (err) {
            setError('Failed to load calls.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCalls();
    }, [datasetFilter, agentFilter]);

    const handleUploadSuccess = () => {
        fetchCalls();
        setIsUploadOpen(false);
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row justify-between items-center bg-white p-6 rounded-lg shadow-sm">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Calls List</h1>
                    <p className="text-gray-500 text-sm">Manage and analyze recorded calls</p>
                </div>
                <button
                    onClick={() => setIsUploadOpen(true)}
                    className="mt-4 md:mt-0 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 flex items-center font-medium transition"
                >
                    <Plus className="h-5 w-5 mr-2" />
                    Upload Call
                </button>
            </div>

            {/* Filters Bar */}
            <div className="bg-white p-4 rounded-lg shadow-sm flex flex-wrap items-center gap-4">
                <div className="flex items-center text-gray-500 font-medium">
                    <Filter className="h-5 w-5 mr-2" />
                    Filters:
                </div>

                <select
                    value={datasetFilter}
                    onChange={(e) => setDatasetFilter(e.target.value)}
                    className="border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                >
                    <option value="">All Datasets</option>
                    <option value="CREMA-D">CREMA-D (Acted)</option>
                    <option value="IEMOCAP">IEMOCAP (Live)</option>
                </select>

                <input
                    type="text"
                    placeholder="Filter by Agent ID..."
                    value={agentFilter}
                    onChange={(e) => setAgentFilter(e.target.value)}
                    className="border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                />
            </div>

            {loading ? (
                <LoadingSpinner />
            ) : (
                <div className="bg-white rounded-lg shadow overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Call ID</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dataset</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Emotion</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sentiment</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {calls.map((call) => (
                                <tr key={call.call_id} className="hover:bg-gray-50 transition">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">
                                        <a href={`/call/${call.call_id}`} className="hover:underline">{call.call_id}</a>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {call.agent_id}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`dataset-badge ${call.call_id.startsWith('iemocap') ? 'iemocap' : 'crema-d'}`}>
                                            {call.dataset || (call.call_id.startsWith('iemocap') ? 'IEMOCAP' : 'CREMA-D')}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`emotion-tag ${call.dominant_emotion || 'neutral'}`}>
                                            {call.dominant_emotion || 'neutral'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {call.avg_sentiment?.toFixed(2)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {new Date(call.timestamp).toLocaleDateString()}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                        <a href={`/call/${call.call_id}`} className="text-blue-600 hover:text-blue-900">View</a>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {calls.length === 0 && (
                        <div className="p-8 text-center text-gray-500">No calls found matching filters.</div>
                    )}
                </div>
            )}

            {isUploadOpen && (
                <UploadModal
                    onClose={() => setIsUploadOpen(false)}
                    onSuccess={handleUploadSuccess}
                />
            )}

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}
        </div>
    );
};

export default CallsListPage;
