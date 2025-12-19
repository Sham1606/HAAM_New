
import React, { useState, useEffect } from 'react';
import { callsAPI } from '../services/api';
import CallsTable from '../components/CallsList/CallsTable';
import CallFilters from '../components/CallsList/CallFilters';
import UploadModal from '../components/CallsList/UploadModal';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import { Plus } from 'lucide-react';

const CallsListPage = () => {
    const [calls, setCalls] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isUploadOpen, setIsUploadOpen] = useState(false);
    const [filters, setFilters] = useState({
        agent_id: '',
        emotion: ''
    });

    const fetchCalls = async () => {
        setLoading(true);
        try {
            const response = await callsAPI.getAll(filters);
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
    }, [filters]);

    const handleUploadSuccess = () => {
        fetchCalls();
        setIsUploadOpen(false);
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold text-gray-900">Calls List</h1>
                <button
                    onClick={() => setIsUploadOpen(true)}
                    className="bg-primary text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center"
                >
                    <Plus className="h-5 w-5 mr-2" />
                    Upload Call
                </button>
            </div>

            <CallFilters onFilterChange={setFilters} />

            {loading ? (
                <LoadingSpinner />
            ) : (
                <CallsTable calls={calls} />
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
