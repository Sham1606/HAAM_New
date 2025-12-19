
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { callsAPI } from '../services/api';
import MetricsCards from '../components/CallDetail/MetricsCards';
import TranscriptView from '../components/CallDetail/TranscriptView';
import EmotionTimeline from '../components/CallDetail/EmotionTimeline';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import { ArrowLeft } from 'lucide-react';

const CallDetailPage = () => {
    const { callId } = useParams();
    const [callData, setCallData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchCall = async () => {
            try {
                const response = await callsAPI.getById(callId);
                setCallData(response.data);
            } catch (err) {
                setError('Failed to load call details.');
            } finally {
                setLoading(false);
            }
        };
        fetchCall();
    }, [callId]);

    if (loading) return <LoadingSpinner />;

    return (
        <div className="space-y-6">
            <div className="flex items-center">
                <Link to="/" className="text-gray-500 hover:text-gray-700 mr-4">
                    <ArrowLeft className="h-6 w-6" />
                </Link>
                <h1 className="text-2xl font-bold text-gray-900">Call Analysis: {callId}</h1>
            </div>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            {callData && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-6">
                        <EmotionTimeline segments={callData.segments} />
                        <TranscriptView segments={callData.segments} />
                    </div>
                    <div className="space-y-6">
                        <MetricsCards metrics={callData.overall_metrics} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default CallDetailPage;
