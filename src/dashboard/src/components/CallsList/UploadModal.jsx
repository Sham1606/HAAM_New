
import React, { useState } from 'react';
import { callsAPI } from '../../services/api';
import { Upload, X } from 'lucide-react';

const UploadModal = ({ onClose, onSuccess }) => {
    const [file, setFile] = useState(null);
    const [agentId, setAgentId] = useState('');
    const [callId, setCallId] = useState('');
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file || !agentId || !callId) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('agent_id', agentId);
        formData.append('call_id', callId);

        try {
            await callsAPI.process(formData);
            onSuccess();
        } catch (err) {
            setError('Upload failed. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50 flex items-center justify-center">
            <div className="relative bg-white rounded-lg shadow-xl p-8 max-w-md w-full">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-400 hover:text-gray-500"
                >
                    <X className="h-6 w-6" />
                </button>

                <h2 className="text-xl font-bold mb-6 text-gray-900">Upload New Call</h2>

                {error && (
                    <div className="mb-4 bg-red-50 text-red-700 p-3 rounded text-sm">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Call ID</label>
                        <input
                            type="text"
                            required
                            value={callId}
                            onChange={(e) => setCallId(e.target.value)}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700">Agent ID</label>
                        <input
                            type="text"
                            required
                            value={agentId}
                            onChange={(e) => setAgentId(e.target.value)}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700">Audio File (.wav, .mp3)</label>
                        <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                            <div className="space-y-1 text-center">
                                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="flex text-sm text-gray-600">
                                    <label className="relative cursor-pointer bg-white rounded-md font-medium text-primary hover:text-blue-500 focus-within:outline-none">
                                        <span>Upload a file</span>
                                        <input
                                            type="file"
                                            className="sr-only"
                                            accept=".wav,.mp3"
                                            onChange={(e) => setFile(e.target.files[0])}
                                        />
                                    </label>
                                </div>
                                <p className="text-xs text-gray-500">
                                    {file ? file.name : "WAV or MP3 up to 50MB"}
                                </p>
                            </div>
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={uploading}
                        className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary ${uploading ? 'opacity-50 cursor-not-allowed' : ''
                            }`}
                    >
                        {uploading ? 'Processing...' : 'Start Processing'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default UploadModal;
