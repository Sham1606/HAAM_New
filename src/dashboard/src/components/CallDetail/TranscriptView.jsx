
import React from 'react';

const TranscriptView = ({ segments }) => {
    const getBubbleStyle = (emotion) => {
        switch (emotion.toLowerCase()) {
            case 'anger': return 'bg-red-50 border-l-4 border-red-500';
            case 'joy': return 'bg-green-50 border-l-4 border-green-500';
            case 'neutral': return 'bg-gray-50 border-l-4 border-gray-400';
            case 'sadness': return 'bg-blue-50 border-l-4 border-blue-500';
            default: return 'bg-gray-50';
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Transcript</h2>
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                {segments.map((seg, idx) => (
                    <div key={idx} className={`p-4 rounded-r-md ${getBubbleStyle(seg.emotion)}`}>
                        <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-mono text-gray-500">
                                {seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s
                            </span>
                            <span className="text-xs font-medium uppercase text-gray-400">
                                {seg.emotion}
                            </span>
                        </div>
                        <p className="text-gray-800 text-sm leading-relaxed">{seg.text}</p>
                        <div className="mt-2 flex justify-end">
                            <span className={`text-xs ${seg.sentiment_score < 0 ? 'text-red-500' : 'text-green-500'}`}>
                                Sentiment: {seg.sentiment_score.toFixed(2)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default TranscriptView;
