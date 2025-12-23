
import React, { useState, useRef } from 'react';
import { modelAPI } from '../services/api';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorToast from '../components/Common/ErrorToast';
import { Upload, RefreshCw, Play, Mic, Activity } from 'lucide-react';

const PredictEmotionPage = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            if (selected.size > 10 * 1024 * 1024) {
                setError("File too large. Max 10MB.");
                return;
            }
            if (!selected.name.match(/\.(wav|mp3)$/i)) {
                setError("Invalid format. Please upload .wav or .mp3");
                return;
            }
            setFile(selected);
            setResult(null);
            setError(null);
        }
    };

    const handlePredict = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('audio', file);

        try {
            const response = await modelAPI.predict(formData);
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Prediction failed. Check connection.");
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setResult(null);
        setError(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div className="text-center space-y-2">
                <h1 className="text-3xl font-bold text-gray-900">Live Emotion Prediction</h1>
                <p className="text-gray-500">Upload an audio file to analyze emotion, sentiment, and acoustic features.</p>
            </div>

            {error && <ErrorToast message={error} onClose={() => setError(null)} />}

            {/* Upload Section */}
            {!result && !loading && (
                <div className="bg-white p-10 rounded-xl shadow-sm border-2 border-dashed border-gray-300 text-center hover:border-blue-400 transition-colors">
                    <input
                        type="file"
                        ref={fileInputRef}
                        accept=".wav,.mp3"
                        onChange={handleFileChange}
                        id="audio-upload"
                    />

                    {!file ? (
                        <label htmlFor="audio-upload" className="cursor-pointer block">
                            <Upload className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-gray-700">Drop audio file here</h3>
                            <p className="text-gray-500 mt-2">or click to browse (WAV, MP3)</p>
                            <p className="text-xs text-gray-400 mt-4">Max 10MB</p>
                        </label>
                    ) : (
                        <div className="space-y-6">
                            <div className="flex items-center justify-center space-x-2 text-xl font-medium text-gray-800">
                                <Mic className="h-6 w-6 text-blue-500" />
                                <span>{file.name}</span>
                            </div>
                            <div className="flex justify-center space-x-4">
                                <button
                                    onClick={handlePredict}
                                    className="predict-button flex items-center"
                                >
                                    <Play className="h-5 w-5 mr-2" />
                                    Analyze Audio
                                </button>
                                <button
                                    onClick={handleReset}
                                    className="reset-button"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {loading && (
                <div className="bg-white p-12 rounded-xl shadow-sm text-center">
                    <div className="spinner mx-auto mb-6"></div>
                    <h3 className="text-xl font-semibold text-gray-800">Analyzing Audio Pattern...</h3>
                    <p className="text-gray-500">Extracting features and running hybrid model</p>
                </div>
            )}

            {/* Results Section */}
            {result && (
                <div className="prediction-results space-y-6">

                    {/* Header Result */}
                    <div className="bg-white p-8 rounded-xl shadow-sm border-t-8 border-purple-600 flex flex-col md:flex-row justify-between items-center text-center md:text-left">
                        <div className="mb-6 md:mb-0">
                            <h2 className="text-gray-500 font-bold uppercase tracking-wider mb-2">Detected Emotion</h2>
                            <div className={`text-5xl font-extrabold capitalize emotion-tag ${result.predicted_emotion} px-8 py-4`}>
                                {result.predicted_emotion}
                            </div>
                        </div>

                        <div className="flex items-center space-x-8">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-gray-800">{Math.round(result.confidence * 100)}%</div>
                                <div className="text-sm text-gray-500 uppercase tracking-wide">Confidence</div>
                            </div>
                            <div className="text-center">
                                <div className="text-2xl font-bold text-gray-600">
                                    {result.inference_time_ms || result.latency_ms || 0}ms
                                </div>
                                <div className="text-sm text-gray-500 uppercase tracking-wide">Latency</div>
                            </div>
                        </div>

                        {/* Modality Attention Section */}
                        {result.fusion_weights && (
                            <div className="flex-1 max-w-xs ml-8 hidden md:block">
                                <h3 className="text-xs font-bold text-gray-400 uppercase mb-2">Modality Attention</h3>
                                <div className="space-y-3">
                                    <div className="flex justify-between text-xs font-semibold">
                                        <span>Acoustic</span>
                                        <span className="text-blue-500">{Math.round(result.fusion_weights.acoustic * 100)}%</span>
                                    </div>
                                    <div className="w-full bg-gray-100 rounded-full h-2">
                                        <div 
                                            className="bg-blue-500 h-2 rounded-full transition-all duration-1000" 
                                            style={{ width: `${result.fusion_weights.acoustic * 100}%` }}
                                        ></div>
                                    </div>
                                    <div className="flex justify-between text-xs font-semibold">
                                        <span>Text</span>
                                        <span className="text-purple-500">{Math.round(result.fusion_weights.text * 100)}%</span>
                                    </div>
                                    <div className="w-full bg-gray-100 rounded-full h-2">
                                        <div 
                                            className="bg-purple-500 h-2 rounded-full transition-all duration-1000" 
                                            style={{ width: `${result.fusion_weights.text * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        <button onClick={handleReset} className="mt-4 md:mt-0 p-2 hover:bg-gray-100 rounded-full transition">
                            <RefreshCw className="h-6 w-6 text-gray-500" />
                        </button>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                        {/* Features */}
                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
                                <Activity className="w-5 h-5 mr-2 text-blue-500" />
                                Acoustic Metrics
                            </h3>

                            <div className="features-grid">
                                {result.acoustic_summary ? (
                                    <>
                                        <div className="feature-card">
                                            <div className="text-sm text-gray-500">Pitch (Mean)</div>
                                            <div className="feature-value">{result.acoustic_summary.pitch_mean} Hz</div>
                                            <div className="feature-bar">
                                                <div className="bar-fill" style={{ width: `${Math.min(result.acoustic_summary.pitch_mean / 300 * 100, 100)}%` }}></div>
                                            </div>
                                        </div>

                                        <div className="feature-card">
                                            <div className="text-sm text-gray-500">Amplitude (RMS)</div>
                                            <div className="feature-value">{result.acoustic_summary.rms_mean?.toFixed(3)}</div>
                                            <div className="feature-bar">
                                                <div className="bar-fill" style={{ width: `${Math.min(result.acoustic_summary.rms_mean * 1000, 100)}%` }}></div>
                                            </div>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="feature-card">
                                            <div className="text-sm text-gray-500">Pitch (Mean)</div>
                                            <div className="feature-value">{result.acoustic_features?.pitch || 0} Hz</div>
                                            <div className="feature-bar">
                                                <div className="bar-fill" style={{ width: `${Math.min((result.acoustic_features?.pitch || 0) / 300 * 100, 100)}%` }}></div>
                                            </div>
                                        </div>

                                        <div className="feature-card">
                                            <div className="text-sm text-gray-500">Speech Rate</div>
                                            <div className="feature-value">{result.acoustic_features?.speech_rate || 0} <span className="text-sm">WPM</span></div>
                                            <div className="feature-bar">
                                                <div className="bar-fill" style={{ width: `${Math.min((result.acoustic_features?.speech_rate || 0) / 250 * 100, 100)}%` }}></div>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Probabilities */}
                        <div className="bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
                                <Activity className="w-5 h-5 mr-2 text-purple-500" />
                                Emotion Confidence
                            </h3>

                            <div className="sentiment-distribution">
                                {(result.top_3_predictions || result.sentiment_distribution ? Object.entries(result.sentiment_distribution || {}) : []).map(([emo, score]) => (
                                    <div key={emo} className="sentiment-item">
                                        <div className="emotion-label">{emo}</div>
                                        <div className="sentiment-bar">
                                            <div className={`bar-fill ${emo}`} style={{ width: `${score * 100}%` }}></div>
                                        </div>
                                        <div className="text-sm font-bold w-12 text-right">{Math.round(score * 100)}%</div>
                                    </div>
                                ))}
                                
                                {result.top_3_predictions && result.top_3_predictions.map(([emo, score]) => (
                                    <div key={emo} className="sentiment-item">
                                        <div className="emotion-label">{emo}</div>
                                        <div className="sentiment-bar">
                                            <div className={`bar-fill ${emo}`} style={{ width: `${score * 100}%` }}></div>
                                        </div>
                                        <div className="text-sm font-bold w-12 text-right">{Math.round(score * 100)}%</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Transcript */}
                        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Transcript</h3>
                            <div className="bg-gray-50 p-6 rounded-lg text-lg text-gray-700 italic border-l-4 border-blue-400">
                                "{result.transcript}"
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PredictEmotionPage;
