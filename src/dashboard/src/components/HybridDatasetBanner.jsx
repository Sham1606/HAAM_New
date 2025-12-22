import React from 'react';
import './HybridDatasetBanner.css';

const HybridDatasetBanner = ({ cremaCount, iemocapCount, metrics }) => {
    const total = cremaCount + iemocapCount;
    const cremaPercent = total > 0 ? ((cremaCount / total) * 100).toFixed(1) : 0;
    const iemocapPercent = total > 0 ? ((iemocapCount / total) * 100).toFixed(1) : 0;

    return (
        <div className="hybrid-banner">
            <div className="banner-header">
                <h2>🎯 Hybrid Validation Dataset</h2>
                <span className="total-badge">{total.toLocaleString()} samples</span>
            </div>

            <div className="dataset-split">
                <div className="split-bar">
                    <div
                        className="split-segment crema"
                        style={{ width: `${cremaPercent}%` }}
                        title={`CREMA-D: ${cremaCount} samples`}
                    />
                    <div
                        className="split-segment iemocap"
                        style={{ width: `${iemocapPercent}%` }}
                        title={`IEMOCAP: ${iemocapCount} samples`}
                    />
                </div>
            </div>

            <div className="dataset-details">
                <div className="dataset-card crema">
                    <div className="card-header">
                        <span className="dataset-badge crema-badge">CREMA-D</span>
                        <span className="dataset-type">Single Utterances</span>
                    </div>
                    <div className="card-stats">
                        <div className="stat">
                            <span className="stat-value">{cremaCount.toLocaleString()}</span>
                            <span className="stat-label">Samples ({cremaPercent}%)</span>
                        </div>
                        <div className="stat">
                            <span className="stat-value">{metrics?.crema_d_accuracy?.toFixed(1)}%</span>
                            <span className="stat-label">Accuracy</span>
                        </div>
                    </div>
                </div>

                <div className="dataset-card iemocap">
                    <div className="card-header">
                        <span className="dataset-badge iemocap-badge">IEMOCAP</span>
                        <span className="dataset-type">Conversations</span>
                    </div>
                    <div className="card-stats">
                        <div className="stat">
                            <span className="stat-value">{iemocapCount.toLocaleString()}</span>
                            <span className="stat-label">Samples ({iemocapPercent}%)</span>
                        </div>
                        <div className="stat">
                            <span className="stat-value">{metrics?.iemocap_accuracy?.toFixed(1)}%</span>
                            <span className="stat-label">Accuracy</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HybridDatasetBanner;
