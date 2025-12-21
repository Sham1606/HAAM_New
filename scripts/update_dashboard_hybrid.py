
import os

DASHBOARD_SRC = r"D:\haam_framework\src\dashboard\src"
COMPONENTS_DIR = os.path.join(DASHBOARD_SRC, "components")
PAGES_DIR = os.path.join(DASHBOARD_SRC, "pages")

# Component: HybridDatasetBanner
BANNER_JSX = """
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
            style={{width: `${cremaPercent}%`}}
            title={`CREMA-D: ${cremaCount} samples`}
          />
          <div 
            className="split-segment iemocap" 
            style={{width: `${iemocapPercent}%`}}
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
"""

BANNER_CSS = """
.hybrid-banner {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  border-radius: 16px;
  margin-bottom: 30px;
  box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}
.banner-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.banner-header h2 { margin: 0; font-size: 24px; font-weight: 600; color: white; }
.total-badge { background: rgba(255, 255, 255, 0.2); padding: 8px 20px; border-radius: 20px; font-size: 20px; font-weight: bold; }
.split-bar { height: 20px; display: flex; border-radius: 20px; overflow: hidden; margin-bottom: 25px; background: #333; }
.split-segment { height: 100%; }
.split-segment.crema { background-color: #3498db; }
.split-segment.iemocap { background-color: #2ecc71; }
.dataset-details { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.dataset-card { background: rgba(255, 255, 255, 0.15); padding: 20px; border-radius: 12px; }
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.dataset-badge { padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 14px; }
.crema-badge { background: #3498db; }
.iemocap-badge { background: #2ecc71; }
.card-stats { display: flex; justify-content: space-around; }
.stat { text-align: center; }
.stat-value { display: block; font-size: 20px; font-weight: bold; }
.stat-label { font-size: 12px; opacity: 0.8; }
"""

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created {path}")

def main():
    write_file(os.path.join(COMPONENTS_DIR, "HybridDatasetBanner.jsx"), BANNER_JSX)
    write_file(os.path.join(COMPONENTS_DIR, "HybridDatasetBanner.css"), BANNER_CSS)
    
    print("Dashboard components created.")

if __name__ == "__main__":
    main()
