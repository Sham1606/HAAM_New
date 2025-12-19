
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Layout/Navbar';
import CallsListPage from './pages/CallsListPage';
import CallDetailPage from './pages/CallDetailPage';
import AgentRiskPage from './pages/AgentRiskPage';
import AnalyticsPage from './pages/AnalyticsPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<CallsListPage />} />
            <Route path="/call/:callId" element={<CallDetailPage />} />
            <Route path="/agents" element={<AgentRiskPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
