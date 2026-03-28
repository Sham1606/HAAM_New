
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './services/AuthContext';
import Navbar from './components/Layout/Navbar';

// Pages — Shared
import LoginPage from './pages/LoginPage';
import CallDetailPage from './pages/CallDetailPage';
import LiveAnalysisPage from './pages/LiveAnalysisPage';

// Pages — Admin Only
import CallsListPage from './pages/CallsListPage';
import AgentRiskPage from './pages/AgentRiskPage';
import AnalyticsPage from './pages/AnalyticsPage';
import AlertsSettingsPage from './pages/AlertsSettingsPage';
import AgentGridPage from './pages/AgentGridPage';

// Pages — Agent Only
import AgentDashboardPage from './pages/AgentDashboardPage';
import MyCallsPage from './pages/MyCallsPage';
import MyFeedbackPage from './pages/MyFeedbackPage';

// Protected route
function ProtectedRoute({ children }) {
    const { isAuthenticated, loading } = useAuth();
    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="text-center">
                <svg className="animate-spin h-8 w-8 text-indigo-500 mx-auto mb-3" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <p className="text-sm text-gray-400">Loading…</p>
            </div>
        </div>
    );
    return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function AppRoutes() {
    const { isAuthenticated, isAdmin, loading } = useAuth();

    if (loading) return null;

    // Unauthenticated → Login
    if (!isAuthenticated) {
        return (
            <Routes>
                <Route path="/login" element={<LoginPage />} />
                <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            <Navbar />
            <main className="container mx-auto px-4 py-8">
                <Routes>
                    {isAdmin ? (
                        /* ── Admin Routes ────────────────────────────────── */
                        <>
                            <Route path="/" element={<CallsListPage />} />
                            <Route path="/call/:callId" element={<CallDetailPage />} />
                            <Route path="/agents" element={<AgentRiskPage />} />
                            <Route path="/analytics" element={<AnalyticsPage />} />
                            <Route path="/agent-grid" element={<AgentGridPage />} />
                            <Route path="/alerts" element={<AlertsSettingsPage />} />
                            <Route path="/live" element={<LiveAnalysisPage />} />
                        </>
                    ) : (
                        /* ── Agent Routes ────────────────────────────────── */
                        <>
                            <Route path="/" element={<AgentDashboardPage />} />
                            <Route path="/live" element={<LiveAnalysisPage />} />
                            <Route path="/my-calls" element={<MyCallsPage />} />
                            <Route path="/my-feedback" element={<MyFeedbackPage />} />
                            <Route path="/call/:callId" element={<CallDetailPage />} />
                        </>
                    )}
                    <Route path="/login" element={<Navigate to="/" replace />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </main>
        </div>
    );
}

function App() {
    return (
        <BrowserRouter>
            <AuthProvider>
                <AppRoutes />
            </AuthProvider>
        </BrowserRouter>
    );
}

export default App;
