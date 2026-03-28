import React, { useState } from 'react';
import { useAuth } from '../services/AuthContext';
import { Lock, User, AlertTriangle, Activity, Eye, EyeOff } from 'lucide-react';

const LoginPage = () => {
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(username, password);
        } catch (err) {
            const detail = err.response?.data?.detail || 'Login failed. Check credentials.';
            setError(detail);
        } finally {
            setLoading(false);
        }
    };

    const quickLogin = async (user, pass) => {
        setUsername(user);
        setPassword(pass);
        setError('');
        setLoading(true);
        try {
            await login(user, pass);
        } catch (err) {
            setError(err.response?.data?.detail || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-900 via-purple-900 to-slate-900 relative overflow-hidden">
            {/* Animated background blobs */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full opacity-10 animate-pulse" />
                <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-indigo-500 rounded-full opacity-10 animate-pulse" style={{ animationDelay: '1s' }} />
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-violet-500 rounded-full opacity-5 animate-pulse" style={{ animationDelay: '2s' }} />
            </div>

            <div className="relative z-10 w-full max-w-md mx-4">
                {/* Logo / Title */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-white/10 backdrop-blur-sm rounded-2xl mb-4 border border-white/20">
                        <Activity className="h-8 w-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-white">HAAM Framework</h1>
                    <p className="text-purple-200 mt-2 text-sm">Human Affect Analysis & Monitoring</p>
                </div>

                {/* Login Card */}
                <div className="bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 shadow-2xl p-8">
                    <h2 className="text-xl font-semibold text-white mb-6 text-center">Sign In</h2>

                    {error && (
                        <div className="mb-4 flex items-center gap-2 bg-red-500/20 border border-red-400/30 text-red-200 rounded-xl px-4 py-3 text-sm">
                            <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-purple-200 mb-1.5">Username</label>
                            <div className="relative">
                                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-purple-300" />
                                <input
                                    id="login-username"
                                    type="text"
                                    value={username}
                                    onChange={e => setUsername(e.target.value)}
                                    placeholder="Enter username"
                                    required
                                    className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-purple-300/50 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent transition"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-purple-200 mb-1.5">Password</label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-purple-300" />
                                <input
                                    id="login-password"
                                    type={showPassword ? 'text' : 'password'}
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    placeholder="Enter password"
                                    required
                                    className="w-full pl-10 pr-12 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-purple-300/50 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent transition"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-purple-300 hover:text-white transition"
                                >
                                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                </button>
                            </div>
                        </div>

                        <button
                            id="login-submit"
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-bold rounded-xl hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 shadow-lg shadow-purple-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                                    Signing in…
                                </span>
                            ) : 'Sign In'}
                        </button>
                    </form>

                    {/* Quick login buttons */}
                    <div className="mt-6 pt-5 border-t border-white/10">
                        <p className="text-xs text-purple-300 text-center mb-3">Quick Demo Login</p>
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => quickLogin('admin', 'admin123')}
                                className="px-3 py-2 bg-white/10 hover:bg-white/20 text-purple-200 text-xs rounded-lg border border-white/10 transition font-medium"
                            >
                                👑 Admin
                            </button>
                            <button
                                onClick={() => quickLogin('sham', 'pass123')}
                                className="px-3 py-2 bg-white/10 hover:bg-white/20 text-purple-200 text-xs rounded-lg border border-white/10 transition font-medium"
                            >
                                🎧 Sham
                            </button>
                            <button
                                onClick={() => quickLogin('priya', 'pass123')}
                                className="px-3 py-2 bg-white/10 hover:bg-white/20 text-purple-200 text-xs rounded-lg border border-white/10 transition font-medium"
                            >
                                🎧 Priya
                            </button>
                            <button
                                onClick={() => quickLogin('rahul', 'pass123')}
                                className="px-3 py-2 bg-white/10 hover:bg-white/20 text-purple-200 text-xs rounded-lg border border-white/10 transition font-medium"
                            >
                                🎧 Rahul
                            </button>
                        </div>
                    </div>
                </div>

                <p className="text-center text-xs text-purple-300/50 mt-6">
                    HAAM Framework v2.0 · SQLite + JWT Auth
                </p>
            </div>
        </div>
    );
};

export default LoginPage;
