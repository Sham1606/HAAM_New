/**
 * AuthContext: React context providing JWT authentication state.
 * 
 * Provides: user, token, login(), logout(), isAdmin, isAuthenticated
 * Persists token in localStorage for page refreshes.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const AuthContext = createContext(null);

const API_BASE = 'http://localhost:8000';

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(() => localStorage.getItem('haam_token'));
    const [loading, setLoading] = useState(true);

    // Configure axios default auth header when token changes
    useEffect(() => {
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            localStorage.setItem('haam_token', token);
        } else {
            delete axios.defaults.headers.common['Authorization'];
            localStorage.removeItem('haam_token');
        }
    }, [token]);

    // On mount: validate stored token by fetching /auth/me
    useEffect(() => {
        if (token) {
            axios.get(`${API_BASE}/auth/me`, {
                headers: { Authorization: `Bearer ${token}` }
            })
                .then(res => {
                    setUser(res.data);
                    setLoading(false);
                })
                .catch(() => {
                    // Token expired or invalid
                    setToken(null);
                    setUser(null);
                    setLoading(false);
                });
        } else {
            setLoading(false);
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const login = useCallback(async (username, password) => {
        const res = await axios.post(`${API_BASE}/auth/login`, { username, password });
        const data = res.data;
        setToken(data.access_token);
        setUser({
            id: data.agent_id,
            role: data.role,
            display_name: data.display_name,
            username: username,
        });
        return data;
    }, []);

    const logout = useCallback(async () => {
        try {
            await axios.post(`${API_BASE}/auth/logout`, {}, {
                headers: { Authorization: `Bearer ${token}` }
            });
        } catch (_) { /* ignore */ }
        setToken(null);
        setUser(null);
    }, [token]);

    const value = {
        user,
        token,
        login,
        logout,
        loading,
        isAuthenticated: !!user,
        isAdmin: user?.role === 'admin',
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
}

export default AuthContext;
