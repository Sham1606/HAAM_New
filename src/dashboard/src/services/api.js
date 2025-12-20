
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: { 'Content-Type': 'application/json' }
});

// Error handling interceptor
api.interceptors.response.use(
    response => response,
    error => {
        console.error('API Error:', error);
        return Promise.reject(error);
    }
);

export const callsAPI = {
    getAll: (params) => api.get('/calls', { params }),
    getById: (id) => api.get(`/calls/${id}`),
    process: (formData) => api.post('/calls/process', formData)
};

export const agentsAPI = {
    getAll: () => api.get('/agents'),
    getRisk: (id) => api.get(`/agents/${id}/risk`)
};

export const analyticsAPI = {
    getOverview: () => api.get('/analytics/overview')
};

export default api;
