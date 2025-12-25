
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

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
    process: (formData) => api.post('/calls/process', formData),
    getXaiReport: (id) => api.get(`/calls/${id}/xai-report`),
    getXaiPlotURL: (id, type) => `${API_BASE_URL}/calls/${id}/xai-plot/${type}`
};

export const agentsAPI = {
    getAll: () => api.get('/agents'),
    getRisk: (id) => api.get(`/agents/${id}/risk`)
};

export const analyticsAPI = {
    getOverview: () => api.get('/analytics/overview')
};

export const modelAPI = {
    getHealth: () => api.get('/health'),
    getInfo: () => api.get('/model/info'),
    predict: (formData) => api.post('/predict/emotion', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
};

export default api;
