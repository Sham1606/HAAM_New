import React, { useState, useEffect } from 'react';
import { Mail, MessageSquare, Bell, BellOff, Send, CheckCircle, AlertCircle, Info } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

// ─── Toggle ──────────────────────────────────────────────────────────────────
const Toggle = ({ checked, onChange }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-6 rounded-full transition-colors duration-200 ${checked ? 'bg-indigo-600' : 'bg-gray-200'}`}
    >
        <span className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200 ${checked ? 'translate-x-4' : ''}`} />
    </button>
);

// ─── Field ────────────────────────────────────────────────────────────────────
const Field = ({ label, type = 'text', value, onChange, placeholder, hint }) => (
    <div>
        <label className="block text-xs font-semibold text-gray-600 mb-1">{label}</label>
        <input
            type={type}
            value={value}
            onChange={e => onChange(e.target.value)}
            placeholder={placeholder}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
        />
        {hint && <p className="text-xs text-gray-400 mt-1">{hint}</p>}
    </div>
);

// ─── Main ─────────────────────────────────────────────────────────────────────
const AlertsSettingsPage = () => {
    const [config, setConfig] = useState(null);
    const [saving, setSaving] = useState(false);
    const [testing, setTesting] = useState(false);
    const [toast, setToast] = useState(null);   // {type: 'success'|'error', msg}

    const showToast = (type, msg) => {
        setToast({ type, msg });
        setTimeout(() => setToast(null), 4000);
    };

    useEffect(() => {
        fetch(`${API_BASE}/alerts/config`)
            .then(r => r.json())
            .then(setConfig)
            .catch(() => showToast('error', 'Could not load config. Is the API running?'));
    }, []);

    const setEmail = (key, val) =>
        setConfig(p => ({ ...p, email: { ...p.email, [key]: val } }));
    const setSlack = (key, val) =>
        setConfig(p => ({ ...p, slack: { ...p.slack, [key]: val } }));

    const save = async () => {
        if (!config) return;
        setSaving(true);
        try {
            // Don't send the redacted placeholder back — backend preserves existing password
            const payload = { ...config };
            if (payload.email?.password === '••••••••') {
                payload.email = { ...payload.email, password: '' };
            }
            const res = await fetch(`${API_BASE}/alerts/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error(await res.text());
            showToast('success', 'Settings saved successfully!');
        } catch (e) {
            showToast('error', `Save failed: ${e.message}`);
        } finally {
            setSaving(false);
        }
    };

    const testAlert = async () => {
        setTesting(true);
        try {
            const res = await fetch(`${API_BASE}/alerts/test`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Alert failed');
            showToast('success', `Test sent to: ${data.channels.join(', ') || 'no channels enabled'}`);
        } catch (e) {
            showToast('error', `Test failed: ${e.message}`);
        } finally {
            setTesting(false);
        }
    };

    if (!config) return (
        <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
            Loading configuration…
        </div>
    );

    return (
        <div className="max-w-2xl mx-auto space-y-6 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-rose-600 to-orange-500 rounded-2xl p-6 text-white shadow-lg">
                <h1 className="text-2xl font-bold flex items-center gap-3">
                    <Bell className="h-7 w-7" /> Alert Settings
                </h1>
                <p className="text-rose-100 mt-1 text-sm">
                    Get notified via Email or Slack when agent risk exceeds the threshold
                </p>
            </div>

            {/* Toast */}
            {toast && (
                <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border text-sm font-medium ${toast.type === 'success' ? 'bg-green-50 border-green-200 text-green-800' : 'bg-red-50 border-red-200 text-red-800'
                    }`}>
                    {toast.type === 'success' ? <CheckCircle className="h-4 w-4 flex-shrink-0" /> : <AlertCircle className="h-4 w-4 flex-shrink-0" />}
                    {toast.msg}
                </div>
            )}

            {/* Risk Threshold */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-4">
                <h2 className="font-bold text-gray-700 flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-amber-500" /> Risk Threshold
                </h2>
                <div>
                    <label className="text-xs font-semibold text-gray-600 mb-1 block">
                        Alert when risk ≥ <strong className="text-indigo-600">{Math.round((config.risk_threshold ?? 0.6) * 100)}%</strong>
                    </label>
                    <input
                        type="range" min="0.1" max="1.0" step="0.05"
                        value={config.risk_threshold ?? 0.6}
                        onChange={e => setConfig(p => ({ ...p, risk_threshold: parseFloat(e.target.value) }))}
                        className="w-full accent-indigo-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                        <span>10% (very sensitive)</span>
                        <span>100% (never)</span>
                    </div>
                </div>
            </div>

            {/* Email */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="font-bold text-gray-700 flex items-center gap-2">
                        <Mail className="h-5 w-5 text-blue-500" /> Email Alerts
                    </h2>
                    <Toggle checked={!!config.email?.enabled} onChange={v => setEmail('enabled', v)} />
                </div>
                {config.email?.enabled && (
                    <div className="space-y-3 pt-2 border-t border-gray-50">
                        <div className="grid grid-cols-2 gap-3">
                            <Field label="SMTP Host" value={config.email?.smtp_host ?? ''} onChange={v => setEmail('smtp_host', v)} placeholder="smtp.gmail.com" />
                            <Field label="SMTP Port" type="number" value={config.email?.smtp_port ?? 587} onChange={v => setEmail('smtp_port', parseInt(v))} placeholder="587" />
                        </div>
                        <Field label="Sender Email" type="email" value={config.email?.sender ?? ''} onChange={v => setEmail('sender', v)} placeholder="haam@example.com" />
                        <Field label="App Password" type="password" value={config.email?.password ?? ''} onChange={v => setEmail('password', v)} placeholder="Gmail App Password"
                            hint="For Gmail: enable 2FA → generate App Password" />
                        <div>
                            <label className="block text-xs font-semibold text-gray-600 mb-1">Recipients (comma-separated)</label>
                            <input
                                type="text"
                                value={(config.email?.recipients ?? []).join(', ')}
                                onChange={e => setEmail('recipients', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                                placeholder="manager@company.com, team@company.com"
                                className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Slack */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="font-bold text-gray-700 flex items-center gap-2">
                        <MessageSquare className="h-5 w-5 text-purple-500" /> Slack Alerts
                    </h2>
                    <Toggle checked={!!config.slack?.enabled} onChange={v => setSlack('enabled', v)} />
                </div>
                {config.slack?.enabled && (
                    <div className="space-y-3 pt-2 border-t border-gray-50">
                        <Field label="Slack Webhook URL" value={config.slack?.webhook_url ?? ''} onChange={v => setSlack('webhook_url', v)}
                            placeholder="https://hooks.slack.com/services/..."
                            hint="Create one at api.slack.com/apps → Incoming Webhooks" />
                    </div>
                )}
            </div>

            {/* Info note */}
            <div className="flex gap-3 bg-blue-50 border border-blue-100 rounded-xl p-4 text-sm text-blue-800">
                <Info className="h-4 w-4 flex-shrink-0 mt-0.5 text-blue-500" />
                <p>Alerts fire automatically when a processed call's agent stress score exceeds the threshold above. Click <strong>Test Alert</strong> to verify your setup before deploying.</p>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
                <button
                    onClick={save}
                    disabled={saving}
                    className="flex-1 py-3 bg-indigo-600 text-white font-bold rounded-xl hover:bg-indigo-700 transition disabled:opacity-50"
                >
                    {saving ? 'Saving…' : '💾 Save Settings'}
                </button>
                <button
                    onClick={testAlert}
                    disabled={testing || (!config.email?.enabled && !config.slack?.enabled)}
                    className="flex items-center gap-2 px-6 py-3 bg-white border border-gray-200 text-gray-700 font-semibold rounded-xl hover:bg-gray-50 transition disabled:opacity-40"
                >
                    <Send className="h-4 w-4" /> {testing ? 'Sending…' : 'Test Alert'}
                </button>
            </div>
        </div>
    );
};

export default AlertsSettingsPage;
