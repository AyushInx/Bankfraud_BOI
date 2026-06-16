// src/hooks/useApi.js — Axios API client and React hooks

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Generic fetch hook ──────────────────────────────────────────────────────
export function useFetch(url, params = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    if (!url) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(url, { params });
      setData(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => { refetch(); }, [refetch]);
  return { data, loading, error, refetch };
}

// ── Dashboard stats ─────────────────────────────────────────────────────────
export function useDashboardStats() {
  return useFetch('/api/v1/stats');
}

// ── Model metrics ───────────────────────────────────────────────────────────
export function useModelMetrics() {
  return useFetch('/api/v1/metrics');
}

// ── Feature importance ──────────────────────────────────────────────────────
export function useFeatureImportance() {
  return useFetch('/api/v1/features');
}

// ── Health ──────────────────────────────────────────────────────────────────
export function useHealth() {
  return useFetch('/api/v1/health');
}

// ── Accounts list ───────────────────────────────────────────────────────────
export function useAccounts(page = 1, pageSize = 20, label = null) {
  const params = { page, page_size: pageSize };
  if (label) params.label = label;
  return useFetch('/api/v1/accounts', params);
}

// ── Single account ──────────────────────────────────────────────────────────
export function useAccount(accountId) {
  return useFetch(accountId ? `/api/v1/accounts/${accountId}` : null);
}

// ── Alerts ──────────────────────────────────────────────────────────────────
export function useAlerts(status = null, severity = null) {
  const params = { limit: 100 };
  if (status) params.status = status;
  if (severity) params.severity = severity;
  return useFetch('/api/v1/alerts', params);
}

// ── Prediction API calls ────────────────────────────────────────────────────
export async function predictSingle(features, accountRef = null) {
  const res = await api.post('/api/v1/predict', {
    features,
    account_ref: accountRef,
  });
  return res.data;
}

export async function uploadBatch(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/api/v1/batch-predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

export async function getBatchStatus(jobId) {
  const res = await api.get(`/api/v1/batch/${jobId}`);
  return res.data;
}

export async function updateAlertStatus(alertId, status, notes = null) {
  const res = await api.patch(`/api/v1/alerts/${alertId}/status`, { status, notes });
  return res.data;
}
