// Optimized API client for frontend
import axios from 'axios'

const API_BASE_URL = process.env.VUE_APP_API_URL || 'http://localhost:8000'

// Simple cache implementation
const cache = new Map()
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes

function getCached(key) {
    const cached = cache.get(key)
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.data
    }
    cache.delete(key)
    return null
}

function setCache(key, data) {
    cache.set(key, { data, timestamp: Date.now() })
}

// Create axios instance with defaults
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 second timeout
    headers: {
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br' // Request compression
    }
})

// Request interceptor: Add API key automatically
apiClient.interceptors.request.use(
    (config) => {
        const apiKey = localStorage.getItem('api-key')
        if (apiKey) {
            config.headers['X-API-Key'] = apiKey
        }
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)

// Response interceptor: Handle errors globally
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response) {
            // Server responded with error
            if (error.response.status === 403) {
                // Invalid API key - redirect to login
                localStorage.removeItem('api-key')
                window.location.href = '/login'
            }
        } else if (error.request) {
            // Request made but no response
            console.error('No response from server')
        }
        return Promise.reject(error)
    }
)

// API methods with caching
export default {
    // Models (cacheable)
    async getModelVersions() {
        const cacheKey = 'model-versions'
        const cached = getCached(cacheKey)
        if (cached) return cached

        const response = await apiClient.get('/model/version')
        setCache(cacheKey, response.data)
        return response.data
    },

    async getModels() {
        const cacheKey = 'models'
        const cached = getCached(cacheKey)
        if (cached) return cached

        const response = await apiClient.get('/models')
        setCache(cacheKey, response.data)
        return response.data
    },

    // Metrics (cacheable for 1 minute)
    async getMetrics() {
        const response = await apiClient.get('/metrics')
        return response.data
    },

    // Predictions (not cacheable - always fresh)
    async makePrediction(instances, returnProba = false) {
        const response = await apiClient.post('/predict', {
            instances,
            return_proba: returnProba
        })
        return response.data
    },

    async submitSurvey(customerData) {
        const response = await apiClient.post('/survey/submit', customerData)
        return response.data
    },

    // A/B Testing (cacheable)
    async getABConfig() {
        const cacheKey = 'ab-config'
        const cached = getCached(cacheKey)
        if (cached) return cached

        const response = await apiClient.get('/ab/config')
        setCache(cacheKey, response.data)
        return response.data
    },

    async getABResults(metric = 'latency') {
        const cacheKey = `ab-results-${metric}`
        const cached = getCached(cacheKey)
        if (cached) return cached

        const response = await apiClient.get(`/ab/results?metric=${metric}`)
        setCache(cacheKey, response.data)
        return response.data
    },

    // Health check (no cache)
    async healthCheck() {
        const response = await apiClient.get('/health')
        return response.data
    },

    // Parallel dashboard data loading (FAST! + Cached)
    async getDashboardData() {
        const [versions, models, metrics] = await Promise.all([
            this.getModelVersions(), // Cached
            this.getModels(),         // Cached
            apiClient.get('/metrics').then(r => r.data) // Fresh
        ])

        return {
            versions,
            models,
            metrics
        }
    },

    // Clear cache utility
    clearCache() {
        cache.clear()
    }
}

