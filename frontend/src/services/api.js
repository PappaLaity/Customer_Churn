// Optimized API client for frontend
import axios from 'axios'

const API_BASE_URL = process.env.VUE_APP_API_URL || 'http://localhost:8000'

// Create axios instance with defaults
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 second timeout
    headers: {
        'Content-Type': 'application/json'
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

// API methods
export default {
    // Models
    async getModelVersions() {
        const response = await apiClient.get('/model/version')
        return response.data
    },

    async getModels() {
        const response = await apiClient.get('/models')
        return response.data
    },

    // Metrics
    async getMetrics() {
        const response = await apiClient.get('/metrics')
        return response.data
    },

    // Predictions
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

    // A/B Testing
    async getABConfig() {
        const response = await apiClient.get('/ab/config')
        return response.data
    },

    async getABResults(metric = 'latency') {
        const response = await apiClient.get(`/ab/results?metric=${metric}`)
        return response.data
    },

    // Health check
    async healthCheck() {
        const response = await apiClient.get('/health')
        return response.data
    },

    // Parallel dashboard data loading (FAST!)
    async getDashboardData() {
        const [versions, models, metrics] = await Promise.all([
            this.getModelVersions(),
            this.getModels(),
            apiClient.get('/metrics').then(r => r.data) // Returns text
        ])

        return {
            versions,
            models,
            metrics
        }
    }
}
