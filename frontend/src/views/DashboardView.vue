<template>
  <AdminLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">Dashboard</h1>
        <p class="text-slate-600">Vue d'ensemble du système et des modèles</p>
      </div>

      <!-- État de chargement -->
      <div v-if="loading" class="flex items-center justify-center py-20">
        <div class="text-center">
          <div class="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p class="text-slate-600 font-medium">Chargement du dashboard...</p>
        </div>
      </div>

      <!-- Message d'erreur -->
      <div v-else-if="errorMessage" class="p-6 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-4">
        <svg class="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
        </svg>
        <div>
          <p class="font-semibold text-red-700">Erreur</p>
          <p class="text-red-600">{{ errorMessage }}</p>
        </div>
      </div>

      <div v-else class="space-y-6">
        <!-- Model Versions Section -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-white border border-slate-200 rounded-lg p-6 shadow hover:shadow-lg transition-shadow">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-semibold text-slate-900">Production Model</h3>
              <div class="w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-lg flex items-center justify-center">
                <svg class="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 3.062v6.372a3.066 3.066 0 01-2.812 3.062 3.066 3.066 0 01-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 01-1.745-.723 3.066 3.066 0 01-2.812-3.062V6.517a3.066 3.066 0 012.812-3.062zm7.958 5.529a.75.75 0 00-1.214-.882l-3.083 4.409-1.596-1.591a.75.75 0 10-1.06 1.061l2.12 2.121a.75.75 0 001.137-.089l3.696-5.27z" clip-rule="evenodd"></path>
                </svg>
              </div>
            </div>
            <div class="space-y-3">
              <div class="flex justify-between items-center">
                <span class="text-slate-600 text-sm font-medium">Version</span>
                <span class="text-2xl font-bold text-green-600">{{ modelVersions.production_model_version || 'N/A' }}</span>
              </div>
              <div class="pt-2 border-t border-slate-200">
                <p class="text-xs text-slate-500 uppercase mb-2">Status</p>
                <span class="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full inline-block">
                  ✓ Actif
                </span>
              </div>
            </div>
          </div>

          <div class="bg-white border border-slate-200 rounded-lg p-6 shadow hover:shadow-lg transition-shadow">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-semibold text-slate-900">Staging Model</h3>
              <div class="w-12 h-12 bg-gradient-to-br from-orange-400 to-amber-500 rounded-lg flex items-center justify-center">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
            </div>
            <div class="space-y-3">
              <div class="flex justify-between items-center">
                <span class="text-slate-600 text-sm font-medium">Version</span>
                <span class="text-2xl font-bold text-orange-600">{{ modelVersions.staging_model_version || 'N/A' }}</span>
              </div>
              <div class="pt-2 border-t border-slate-200">
                <p class="text-xs text-slate-500 uppercase mb-2">Status</p>
                <span v-if="modelVersions.staging_model_version" class="px-3 py-1 bg-orange-100 text-orange-700 text-xs font-semibold rounded-full inline-block">
                  ⚙ En test
                </span>
                <span v-else class="px-3 py-1 bg-slate-100 text-slate-700 text-xs font-semibold rounded-full inline-block">
                  ✗ Aucun modèle
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Models Section -->
        <div class="bg-white border border-slate-200 rounded-lg p-6 shadow">
          <h2 class="text-2xl font-bold text-slate-900 mb-6">Modèles Machine Learning</h2>

          <div v-if="models.length === 0" class="flex flex-col items-center justify-center py-12">
            <svg class="w-16 h-16 text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5h.01M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
            </svg>
            <p class="text-slate-600">Aucun modèle trouvé</p>
          </div>

          <div v-else class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div v-for="model in models" :key="model.version" class="border border-slate-200 rounded-lg p-4 hover:shadow-md transition-shadow">
              <div class="flex items-start justify-between mb-4">
                <div>
                  <h3 class="text-lg font-bold text-slate-900">{{ model.model_name }}</h3>
                  <p class="text-sm text-slate-500">Version {{ model.version }}</p>
                </div>
                <span :class="[
                  'px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap',
                  model.current_stage === 'Production'
                    ? 'bg-green-100 text-green-700'
                    : model.current_stage === 'Staging'
                    ? 'bg-orange-100 text-orange-700'
                    : 'bg-slate-100 text-slate-700'
                ]">
                  {{ model.current_stage }}
                </span>
              </div>

              <p class="text-sm text-slate-600 mb-4">{{ model.description }}</p>

              <div class="grid grid-cols-2 gap-4 py-4 border-y border-slate-200">
                <div>
                  <p class="text-xs text-slate-500 uppercase font-semibold mb-1">CV Mean</p>
                  <p class="text-xl font-bold text-blue-600">{{ (parseFloat(model.cv_mean) * 100).toFixed(2) }}%</p>
                </div>
                <div>
                  <p class="text-xs text-slate-500 uppercase font-semibold mb-1">Test Accuracy</p>
                  <p class="text-xl font-bold text-green-600">{{ (parseFloat(model.test_accuracy) * 100).toFixed(2) }}%</p>
                </div>
              </div>

              <div class="grid grid-cols-2 gap-4 mt-4 text-xs text-slate-600">
                <div>
                  <p class="font-semibold text-slate-700 mb-1">Créé</p>
                  <p>{{ formatDate(model.creation_timestamp) }}</p>
                </div>
                <div>
                  <p class="font-semibold text-slate-700 mb-1">Mis à jour</p>
                  <p>{{ formatDate(model.last_updated_timestamp) }}</p>
                </div>
              </div>

              <div class="mt-4 pt-4 border-t border-slate-200">
                <p class="text-xs text-slate-500 break-all">Run ID: {{ model.run_id }}</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Metrics Section -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-lg p-4">
            <p class="text-slate-600 text-sm font-medium mb-1">Total Requêtes</p>
            <p class="text-3xl font-bold text-blue-600">{{ metrics.total_requests }}</p>
            <p class="text-xs text-slate-500 mt-2">{{ metrics.success_rate }}% de succès</p>
          </div>

          <div class="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4">
            <p class="text-slate-600 text-sm font-medium mb-1">Requêtes Réussies</p>
            <p class="text-3xl font-bold text-green-600">{{ metrics.successful_requests }}</p>
            <p class="text-xs text-slate-500 mt-2">2xx responses</p>
          </div>

          <div class="bg-gradient-to-br from-orange-50 to-amber-50 border border-orange-200 rounded-lg p-4">
            <p class="text-slate-600 text-sm font-medium mb-1">Erreurs Client</p>
            <p class="text-3xl font-bold text-orange-600">{{ metrics.client_errors }}</p>
            <p class="text-xs text-slate-500 mt-2">4xx responses</p>
          </div>

          <div class="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
            <p class="text-slate-600 text-sm font-medium mb-1">Mémoire Utilisée</p>
            <p class="text-3xl font-bold text-purple-600">{{ metrics.memory_used }}</p>
            <p class="text-xs text-slate-500 mt-2">{{ metrics.memory_percent }}%</p>
          </div>
        </div>

        <!-- Endpoints Stats -->
        <div class="bg-white border border-slate-200 rounded-lg p-6 shadow">
          <h2 class="text-2xl font-bold text-slate-900 mb-6">Statistiques par Endpoint</h2>

          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-slate-200 bg-slate-50">
                  <th class="px-4 py-3 text-left font-semibold text-slate-700">Endpoint</th>
                  <th class="px-4 py-3 text-left font-semibold text-slate-700">Méthode</th>
                  <th class="px-4 py-3 text-center font-semibold text-slate-700">Requêtes</th>
                  <th class="px-4 py-3 text-center font-semibold text-slate-700">Succès</th>
                  <th class="px-4 py-3 text-center font-semibold text-slate-700">Erreurs</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-200">
                <tr v-for="(stat, idx) in endpointStats" :key="idx" class="hover:bg-slate-50 transition-colors">
                  <td class="px-4 py-3 font-medium text-slate-900">{{ stat.handler }}</td>
                  <td class="px-4 py-3">
                    <span :class="[
                      'px-2 py-1 rounded text-xs font-semibold',
                      stat.method === 'GET' ? 'bg-blue-100 text-blue-700'
                        : stat.method === 'POST' ? 'bg-green-100 text-green-700'
                        : stat.method === 'PUT' ? 'bg-orange-100 text-orange-700'
                        : stat.method === 'DELETE' ? 'bg-red-100 text-red-700'
                        : 'bg-slate-100 text-slate-700'
                    ]">
                      {{ stat.method }}
                    </span>
                  </td>
                  <td class="px-4 py-3 text-center font-semibold text-slate-700">{{ stat.total }}</td>
                  <td class="px-4 py-3 text-center">
                    <span class="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-semibold">{{ stat.success }}</span>
                  </td>
                  <td class="px-4 py-3 text-center">
                    <span class="px-2 py-1 bg-red-100 text-red-700 rounded text-xs font-semibold">{{ stat.errors }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </AdminLayout>
</template>

<script setup>
import AdminLayout from '../components/AdminLayout.vue'
import API_BASE_URL from '@/config/api'
import { ref, onMounted, computed } from 'vue'

const loading = ref(true)
const errorMessage = ref('')
const modelVersions = ref({ production_model_version: null, staging_model_version: null })
const models = ref([])
const metricsData = ref(null)

const metrics = computed(() => {
  if (!metricsData.value) {
    return {
      total_requests: 0,
      successful_requests: 0,
      client_errors: 0,
      memory_used: '0 MB',
      memory_percent: 0,
      success_rate: 0
    }
  }

  const text = metricsData.value

  // Parse metrics
  const getTotalRequests = () => {
    const match = text.match(/http_requests_total\{.*?status="2xx"\}.*?(\d+(?:\.\d+)?)/g)
    const total2xx = match ? match.reduce((sum, m) => sum + parseFloat(m.match(/\d+(?:\.\d+)$/)[0]), 0) : 0
    const match4xx = text.match(/http_requests_total\{.*?status="4xx"\}.*?(\d+(?:\.\d+)?)/g)
    const total4xx = match4xx ? match4xx.reduce((sum, m) => sum + parseFloat(m.match(/\d+(?:\.\d+)$/)[0]), 0) : 0
    return { total2xx, total4xx, totalAll: total2xx + total4xx }
  }

  const { total2xx, total4xx, totalAll } = getTotalRequests()

  const memoryMatch = text.match(/process_resident_memory_bytes\s+([\d.e+]+)/)
  const memoryBytes = memoryMatch ? parseFloat(memoryMatch[1]) : 0
  const memoryMB = (memoryBytes / (1024 * 1024)).toFixed(1)
  const memoryPercent = ((memoryBytes / (2.622e9)) * 100).toFixed(1)

  return {
    total_requests: Math.round(totalAll),
    successful_requests: Math.round(total2xx),
    client_errors: Math.round(total4xx),
    memory_used: `${memoryMB} MB`,
    memory_percent: memoryPercent,
    success_rate: totalAll > 0 ? ((total2xx / totalAll) * 100).toFixed(1) : 0
  }
})

const endpointStats = computed(() => {
  if (!metricsData.value) return []

  const text = metricsData.value
  const stats = {}

  // Parse each endpoint
  const regex = /http_requests_total\{handler="([^"]+)",method="([^"]+)",status="([^"]+)"\}\s+([\d.]+)/g
  let match

  while ((match = regex.exec(text)) !== null) {
    const [, handler, method, status, count] = match
    const key = `${handler}|${method}`

    if (!stats[key]) {
      stats[key] = { handler, method, success: 0, errors: 0 }
    }

    if (status === '2xx') {
      stats[key].success = parseInt(count)
    } else if (status === '4xx') {
      stats[key].errors = parseInt(count)
    }
  }

  return Object.values(stats)
    .map(s => ({ ...s, total: s.success + s.errors }))
    .sort((a, b) => b.total - a.total)
})

const formatDate = (timestamp) => {
  return new Date(timestamp).toLocaleDateString('fr-FR', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

onMounted(async () => {
  const apiKey = localStorage.getItem('api-key')

  if (!apiKey) {
    errorMessage.value = 'Pas de clé API. Veuillez vous connecter.'
    loading.value = false
    return
  }

  try {
    // Fetch model versions
    const versionRes = await fetch(`${API_BASE_URL}/model/version`, {
      headers: { 'x-api-key': apiKey }
    })
    const versionData = await versionRes.json()
    modelVersions.value = versionData

    // Fetch models
    const modelsRes = await fetch(`${API_BASE_URL}/models`, {
      headers: { 'x-api-key': apiKey }
    })
    const modelsData = await modelsRes.json()
    models.value = modelsData.models || []

    // Fetch metrics
    const metricsRes = await fetch(`${API_BASE_URL}/metrics`, {
      headers: { 'x-api-key': apiKey }
    })
    metricsData.value = await metricsRes.text()
  } catch (err) {
    console.error(err)
    errorMessage.value = 'Erreur lors du chargement des données'
  } finally {
    loading.value = false
  }
})
</script>
