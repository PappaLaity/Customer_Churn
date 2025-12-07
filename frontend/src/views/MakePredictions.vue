<template>
  <AdminLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-2">Make Predictions</h1>
        <p class="text-slate-600">Upload a CSV file to get batch churn predictions</p>
      </div>

      <!-- Upload Section -->
      <div class="bg-white border border-slate-200 rounded-lg p-6 shadow">
        <h2 class="text-xl font-semibold text-slate-900 mb-4">Upload CSV File</h2>

        <!-- File dropzone -->
        <div
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleDrop"
          :class="[
            'border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer',
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'
          ]"
          @click="triggerFileInput"
        >
          <input
            ref="fileInput"
            type="file"
            accept=".csv"
            class="hidden"
            @change="handleFileSelect"
          />
          <svg class="w-12 h-12 mx-auto mb-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
          </svg>
          <p class="text-slate-600 mb-2">
            <span v-if="!selectedFile">Glissez-déposez votre fichier CSV ici, ou cliquez pour parcourir</span>
            <span v-else class="text-green-600 font-medium">{{ selectedFile.name }}</span>
          </p>
          <p class="text-slate-400 text-sm">Fichiers CSV avec les colonnes requises</p>
        </div>

        <!-- Required columns info -->
        <div class="mt-4 p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <p class="text-sm text-slate-600 font-medium mb-2">Colonnes requises:</p>
          <div class="flex flex-wrap gap-2">
            <span v-for="col in requiredColumns" :key="col"
              class="px-2 py-1 bg-white border border-slate-200 text-slate-700 rounded text-xs font-mono">
              {{ col }}
            </span>
          </div>
        </div>

        <!-- Upload button -->
        <button
          @click="uploadAndPredict"
          :disabled="!selectedFile || loading"
          class="mt-4 w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-slate-400 disabled:to-slate-400 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-all flex items-center justify-center gap-2"
        >
          <svg v-if="loading" class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>{{ loading ? 'Traitement en cours...' : 'Upload et Prédire' }}</span>
        </button>

        <!-- Error message -->
        <div v-if="errorMessage" class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p class="text-red-600 text-sm">{{ errorMessage }}</p>
        </div>
      </div>

      <!-- Results Section -->
      <div v-if="predictions" class="bg-white border border-slate-200 rounded-lg p-6 shadow">
        <!-- Summary -->
        <div class="flex items-center justify-between mb-6">
          <div>
            <h2 class="text-xl font-semibold text-slate-900 mb-2">Résultats des Prédictions</h2>
            <p class="text-slate-500 text-sm">Batch ID: {{ predictions.batch_id }}</p>
          </div>
          <button @click="downloadCSV"
            class="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-all">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Télécharger CSV
          </button>
        </div>

        <!-- Stats cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div class="bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl p-4">
            <p class="text-slate-600 text-sm">Total Lignes</p>
            <p class="text-2xl font-bold text-slate-900">{{ predictions.total_rows }}</p>
          </div>
          <div class="bg-gradient-to-br from-red-50 to-red-100 border border-red-200 rounded-xl p-4">
            <p class="text-red-600 text-sm">Vont partir</p>
            <p class="text-2xl font-bold text-red-600">{{ predictions.churn_count }}</p>
          </div>
          <div class="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-xl p-4">
            <p class="text-green-600 text-sm">Vont rester</p>
            <p class="text-2xl font-bold text-green-600">{{ predictions.stay_count }}</p>
          </div>
          <div class="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-4">
            <p class="text-blue-600 text-sm">Probabilité Moy.</p>
            <p class="text-2xl font-bold text-blue-600">{{ (predictions.avg_probability * 100).toFixed(1) }}%</p>
          </div>
        </div>

        <!-- Filter buttons -->
        <div class="flex items-center gap-2 mb-4">
          <span class="text-slate-600 text-sm font-medium mr-2">Filtrer:</span>
          <button @click="filter = 'all'"
            :class="[
              'px-4 py-2 rounded-lg text-sm font-medium transition-all',
              filter === 'all'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            ]">
            Tous ({{ predictions.total_rows }})
          </button>
          <button @click="filter = 'churn'"
            :class="[
              'px-4 py-2 rounded-lg text-sm font-medium transition-all',
              filter === 'churn'
                ? 'bg-red-600 text-white shadow-lg'
                : 'bg-red-50 text-red-600 hover:bg-red-100'
            ]">
            Vont partir ({{ predictions.churn_count }})
          </button>
          <button @click="filter = 'stay'"
            :class="[
              'px-4 py-2 rounded-lg text-sm font-medium transition-all',
              filter === 'stay'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-green-50 text-green-600 hover:bg-green-100'
            ]">
            Vont rester ({{ predictions.stay_count }})
          </button>
        </div>

        <!-- Results table -->
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-slate-200 bg-slate-50">
                <th class="text-left py-3 px-4 text-slate-700 font-semibold">Ligne</th>
                <th class="text-left py-3 px-4 text-slate-700 font-semibold">Prédiction</th>
                <th class="text-left py-3 px-4 text-slate-700 font-semibold">Probabilité</th>
                <th class="text-left py-3 px-4 text-slate-700 font-semibold">Statut</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-200">
              <tr v-for="pred in filteredPredictions" :key="pred.row_index"
                class="hover:bg-slate-50 transition-colors">
                <td class="py-3 px-4 text-slate-700">{{ pred.row_index + 1 }}</td>
                <td class="py-3 px-4">
                  <span :class="pred.prediction === 1 ? 'text-red-600 font-medium' : 'text-green-600 font-medium'">
                    {{ pred.prediction === 1 ? 'Churn' : 'Fidèle' }}
                  </span>
                </td>
                <td class="py-3 px-4">
                  <div class="flex items-center gap-2">
                    <div class="w-20 h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        :class="pred.probability > 0.5 ? 'bg-red-500' : 'bg-green-500'"
                        :style="{ width: (pred.probability * 100) + '%' }"
                        class="h-full transition-all"
                      ></div>
                    </div>
                    <span class="text-slate-600">{{ (pred.probability * 100).toFixed(1) }}%</span>
                  </div>
                </td>
                <td class="py-3 px-4">
                  <span :class="[
                    'px-2 py-1 rounded-full text-xs font-semibold',
                    pred.will_churn ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  ]">
                    {{ pred.will_churn ? '⚠️ À risque' : '✅ Sécurisé' }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Model info -->
        <div class="mt-4 pt-4 border-t border-slate-200 flex items-center justify-between text-sm text-slate-500">
          <span>Version du modèle: {{ predictions.model_version }}</span>
          <span>Latence: {{ predictions.latency_seconds }}s</span>
        </div>
      </div>
    </div>
  </AdminLayout>
</template>

<script>
import AdminLayout from '../components/AdminLayout.vue'
import API_BASE_URL from '@/config/api'

export default {
  name: 'MakePredictions',
  components: {
    AdminLayout
  },
  data () {
    return {
      selectedFile: null,
      isDragging: false,
      loading: false,
      errorMessage: '',
      predictions: null,
      filter: 'all',
      requiredColumns: [
        'tenure',
        'InternetService_Fiber_optic',
        'Contract_Two_year',
        'PaymentMethod_Electronic_check',
        'No_internet_service',
        'TotalCharges',
        'MonthlyCharges',
        'PaperlessBilling'
      ]
    }
  },
  computed: {
    filteredPredictions () {
      if (!this.predictions || !this.predictions.predictions) return []
      if (this.filter === 'all') return this.predictions.predictions
      if (this.filter === 'churn') return this.predictions.predictions.filter(p => p.will_churn === true)
      if (this.filter === 'stay') return this.predictions.predictions.filter(p => p.will_churn === false)
      return this.predictions.predictions
    }
  },
  methods: {
    triggerFileInput () {
      this.$refs.fileInput.click()
    },
    handleFileSelect (event) {
      const file = event.target.files[0]
      if (file && file.name.endsWith('.csv')) {
        this.selectedFile = file
        this.errorMessage = ''
      } else {
        this.errorMessage = 'Veuillez sélectionner un fichier CSV'
      }
    },
    handleDrop (event) {
      this.isDragging = false
      const file = event.dataTransfer.files[0]
      if (file && file.name.endsWith('.csv')) {
        this.selectedFile = file
        this.errorMessage = ''
      } else {
        this.errorMessage = 'Veuillez déposer un fichier CSV'
      }
    },
    async uploadAndPredict () {
      if (!this.selectedFile) return

      this.loading = true
      this.errorMessage = ''
      this.predictions = null

      const apiKey = localStorage.getItem('api-key')
      if (!apiKey) {
        this.errorMessage = 'Authentification requise. Veuillez vous connecter.'
        this.loading = false
        return
      }

      const formData = new FormData()
      formData.append('file', this.selectedFile)

      try {
        const response = await fetch(`${API_BASE_URL}/predict/csv`, {
          method: 'POST',
          headers: {
            'X-API-Key': apiKey
          },
          body: formData
        })

        const data = await response.json()

        if (!response.ok) {
          this.errorMessage = data.detail || 'Erreur lors du traitement du CSV'
          return
        }

        this.predictions = data
      } catch (err) {
        console.error(err)
        this.errorMessage = 'Impossible de se connecter au serveur'
      } finally {
        this.loading = false
      }
    },
    downloadCSV () {
      if (!this.predictions) return

      // Generate filename with current date and time
      const now = new Date()
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hours = String(now.getHours()).padStart(2, '0')
      const minutes = String(now.getMinutes()).padStart(2, '0')
      const seconds = String(now.getSeconds()).padStart(2, '0')
      const filename = 'prediction_' + year + '-' + month + '-' + day + '_' + hours + '-' + minutes + '-' + seconds + '.csv'

      // Build CSV content
      const lines = []
      lines.push('Ligne;Prediction;Probabilite;Va_Partir')

      for (let i = 0; i < this.predictions.predictions.length; i++) {
        const p = this.predictions.predictions[i]
        const ligne = p.row_index + 1
        const prediction = p.prediction === 1 ? 'Churn' : 'Fidele'
        const probabilite = (p.probability * 100).toFixed(2) + '%'
        const vaPartir = p.will_churn ? 'Oui' : 'Non'
        lines.push(ligne + ';' + prediction + ';' + probabilite + ';' + vaPartir)
      }

      const csvContent = lines.join('\r\n')

      // Create download using data URI
      const encodedUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent)
      const link = document.createElement('a')
      link.setAttribute('href', encodedUri)
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }
}
</script>
