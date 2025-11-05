<!-- <template>
    <AdminLayout>
        <div class="about">
            <h1 class="text-2xl font-bold mb-4">Customer Infos</h1>

            <div v-if="loading" class="text-gray-500">Data loading...</div>
            <div v-else-if="errorMessage" class="text-red-500">{{ errorMessage }}</div>
            <div v-else-if="emptyData" class="text-black">{{ emptyData }}</div>

            <table v-else class="min-w-full border border-gray-200">
                <thead class="bg-gray-100">
                    <tr>
                        <th v-for="col in columns" :key="col" class="border px-4 py-2 text-left">
                            {{ col }}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(customer, index) in customers" :key="index">
                        <td v-for="col in columns" :key="col" class="border px-2 py-2">
                            {{ customer[col] }}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </AdminLayout>
</template>

<script setup>
import AdminLayout from '../components/AdminLayout.vue'
import { ref, onMounted } from 'vue'

const customers = ref([])
const columns = ref([])
const loading = ref(true)
const errorMessage = ref('')
const emptyData = ref('')

onMounted(async () => {
  const apiKey = localStorage.getItem('api-key')
  if (!apiKey) {
    errorMessage.value = 'Pas de clé API. Veuillez vous connecter.'
    loading.value = false
    return
  }

  try {
    const response = await fetch('http://localhost:8000/customers/infos', {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey
      }
    })

    if (!response.ok) {
      const errData = await response.json()
      errorMessage.value = errData.message || 'Erreur lors de la récupération des données.'
      loading.value = false
      // console.log('API response not ok:', errData)
      return
    }

    const data = await response.json()
    // customers.value = data.data
    // console.log('Fetched customer data:', data)

    // Détecte dynamiquement les colonnes depuis le premier élément
    if (data.count > 0) {
      columns.value = data.columns
      customers.value = data.data
    //   columns.value = json.columns
    } else {
      emptyData.value = 'No data available.'
    }
  } catch (err) {
    console.error(err)
    errorMessage.value = 'Impossible de contacter le serveur.'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
table th,
table td {
    text-align: left;
}
</style> -->
<template>
    <AdminLayout>
        <div class="about">
            <!-- Header -->
            <div class="mb-6">
                <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">Customer Infos</h1>
                <p class="text-slate-500">{{ customers.length }} clients trouvés</p>
            </div>

            <!-- États de chargement -->
            <div v-if="loading" class="flex items-center justify-center py-12">
                <div class="text-center">
                    <div class="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-3"></div>
                    <p class="text-slate-600">Chargement des données...</p>
                </div>
            </div>

            <div v-else-if="errorMessage" class="p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-3">
                <svg class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                </svg>
                <div>
                    <p class="font-semibold text-red-700">Erreur</p>
                    <p class="text-red-600 text-sm">{{ errorMessage }}</p>
                </div>
            </div>

            <div v-else-if="emptyData" class="flex flex-col items-center justify-center py-12">
                <svg class="w-16 h-16 text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path>
                </svg>
                <p class="text-slate-600">{{ emptyData }}</p>
            </div>

            <!-- Table responsive -->
            <div v-else class="space-y-4">
                <!-- Filtres et options -->
                <div class="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
                    <div class="w-full md:w-64">
                        <input
                            v-model="searchQuery"
                            type="text"
                            placeholder="Rechercher..."
                            class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                        />
                    </div>
                    <div class="flex gap-2 flex-wrap">
                        <button
                            v-for="option in filterOptions"
                            :key="option"
                            @click="filterChurn = option"
                            :class="[
                                'px-4 py-2 rounded-lg font-medium transition-all',
                                filterChurn === option
                                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                            ]"
                        >
                            {{ option }}
                        </button>
                    </div>
                </div>

                <!-- Desktop Table -->
                <div class="hidden lg:block overflow-hidden rounded-lg border border-slate-200 shadow-lg">
                    <table class="w-full">
                        <thead>
                            <tr class="bg-gradient-to-r from-slate-800 to-slate-700 text-white">
                                <th v-for="col in columns" :key="col" class="px-6 py-3 text-left text-sm font-semibold">
                                    <div class="flex items-center gap-2 cursor-pointer hover:text-blue-300" @click="sortBy(col)">
                                        {{ formatHeader(col) }}
                                        <svg v-if="sortColumn === col" class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                            <path v-if="sortOrder === 'asc'" d="M3 9a1 1 0 011-1h12a1 1 0 011 1v1a1 1 0 01-1 1H4a1 1 0 01-1-1V9z"></path>
                                            <path v-else d="M3 11a1 1 0 011 1v1a1 1 0 01-1 1H1a1 1 0 01-1-1v-1a1 1 0 011-1h2z"></path>
                                        </svg>
                                    </div>
                                </th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-200">
                            <tr v-for="(customer, index) in filteredCustomers" :key="index" class="hover:bg-slate-50 transition-colors">
                                <td v-for="col in columns" :key="col" class="px-6 py-4 text-sm text-slate-700">
                                    <div v-if="col.toLowerCase() === 'churn'">
                                        <span :class="[
                                            'px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1',
                                            customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true
                                                ? 'bg-red-100 text-red-700'
                                                : 'bg-green-100 text-green-700'
                                        ]">
                                            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                                                <path v-if="customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true" fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                                <path v-else fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            {{ customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true ? 'Yes' : 'No' }}
                                        </span>
                                    </div>
                                    <div v-else>
                                        {{ truncate(customer[col], 50) }}
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Mobile/Tablet Cards -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:hidden gap-4">
                    <div v-for="(customer, index) in filteredCustomers" :key="index" class="bg-white border border-slate-200 rounded-lg p-4 shadow hover:shadow-lg transition-shadow">
                        <div v-for="col in columns" :key="col" class="mb-3 pb-3 border-b border-slate-100 last:border-b-0">
                            <p class="text-xs font-semibold text-slate-500 uppercase mb-1">{{ formatHeader(col) }}</p>
                            <div v-if="col.toLowerCase() === 'churn'">
                                <span :class="[
                                    'px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1',
                                    customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true
                                        ? 'bg-red-100 text-red-700'
                                        : 'bg-green-100 text-green-700'
                                ]">
                                    <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                                        <path v-if="customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true" fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                                        <path v-else fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                    </svg>
                                    {{ customer[col] === 'Yes' || customer[col] === 1 || customer[col] === true ? 'Yes' : 'No' }}
                                </span>
                            </div>
                            <p v-else class="text-slate-700">{{ truncate(customer[col], 50) }}</p>
                        </div>
                    </div>
                </div>

                <!-- Pagination -->
                <div v-if="filteredCustomers.length > 0" class="flex items-center justify-between mt-6 pt-4 border-t border-slate-200">
                    <p class="text-sm text-slate-600">
                        Affichage de <span class="font-semibold">{{ filteredCustomers.length }}</span> clients
                    </p>
                </div>
            </div>
        </div>
    </AdminLayout>
</template>

<script setup>
import AdminLayout from '../components/AdminLayout.vue'
import API_BASE_URL from '@/config/api'
import { ref, onMounted, computed } from 'vue'

const customers = ref([])
const columns = ref([])
const loading = ref(true)
const errorMessage = ref('')
const emptyData = ref('')
const searchQuery = ref('')
const filterChurn = ref('All')
const sortColumn = ref(null)
const sortOrder = ref('asc')
const filterOptions = ref(['All', 'Yes', 'No'])

onMounted(async () => {
  const apiKey = localStorage.getItem('api-key')
  if (!apiKey) {
    errorMessage.value = 'Pas de clé API. Veuillez vous connecter.'
    loading.value = false
    return
  }

  try {
    const response = await fetch(`${API_BASE_URL}/customers/infos`, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey
      }
    })

    if (!response.ok) {
      const errData = await response.json()
      errorMessage.value = errData.message || 'Erreur lors de la récupération des données.'
      loading.value = false
      return
    }

    const data = await response.json()

    if (data.count > 0) {
      columns.value = data.columns
      customers.value = data.data
    } else {
      emptyData.value = 'Aucune donnée disponible.'
    }
  } catch (err) {
    console.error(err)
    errorMessage.value = 'Impossible de contacter le serveur.'
  } finally {
    loading.value = false
  }
})

const filteredCustomers = computed(() => {
  let filtered = customers.value

  // Filtre par recherche
  if (searchQuery.value) {
    filtered = filtered.filter(customer =>
      JSON.stringify(customer).toLowerCase().includes(searchQuery.value.toLowerCase())
    )
  }

  // Filtre par Churn
  if (filterChurn.value !== 'All') {
    const churnValue = filterChurn.value === 'Yes'
    filtered = filtered.filter(customer => {
      const val = customer.Churn
      return (val === 'Yes' || val === 1 || val === true) === churnValue
    })
  }

  // Tri
  if (sortColumn.value) {
    filtered.sort((a, b) => {
      const aVal = a[sortColumn.value]
      const bVal = b[sortColumn.value]

      if (typeof aVal === 'string') {
        return sortOrder.value === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      } else {
        return sortOrder.value === 'asc' ? aVal - bVal : bVal - aVal
      }
    })
  }

  return filtered
})

const sortBy = (col) => {
  if (sortColumn.value === col) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortColumn.value = col
    sortOrder.value = 'asc'
  }
}

const formatHeader = (header) => {
  return header
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase())
    .trim()
}

const truncate = (str, length) => {
  if (typeof str !== 'string') return str
  return str.length > length ? str.substring(0, length) + '...' : str
}
</script>

<style scoped>
table {
    border-collapse: collapse;
}
</style>
