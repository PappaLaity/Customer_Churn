<template>
    <AdminLayout>
        <div class="space-y-6">
            <!-- Header avec bouton -->
            <div class="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                    <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">Utilisateurs</h1>
                    <p class="text-slate-600">Gérez les utilisateurs du système</p>
                </div>
                <button
                    @click="showCreateModal = true"
                    class="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold rounded-lg transition-all duration-200 shadow-lg shadow-green-500/50 hover:shadow-green-500/75 h-fit whitespace-nowrap"
                >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
                    </svg>
                    Créer un utilisateur
                </button>
            </div>

            <!-- État de chargement -->
            <div v-if="loading" class="flex items-center justify-center py-20">
                <div class="text-center">
                    <div class="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
                    <p class="text-slate-600 font-medium">Chargement des utilisateurs...</p>
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

            <!-- État vide -->
            <div v-else-if="users.length === 0" class="flex flex-col items-center justify-center py-20">
                <svg class="w-20 h-20 text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.856-1.487M15 6a3 3 0 11-6 0 3 3 0 016 0zM6 20a9 9 0 0118 0v2H6v-2z"></path>
                </svg>
                <p class="text-slate-600 font-medium">Aucun utilisateur trouvé</p>
            </div>

            <!-- Tableau des utilisateurs -->
            <div v-else class="space-y-4">
                <!-- Desktop Table -->
                <div class="hidden lg:block overflow-hidden rounded-lg border border-slate-200 shadow-lg">
                    <table class="w-full">
                        <thead>
                            <tr class="bg-gradient-to-r from-slate-800 to-slate-700 text-white">
                                <th class="px-6 py-4 text-left text-sm font-semibold">ID</th>
                                <th class="px-6 py-4 text-left text-sm font-semibold">Nom d'utilisateur</th>
                                <th class="px-6 py-4 text-left text-sm font-semibold">Email</th>
                                <th class="px-6 py-4 text-left text-sm font-semibold">Téléphone</th>
                                <th class="px-6 py-4 text-left text-sm font-semibold">Rôle</th>
                                <th class="px-6 py-4 text-left text-sm font-semibold">Actions</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-200">
                            <tr v-for="user in users" :key="user.id" class="hover:bg-slate-50 transition-colors">
                                <td class="px-6 py-4 text-sm text-slate-700 font-medium">{{ user.id }}</td>
                                <td class="px-6 py-4 text-sm">
                                    <div class="flex items-center gap-3">
                                        <div class="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                                            {{ user.username.charAt(0).toUpperCase() }}
                                        </div>
                                        <div>
                                            <span class="text-slate-700 font-medium">{{ user.username }}</span>
                                            <span v-if="user.id === currentUserId" class="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-semibold rounded">Vous</span>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 text-sm text-slate-600">{{ user.email }}</td>
                                <td class="px-6 py-4 text-sm text-slate-600">{{ user.phone }}</td>
                                <td class="px-6 py-4 text-sm">
                                    <span :class="[
                                        'px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1',
                                        user.role === 'admin'
                                            ? 'bg-purple-100 text-purple-700'
                                            : user.role === 'manager'
                                            ? 'bg-orange-100 text-orange-700'
                                            : user.role === 'supervisor'
                                            ? 'bg-cyan-100 text-cyan-700'
                                            : 'bg-slate-100 text-slate-700'
                                    ]">
                                        <svg v-if="user.role === 'admin'" class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5.951-1.488 5.953 1.488a1 1 0 001.168-1.409l-7-14z"></path>
                                        </svg>
                                        {{ user.role }}
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-sm">
                                    <div class="flex gap-2">
                                        <button
                                            v-if="user.id === currentUserId"
                                            @click="openEditModal(user)"
                                            class="p-2 text-blue-600 hover:bg-blue-100 rounded transition-colors"
                                            title="Modifier mon profil"
                                        >
                                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
                                            </svg>
                                        </button>
                                        <!-- <button
                                            v-if="currentUserId !== user.id"
                                            @click="deleteUser(user.id)"
                                            class="p-2 text-red-600 hover:bg-red-100 rounded transition-colors"
                                            title="Supprimer cet utilisateur"
                                        >
                                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                                            </svg>
                                        </button> -->
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Mobile/Tablet Cards -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:hidden gap-4">
                    <div v-for="user in users" :key="user.id" class="bg-white border border-slate-200 rounded-lg p-4 shadow hover:shadow-lg transition-shadow">
                        <div class="flex items-center justify-between mb-4">
                            <div class="flex items-center gap-3">
                                <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-full flex items-center justify-center text-white font-semibold">
                                    {{ user.username.charAt(0).toUpperCase() }}
                                </div>
                                <div>
                                    <p class="font-semibold text-slate-900">{{ user.username }}</p>
                                    <p class="text-xs text-slate-500">ID: {{ user.id }}</p>
                                </div>
                            </div>
                            <span v-if="user.id === currentUserId" class="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded">Vous</span>
                        </div>

                        <div class="space-y-3 mb-4">
                            <div class="pb-3 border-b border-slate-200">
                                <p class="text-xs font-semibold text-slate-500 uppercase mb-1">Email</p>
                                <p class="text-slate-700 text-sm">{{ user.email }}</p>
                            </div>

                            <div class="pb-3 border-b border-slate-200">
                                <p class="text-xs font-semibold text-slate-500 uppercase mb-1">Téléphone</p>
                                <p class="text-slate-700 text-sm">{{ user.phone }}</p>
                            </div>

                            <div>
                                <p class="text-xs font-semibold text-slate-500 uppercase mb-1">Rôle</p>
                                <span :class="[
                                    'px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1',
                                    user.role === 'admin'
                                        ? 'bg-purple-100 text-purple-700'
                                        : user.role === 'manager'
                                        ? 'bg-orange-100 text-orange-700'
                                        : user.role === 'supervisor'
                                        ? 'bg-cyan-100 text-cyan-700'
                                        : 'bg-slate-100 text-slate-700'
                                ]">
                                    {{ user.role }}
                                </span>
                            </div>
                        </div>

                        <div class="flex gap-2">
                            <button
                                v-if="user.id === currentUserId"
                                @click="openEditModal(user)"
                                class="flex-1 p-2 text-blue-600 border border-blue-300 hover:bg-blue-50 rounded font-medium text-sm transition-colors flex items-center justify-center gap-1"
                            >
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
                                </svg>
                                Modifier
                            </button>
                            <!-- <button
                                v-if="currentUserId !== user.id"
                                @click="deleteUser(user.id)"
                                class="flex-1 p-2 text-red-600 border border-red-300 hover:bg-red-50 rounded font-medium text-sm transition-colors flex items-center justify-center gap-1"
                            >
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                                </svg>
                                Supprimer
                            </button> -->
                        </div>
                    </div>
                </div>

                <!-- Statistiques -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 pt-6 border-t border-slate-200">
                    <div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-lg p-4">
                        <p class="text-slate-600 text-sm font-medium mb-1">Total Utilisateurs</p>
                        <p class="text-3xl font-bold text-blue-600">{{ users.length }}</p>
                    </div>

                    <div class="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
                        <p class="text-slate-600 text-sm font-medium mb-1">Administrateurs</p>
                        <p class="text-3xl font-bold text-purple-600">{{ adminCount }}</p>
                    </div>

                    <div class="bg-gradient-to-br from-orange-50 to-amber-50 border border-orange-200 rounded-lg p-4">
                        <p class="text-slate-600 text-sm font-medium mb-1">Managers</p>
                        <p class="text-3xl font-bold text-orange-600">{{ managerCount }}</p>
                    </div>

                    <div class="bg-gradient-to-br from-cyan-50 to-teal-50 border border-cyan-200 rounded-lg p-4">
                        <p class="text-slate-600 text-sm font-medium mb-1">Superviseurs</p>
                        <p class="text-3xl font-bold text-cyan-600">{{ supervisorCount }}</p>
                    </div>
                </div>
            </div>

            <!-- Modal Créer Utilisateur -->
            <div v-if="showCreateModal" class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
                <div class="bg-white rounded-lg shadow-2xl w-full max-w-md max-h-[90vh] overflow-y-auto">
                    <div class="bg-gradient-to-r from-slate-800 to-slate-700 text-white p-6 flex items-center justify-between sticky top-0 z-10">
                        <h2 class="text-xl font-bold">Créer un utilisateur</h2>
                        <button @click="showCreateModal = false" class="p-1 hover:bg-slate-600 rounded-lg transition-colors">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>

                    <form @submit.prevent="createUser" class="p-6 space-y-4">
                        <div v-if="createError" class="p-3 bg-red-100 border border-red-500 text-red-700 rounded-lg text-sm">
                            {{ createError }}
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Nom d'utilisateur *</label>
                            <input
                                v-model="newUser.username"
                                type="text"
                                placeholder="Ex: Admin"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Email *</label>
                            <input
                                v-model="newUser.email"
                                type="email"
                                placeholder="Ex: admin@example.com"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Téléphone *</label>
                            <input
                                v-model="newUser.phone"
                                type="tel"
                                placeholder="Ex: +221773423567"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Rôle *</label>
                            <select
                                v-model="newUser.role"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            >
                                <option value="">Sélectionner un rôle</option>
                                <option value="admin">Admin</option>
                                <option value="manager">Manager</option>
                                <option value="supervisor">Supervisor</option>
                            </select>
                        </div>

                        <div class="flex gap-3 pt-4">
                            <button
                                type="button"
                                @click="showCreateModal = false"
                                class="flex-1 px-4 py-2 border border-slate-300 text-slate-700 font-medium rounded-lg hover:bg-slate-50 transition-colors"
                            >
                                Annuler
                            </button>
                            <button
                                type="submit"
                                :disabled="loadingCreate"
                                class="flex-1 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-slate-400 disabled:to-slate-400 text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:cursor-not-allowed"
                            >
                                <svg v-if="loadingCreate" class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                </svg>
                                {{ loadingCreate ? 'Création...' : 'Créer' }}
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Modal Modifier Utilisateur -->
            <div v-if="showEditModal" class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
                <div class="bg-white rounded-lg shadow-2xl w-full max-w-md max-h-[90vh] overflow-y-auto">
                    <div class="bg-gradient-to-r from-slate-800 to-slate-700 text-white p-6 flex items-center justify-between sticky top-0 z-10">
                        <h2 class="text-xl font-bold">Modifier mon profil</h2>
                        <button @click="showEditModal = false" class="p-1 hover:bg-slate-600 rounded-lg transition-colors">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>

                    <form @submit.prevent="updateUser" class="p-6 space-y-4">
                        <div v-if="editError" class="p-3 bg-red-100 border border-red-500 text-red-700 rounded-lg text-sm">
                            {{ editError }}
                        </div>

                        <div v-if="editSuccess" class="p-3 bg-green-100 border border-green-500 text-green-700 rounded-lg text-sm">
                            {{ editSuccess }}
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Nom d'utilisateur *</label>
                            <input
                                v-model="editingUser.username"
                                type="text"
                                placeholder="Ex: Admin"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Email *</label>
                            <input
                                v-model="editingUser.email"
                                type="email"
                                placeholder="Ex: admin@example.com"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Téléphone *</label>
                            <input
                                v-model="editingUser.phone"
                                type="tel"
                                placeholder="Ex: +221773423567"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Password *</label>
                            <input
                                v-model="editingUser.password"
                                type="password"
                                placeholder="*********"
                                required
                                class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                        </div>

                        <div class="flex gap-3 pt-4">
                            <button
                                type="button"
                                @click="showEditModal = false"
                                class="flex-1 px-4 py-2 border border-slate-300 text-slate-700 font-medium rounded-lg hover:bg-slate-50 transition-colors"
                            >
                                Annuler
                            </button>
                            <button
                                type="submit"
                                :disabled="loadingEdit"
                                class="flex-1 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:from-slate-400 disabled:to-slate-400 text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:cursor-not-allowed"
                            >
                                <svg v-if="loadingEdit" class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                </svg>
                                {{ loadingEdit ? 'Modification...' : 'Modifier' }}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </AdminLayout>
</template>

<script>
import AdminLayout from '../../components/AdminLayout.vue'
import API_BASE_URL from '@/config/api'
import { ref, onMounted, computed } from 'vue'

export default {
  name: 'UserView',
  components: { AdminLayout },
  setup () {
    const users = ref([])
    const loading = ref(true)
    const errorMessage = ref('')
    const showCreateModal = ref(false)
    const showEditModal = ref(false)
    const loadingCreate = ref(false)
    const loadingEdit = ref(false)
    const createError = ref('')
    const editError = ref('')
    const editSuccess = ref('')
    const currentUserId = ref(null)
    const newUser = ref({
      username: '',
      email: '',
      phone: '',
      role: '',
      password: 'defaultPassword123'
    })
    const editingUser = ref({
      username: '',
      email: '',
      phone: '',
      password: ''
    })

    const adminCount = computed(() => users.value.filter(u => u.role === 'admin').length)
    const managerCount = computed(() => users.value.filter(u => u.role === 'manager').length)
    const supervisorCount = computed(() => users.value.filter(u => u.role === 'supervisor').length)

    onMounted(async () => {
      const apiKey = localStorage.getItem('api-key')
      const userId = localStorage.getItem('user-id')

      currentUserId.value = userId ? parseInt(userId) : null

      if (!apiKey) {
        errorMessage.value = 'Pas de clé API. Veuillez vous connecter.'
        loading.value = false
        return
      }

      try {
        const response = await fetch(`${API_BASE_URL}/users/`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey
          }
        })

        const data = await response.json()

        if (!response.ok) {
          errorMessage.value = data.detail || data.message || 'Erreur lors de la récupération des utilisateurs'
          loading.value = false
          return
        }

        users.value = Array.isArray(data) ? data : data.data || []
      } catch (err) {
        console.error(err)
        errorMessage.value = 'Impossible de contacter le serveur'
      } finally {
        loading.value = false
      }
    })

    const createUser = async () => {
      createError.value = ''

      if (!newUser.value.username || !newUser.value.email || !newUser.value.phone || !newUser.value.role) {
        createError.value = 'Veuillez remplir tous les champs'
        return
      }

      const apiKey = localStorage.getItem('api-key')
      if (!apiKey) {
        createError.value = 'Pas de clé API. Veuillez vous connecter.'
        return
      }

      loadingCreate.value = true

      try {
        const response = await fetch(`${API_BASE_URL}/users/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey
          },
          body: JSON.stringify(newUser.value)
        })

        const data = await response.json()

        if (!response.ok) {
          createError.value = data.detail || data.message || 'Erreur lors de la création'
          return
        }

        users.value.push(data)
        newUser.value = { username: '', email: '', phone: '', role: '', password: 'defaultPassword123' }
        showCreateModal.value = false
      } catch (err) {
        console.error(err)
        createError.value = 'Impossible de contacter le serveur'
      } finally {
        loadingCreate.value = false
      }
    }

    const openEditModal = (user) => {
      editingUser.value = {
        username: user.username,
        email: user.email,
        phone: user.phone
      }
      editError.value = ''
      editSuccess.value = ''
      showEditModal.value = true
    }

    const updateUser = async () => {
      editError.value = ''
      editSuccess.value = ''

      if (!editingUser.value.username || !editingUser.value.email || !editingUser.value.phone || !editingUser.value.password) {
        editError.value = 'Veuillez remplir tous les champs'
        return
      }

      const apiKey = localStorage.getItem('api-key')
      if (!apiKey) {
        editError.value = 'Pas de clé API. Veuillez vous connecter.'
        return
      }

      loadingEdit.value = true

      try {
        const response = await fetch(`${API_BASE_URL}/users/${currentUserId.value}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey
          },
          body: JSON.stringify(editingUser.value)
        })

        const data = await response.json()

        if (!response.ok) {
          editError.value = data.detail || data.message || 'Erreur lors de la modification'
          return
        }

        const userIndex = users.value.findIndex(u => u.id === currentUserId.value)
        if (userIndex !== -1) {
          users.value[userIndex] = { ...users.value[userIndex], ...editingUser.value }
        }

        editSuccess.value = 'Profil modifié avec succès! ✅'

        setTimeout(() => {
          showEditModal.value = false
        }, 1500)
      } catch (err) {
        console.error(err)
        editError.value = 'Impossible de contacter le serveur'
      } finally {
        loadingEdit.value = false
      }
    }

    const deleteUser = async (userId) => {
      if (!confirm('Êtes-vous sûr de vouloir supprimer cet utilisateur?')) {
        return
      }

      const apiKey = localStorage.getItem('api-key')
      if (!apiKey) {
        errorMessage.value = 'Pas de clé API. Veuillez vous connecter.'
        return
      }

      try {
        const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey
          }
        })

        if (!response.ok) {
          const data = await response.json()
          errorMessage.value = data.detail || data.message || 'Erreur lors de la suppression'
          return
        }

        users.value = users.value.filter(u => u.id !== userId)
      } catch (err) {
        console.error(err)
        errorMessage.value = 'Impossible de contacter le serveur'
      }
    }

    return {
      users,
      loading,
      errorMessage,
      adminCount,
      managerCount,
      supervisorCount,
      showCreateModal,
      showEditModal,
      loadingCreate,
      loadingEdit,
      createError,
      editError,
      editSuccess,
      newUser,
      editingUser,
      currentUserId,
      createUser,
      openEditModal,
      updateUser,
      deleteUser
    }
  }
}
</script>

<style scoped>
table {
    border-collapse: collapse;
}
</style>
