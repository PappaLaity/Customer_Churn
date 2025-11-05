<!-- <template>
  <div class="w-screen h-screen flex items-center justify-center bg-gray-100">
    <div class="bg-white p-6 rounded-lg shadow-lg w-full max-w-md">

      <h2 class="text-xl font-bold mb-4 text-center">Login</h2>

      <div v-if="errorMessage" class="mb-3 text-red-600 text-sm">
        {{ errorMessage }}
      </div>

      <input
        v-model="email"
        type="email"
        placeholder="Email"
        class="w-full border px-3 py-2 rounded mb-3"
      />

      <input
        v-model="password"
        type="password"
        placeholder="Password"
        class="w-full border px-3 py-2 rounded mb-3"
      />

      <button
        @click="login"
        class="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 mb-2"
      >
        Connexion
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LoginForm',
  data () {
    return {
      email: '',
      password: '',
      errorMessage: ''
    }
  },
  methods: {
    async login () {
      this.errorMessage = ''

      if (!this.email || !this.password) {
        this.errorMessage = 'Please fill in all fields'
        return
      }

      try {
        // Remplace par ton endpoint réel
        // const response = await fetch('https://api.example.com/login', {
        const response = await fetch('http://localhost:8000/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: this.email, password: this.password })
        })

        const data = await response.json()

        if (!response.ok) {
          // Affiche le message d'erreur renvoyé par l'API
          this.errorMessage = data.detail || 'Erreur de connexion'
          return
        }

        // Succès : stocke la clé/token dans localStorage
        localStorage.setItem('api-key', data.api_key)
        alert('Connexion réussie ✅')

        // Rediriger vers le dashboard si nécessaire
        this.$router.push('/dashboard')
      } catch (err) {
        console.error(err)
        this.errorMessage = 'Impossible de contacter le serveur'
      }
    }
  }
}
</script> -->
<template>
  <div class="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
    <!-- Éléments de décoration -->
    <div class="absolute inset-0 overflow-hidden pointer-events-none">
      <div class="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      <div class="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl"></div>
    </div>

    <!-- Formulaire -->
    <div class="relative w-full max-w-md">
      <div class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-8 md:p-10">

        <!-- Header -->
        <div class="mb-8 text-center">
          <div class="w-16 h-16 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-500/50">
            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <h2 class="text-3xl font-bold text-white mb-2">Bienvenue</h2>
          <p class="text-slate-300">Connectez-vous à votre compte</p>
        </div>

        <!-- Messages d'erreur -->
        <transition name="slide">
          <div v-if="errorMessage" class="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
            <svg class="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
            <span class="text-red-200 text-sm">{{ errorMessage }}</span>
          </div>
        </transition>

        <!-- Formulaire -->
        <form @submit.prevent="login" class="space-y-4">
          <!-- Email -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Email</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-blue-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
              </svg>
              <input
                v-model="email"
                type="email"
                placeholder="exemple@email.com"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-4 py-2.5 transition-all duration-200"
              />
            </div>
          </div>

          <!-- Password -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Mot de passe</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-blue-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
              </svg>
              <input
                v-model="password"
                :type="showPassword ? 'text' : 'password'"
                placeholder="••••••••"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-10 py-2.5 transition-all duration-200"
              />
              <button
                type="button"
                @click="showPassword = !showPassword"
                class="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
              >
                <svg v-if="!showPassword" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                </svg>
                <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-4.803m5.596-3.856a3.375 3.375 0 11-4.753 4.753m4.753-4.753L3.596 3.596"></path>
                </svg>
              </button>
            </div>
          </div>

          <!-- Bouton Connexion -->
          <button
            type="submit"
            :disabled="loading"
            class="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg transition-all duration-200 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/75 flex items-center justify-center gap-2 mt-6"
          >
            <svg v-if="loading" class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            <span>{{ loading ? 'Connexion en cours...' : 'Se connecter' }}</span>
          </button>
        </form>

        <!-- Footer -->
        <div class="mt-8 text-center">
          <p class="text-slate-300 text-sm">
            Pas encore de compte?
            <router-link to="/register" class="text-blue-400 hover:text-cyan-400 font-semibold transition-colors">
              S'inscrire
            </router-link>
          </p>
        </div>
      </div>

      <!-- Support text -->
      <p class="text-center text-slate-400 text-xs mt-6">
        Besoin d'aide? <a href="#" class="text-blue-400 hover:text-cyan-400 transition-colors">Contactez le support</a>
      </p>
    </div>
  </div>
</template>

<script>
import API_BASE_URL from '@/config/api'

export default {
  name: 'LoginForm',
  data () {
    return {
      email: '',
      password: '',
      errorMessage: '',
      loading: false,
      showPassword: false
    }
  },
  methods: {
    async login () {
      this.errorMessage = ''

      if (!this.email || !this.password) {
        this.errorMessage = 'Veuillez remplir tous les champs'
        return
      }

      if (this.email && !this.isValidEmail(this.email)) {
        this.errorMessage = 'Veuillez entrer une adresse email valide'
        return
      }

      this.loading = true

      try {
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: this.email, password: this.password })
        })

        const data = await response.json()

        if (!response.ok) {
          this.errorMessage = data.detail || 'Erreur de connexion'
          return
        }

        localStorage.setItem('api-key', data.api_key)

        this.$router.push('/dashboard')
      } catch (err) {
        console.error(err)
        this.errorMessage = 'Impossible de contacter le serveur'
      } finally {
        this.loading = false
      }
    },
    isValidEmail (email) {
      const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      return re.test(email)
    }
  }
}
</script>

<style scoped>
.slide-enter-active, .slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
