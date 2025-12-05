<template>
  <div class="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
    <!-- Éléments de décoration -->
    <div class="absolute inset-0 overflow-hidden pointer-events-none">
      <div class="absolute top-0 left-1/4 w-96 h-96 bg-green-500/10 rounded-full blur-3xl"></div>
      <div class="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl"></div>
    </div>

    <!-- Formulaire -->
    <div class="relative w-full max-w-md">
      <!-- Bouton Retour à l'accueil -->
      <router-link to="/"
        class="flex items-center gap-2 text-slate-300 hover:text-white mb-4 group transition-colors">
        <svg class="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
        </svg>
        <span>Retour à l'accueil</span>
      </router-link>

      <div class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-8 md:p-10">

        <!-- Header -->
        <div class="mb-8 text-center">
          <div class="w-16 h-16 bg-gradient-to-br from-green-400 to-emerald-500 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-green-500/50">
            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"></path>
            </svg>
          </div>
          <h2 class="text-3xl font-bold text-white mb-2">Inscription</h2>
          <p class="text-slate-300">Créez votre compte</p>
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

        <!-- Message de succès -->
        <transition name="slide">
          <div v-if="successMessage" class="mb-4 p-4 bg-green-500/20 border border-green-500/50 rounded-lg flex items-start gap-3">
            <svg class="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
            </svg>
            <span class="text-green-200 text-sm">{{ successMessage }}</span>
          </div>
        </transition>

        <!-- Formulaire -->
        <form @submit.prevent="register" class="space-y-4">
          <!-- Nom d'utilisateur -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Nom d'utilisateur</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-green-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
              </svg>
              <input
                v-model="username"
                type="text"
                placeholder="Votre nom d'utilisateur"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-green-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-4 py-2.5 transition-all duration-200"
              />
            </div>
          </div>

          <!-- Email -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Email</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-green-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
              </svg>
              <input
                v-model="email"
                type="email"
                placeholder="exemple@email.com"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-green-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-4 py-2.5 transition-all duration-200"
              />
            </div>
          </div>

          <!-- Téléphone -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Téléphone</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-green-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path>
              </svg>
              <input
                v-model="phone"
                type="tel"
                placeholder="+227 90123456"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-green-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-4 py-2.5 transition-all duration-200"
              />
            </div>
          </div>

          <!-- Password -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Mot de passe</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-green-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
              </svg>
              <input
                v-model="password"
                :type="showPassword ? 'text' : 'password'"
                placeholder="••••••••"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-green-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-10 py-2.5 transition-all duration-200"
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
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"></path>
                </svg>
              </button>
            </div>
          </div>

          <!-- Confirmer Password -->
          <div class="group">
            <label class="block text-sm font-medium text-slate-200 mb-2">Confirmer le mot de passe</label>
            <div class="relative">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-green-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
              </svg>
              <input
                v-model="confirmPassword"
                type="password"
                placeholder="••••••••"
                class="w-full bg-white/5 border border-white/20 hover:border-white/30 focus:border-green-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg pl-10 pr-4 py-2.5 transition-all duration-200"
              />
            </div>
          </div>

          <!-- Bouton Inscription -->
          <button
            type="submit"
            :disabled="loading"
            class="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg transition-all duration-200 shadow-lg shadow-green-500/50 hover:shadow-green-500/75 flex items-center justify-center gap-2 mt-6"
          >
            <svg v-if="loading" class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            <span>{{ loading ? 'Inscription en cours...' : "S'inscrire" }}</span>
          </button>
        </form>

        <!-- Footer -->
        <div class="mt-8 text-center">
          <p class="text-slate-300 text-sm">
            Déjà un compte?
            <router-link to="/login" class="text-green-400 hover:text-emerald-400 font-semibold transition-colors">
              Se connecter
            </router-link>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import API_BASE_URL from '@/config/api'

export default {
  name: 'RegisterView',
  data () {
    return {
      username: '',
      email: '',
      phone: '',
      password: '',
      confirmPassword: '',
      errorMessage: '',
      successMessage: '',
      loading: false,
      showPassword: false
    }
  },
  methods: {
    async register () {
      this.errorMessage = ''
      this.successMessage = ''

      // Validations
      if (!this.username || !this.email || !this.phone || !this.password || !this.confirmPassword) {
        this.errorMessage = 'Veuillez remplir tous les champs'
        return
      }

      if (!this.isValidEmail(this.email)) {
        this.errorMessage = 'Veuillez entrer une adresse email valide'
        return
      }

      if (this.password.length < 6) {
        this.errorMessage = 'Le mot de passe doit contenir au moins 6 caractères'
        return
      }

      if (this.password !== this.confirmPassword) {
        this.errorMessage = 'Les mots de passe ne correspondent pas'
        return
      }

      this.loading = true

      try {
        const response = await fetch(`${API_BASE_URL}/users`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username: this.username,
            email: this.email,
            phone: this.phone,
            password: this.password,
            role: 'user'
          })
        })

        const data = await response.json()

        if (!response.ok) {
          this.errorMessage = data.detail || "Erreur lors de l'inscription"
          return
        }

        this.successMessage = 'Compte créé avec succès! Redirection vers la connexion...'

        // Redirection après 2 secondes
        setTimeout(() => {
          this.$router.push('/login')
        }, 2000)
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
