<template>
  <div
    class="w-screen min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
    <!-- Éléments de décoration -->
    <div class="absolute inset-0 overflow-hidden pointer-events-none">
      <div class="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      <div class="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl"></div>
    </div>

    <!-- Formulaire -->
    <div class="relative w-full max-w-2xl max-h-[90vh] flex flex-col">
      <div
        class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-6 md:p-8 flex flex-col h-full">

        <!-- Header fixe -->
        <div class="mb-6 pb-4 border-b border-white/20 flex-shrink-0">
          <h2 class="text-2xl md:text-3xl font-bold text-white mb-1">Customer Survey</h2>
          <p class="text-slate-300 text-sm">Aidez-nous à améliorer notre service</p>
        </div>

        <!-- Message de succès -->
        <transition name="slide">
          <div v-if="successMessage"
            class="mb-4 p-3 bg-green-500/20 border border-green-500/50 rounded-lg flex items-start gap-2 flex-shrink-0">
            <svg class="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clip-rule="evenodd"></path>
            </svg>
            <div>
              <p class="font-semibold text-green-400 text-sm">Succès!</p>
              <p class="text-green-200 text-xs">{{ successMessage }}</p>
            </div>
          </div>
        </transition>

        <!-- Message d'erreur -->
        <transition name="slide">
          <div v-if="errorMessage"
            class="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-2 flex-shrink-0">
            <svg class="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clip-rule="evenodd"></path>
            </svg>
            <div>
              <p class="font-semibold text-red-400 text-sm">Erreur</p>
              <p class="text-red-200 text-xs">{{ errorMessage }}</p>
            </div>
          </div>
        </transition>

        <!-- Contenu scrollable -->
        <form @submit.prevent="send" class="flex-1 overflow-y-auto space-y-4 pr-2">
          <!-- Ligne 1: Nom et Email -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Nom *</label>
              <input v-model="formData.name" type="text" placeholder="Votre nom" required
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200" />
            </div>
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Email *</label>
              <input v-model="formData.email" type="email" placeholder="exemple@email.com" required
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200" />
            </div>
          </div>

          <!-- Ligne 2: Ancienneté et Service Internet -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Ancienneté (mois)</label>
              <input v-model.number="formData.tenure" type="number" placeholder="0" min="0"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200" />
            </div>
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Service Internet</label>
              <select v-model="formData.internetService"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white rounded-lg px-3 py-2 text-sm transition-all duration-200">
                <option value="" class="bg-slate-800">Sélectionner</option>
                <option value="Fiber optic" class="bg-slate-800">Fibre optique</option>
                <option value="DSL" class="bg-slate-800">DSL</option>
                <option value="Cable" class="bg-slate-800">Câble</option>
                <option value="None" class="bg-slate-800">Aucun</option>
              </select>
            </div>
          </div>

          <!-- Ligne 3: Contrat et Paiement -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Type de Contrat</label>
              <select v-model="formData.contract"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white rounded-lg px-3 py-2 text-sm transition-all duration-200">
                <option value="" class="bg-slate-800">Sélectionner</option>
                <option value="Month-to-month" class="bg-slate-800">Mois par mois</option>
                <option value="One year" class="bg-slate-800">1 an</option>
                <option value="Two year" class="bg-slate-800">2 ans</option>
              </select>
            </div>
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Méthode de Paiement</label>
              <select v-model="formData.paymentMethod"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white rounded-lg px-3 py-2 text-sm transition-all duration-200">
                <option value="" class="bg-slate-800">Sélectionner</option>
                <option value="Electronic check" class="bg-slate-800">Chèque électronique</option>
                <option value="Mailed check" class="bg-slate-800">Chèque envoyé</option>
                <option value="Bank transfer" class="bg-slate-800">Virement bancaire</option>
                <option value="Credit card" class="bg-slate-800">Carte de crédit</option>
              </select>
            </div>
          </div>

          <!-- Ligne 4: Frais -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Frais Mensuels ($)</label>
              <input v-model.number="formData.monthlyCharges" type="number" placeholder="0" min="0" step="0.01"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200" />
            </div>
            <div class="group">
              <label class="block text-xs font-medium text-slate-200 mb-1">Frais Totaux ($)</label>
              <input v-model.number="formData.totalCharges" type="number" placeholder="0" min="0" step="0.01"
                class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200" />
            </div>
          </div>

          <!-- Checkboxes -->
          <div class="space-y-2 pt-2">
            <label class="flex items-center gap-2 cursor-pointer group">
              <input v-model="formData.paperlessBilling" type="checkbox"
                class="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-600 focus:ring-blue-500 cursor-pointer" />
              <span class="text-slate-300 text-sm group-hover:text-white transition-colors">Facturation sans
                papier</span>
            </label>

            <label class="flex items-center gap-2 cursor-pointer group">
              <input v-model="formData.noInternetService" type="checkbox"
                class="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-600 focus:ring-blue-500 cursor-pointer" />
              <span class="text-slate-300 text-sm group-hover:text-white transition-colors">Aucun service
                internet</span>
            </label>
          </div>

          <!-- Message/Commentaires -->
          <div class="group">
            <label class="block text-xs font-medium text-slate-200 mb-1">Commentaires (optionnel)</label>
            <textarea v-model="formData.message" placeholder="Partagez votre avis..." rows="3"
              class="w-full bg-white/5 border border-white/20 focus:border-blue-500 focus:bg-white/10 focus:outline-none text-white placeholder-slate-400 rounded-lg px-3 py-2 text-sm transition-all duration-200 resize-none"></textarea>
          </div>
        </form>

        <!-- Bouton fixe en bas -->
        <div class="mt-4 pt-4 border-t border-white/20 flex-shrink-0">
          <button @click="send" :disabled="loading"
            class="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed text-white font-semibold py-2 rounded-lg transition-all duration-200 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/75 flex items-center justify-center gap-2 text-sm">
            <svg v-if="loading" class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15">
              </path>
            </svg>
            <span>{{ loading ? 'Envoi...' : 'Envoyer' }}</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import API_BASE_URL from '@/config/api'
export default {
  name: 'CustomerSurvey',
  data () {
    return {
      formData: {
        name: '',
        email: '',
        tenure: 0,
        internetService: '',
        contract: '',
        paymentMethod: '',
        monthlyCharges: 0,
        totalCharges: 0,
        message: '',
        paperlessBilling: false,
        noInternetService: false
      },
      loading: false,
      errorMessage: '',
      successMessage: ''
    }
  },
  methods: {
    mapFormToAPI () {
      return {
        tenure: this.formData.tenure,
        InternetService_Fiber_optic: this.formData.internetService === 'Fiber optic',
        Contract_Two_year: this.formData.contract === 'Two year',
        PaymentMethod_Electronic_check: this.formData.paymentMethod === 'Electronic check',
        No_internet_service: this.formData.noInternetService ? 1 : 0,
        TotalCharges: parseFloat(this.formData.totalCharges),
        MonthlyCharges: parseFloat(this.formData.monthlyCharges),
        PaperlessBilling: this.formData.paperlessBilling ? 1 : 0
      }
    },
    async send () {
      this.errorMessage = ''
      this.successMessage = ''

      if (!this.formData.name || !this.formData.email) {
        this.errorMessage = 'Veuillez remplir les champs obligatoires (Nom, Email)'
        return
      }

      if (this.formData.email && !this.isValidEmail(this.formData.email)) {
        this.errorMessage = 'Veuillez entrer une adresse email valide'
        return
      }

      const payload = this.mapFormToAPI()
      console.log("Payload envoyé à l'API:", payload)
      this.loading = true

      // console.log('Sending survey data:', this.formData)

      try {
        const response = await fetch(`${API_BASE_URL}/survey/submit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })

        const data = await response.json()

        if (!response.ok) {
          this.errorMessage = data.detail || data.message || 'Erreur lors de l\'envoi du sondage'
          return
        }

        this.successMessage = data.message || 'Survey succesfully Send!'

        setTimeout(() => {
          this.resetForm()
        }, 2000)
      } catch (err) {
        console.error(err)
        this.errorMessage = 'Impossible de contacter le serveur'
      } finally {
        this.loading = false
      }
    },
    resetForm () {
      this.formData = {
        name: '',
        email: '',
        tenure: 0,
        internetService: '',
        contract: '',
        paymentMethod: '',
        monthlyCharges: 0,
        totalCharges: 0,
        message: '',
        paperlessBilling: false,
        noInternetService: false
      }
      this.successMessage = ''
    },
    isValidEmail (email) {
      const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      return re.test(email)
    }
  }
}
</script>

<style scoped>
.slide-enter-active,
.slide-leave-active {
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

::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.3);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.5);
}
</style>
