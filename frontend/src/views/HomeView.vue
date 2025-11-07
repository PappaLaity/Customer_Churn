<template>
  <div class="w-screen min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-y-auto">
    <!-- Background decorations -->
    <div class="fixed inset-0 overflow-hidden pointer-events-none">
      <div class="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      <div class="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl"></div>
    </div>

    <!-- Content -->
    <div class="relative w-full py-8 md:py-16 flex items-center justify-center p-4">
      <div class="max-w-2xl w-full text-center space-y-6 md:space-y-8">
        <!-- Logo/Icon -->
        <div class="flex justify-center">
          <div class="w-16 h-16 md:w-20 md:h-20 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-2xl flex items-center justify-center shadow-2xl shadow-blue-500/50">
            <svg class="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
          </div>
        </div>

        <!-- Title -->
        <div>
          <h1 class="text-3xl md:text-5xl lg:text-6xl font-bold text-white mb-2 md:mb-4">
            Customer <span class="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Churn</span> System
          </h1>
          <p class="text-base md:text-xl text-slate-300">Predict and prevent customer loss</p>
        </div>

        <!-- Description -->
        <div class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-xl md:rounded-2xl p-4 md:p-8 space-y-3 md:space-y-4">
          <h2 class="text-xl md:text-2xl font-bold text-white">What is Customer Churn?</h2>
          <p class="text-slate-200 text-sm md:text-lg leading-relaxed">
            Customer churn refers to the rate at which customers stop using your services or cancel their subscriptions.
            Our system uses advanced machine learning models to predict which customers are likely to leave,
            allowing you to take proactive measures to retain them.
          </p>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 pt-3 md:pt-4">
            <div class="bg-white/5 border border-white/10 rounded-lg p-3">
              <p class="text-cyan-400 font-semibold text-xs md:text-sm mb-1">Predict</p>
              <p class="text-slate-300 text-xs">Identify at-risk customers</p>
            </div>
            <div class="bg-white/5 border border-white/10 rounded-lg p-3">
              <p class="text-cyan-400 font-semibold text-xs md:text-sm mb-1">Analyze</p>
              <p class="text-slate-300 text-xs">Understand churn patterns</p>
            </div>
            <div class="bg-white/5 border border-white/10 rounded-lg p-3">
              <p class="text-cyan-400 font-semibold text-xs md:text-sm mb-1">Retain</p>
              <p class="text-slate-300 text-xs">Take retention actions</p>
            </div>
          </div>
        </div>

        <!-- CTA Buttons -->
        <div class="flex flex-col md:flex-row gap-3 md:gap-4 justify-center pt-4 md:pt-6">
          <router-link
            to="/dashboard"
            class="px-6 md:px-8 py-2.5 md:py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold rounded-lg transition-all duration-200 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/75 flex items-center justify-center gap-2 text-sm md:text-base"
          >
            <svg class="w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
            Go to Dashboard
          </router-link>

          <router-link
            to="/survey"
            class="px-6 md:px-8 py-2.5 md:py-3 border-2 border-cyan-500 text-cyan-400 hover:bg-cyan-500/10 font-semibold rounded-lg transition-all duration-200 hover:border-cyan-400 flex items-center justify-center gap-2 text-sm md:text-base"
          >
            <svg class="w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            Customer Survey
          </router-link>
        </div>

        <!-- Features Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4 pt-6 md:pt-8">
          <div class="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-4">
            <svg class="w-6 h-6 md:w-8 md:h-8 text-green-400 mb-2" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clip-rule="evenodd"></path>
            </svg>
            <p class="font-semibold text-white text-sm md:text-base">ML-Powered Analytics</p>
            <p class="text-xs md:text-sm text-slate-300 mt-1">Advanced models to analyze customer behavior</p>
          </div>

          <div class="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-lg p-4">
            <svg class="w-6 h-6 md:w-8 md:h-8 text-purple-400 mb-2" fill="currentColor" viewBox="0 0 20 20">
              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"></path>
            </svg>
            <p class="font-semibold text-white text-sm md:text-base">Real-Time Insights</p>
            <p class="text-xs md:text-sm text-slate-300 mt-1">Get instant predictions and detailed analytics</p>
          </div>
        </div>

        <!-- Footer -->
        <div class="text-center pt-6 md:pt-8 border-t border-white/10">
          <p class="text-slate-400 text-xs md:text-sm">
            Powered by Machine Learning • Data-Driven Insights • Reduce Churn Rate
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HomeView'
}
</script>

<style scoped>
html, body {
  margin: 0;
  padding: 0;
}
</style>
