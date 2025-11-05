<!-- <template>
    <div class="flex h-full w-full">
        <aside class="w-56 bg-gray-800 text-white flex flex-col p-4 space-y-4">
            <h2 class="text-xl font-bold mb-4">Menu</h2>
            <nav class="space-y-2">
                <ul class="space-y-4">
                    <li> <router-link to="/dashboard"
                            class="w-full text-left px-3 py-2 rounded text-gray-400 hover:bg-gray-700 hover:text-gray-200">Dashboard</router-link>
                    </li>
                    <li>
                        <router-link to="/about"
                            class=" w-full text-left px-3 py-2 rounded text-gray-400 hover:bg-gray-700 hover:text-gray-200">About</router-link>
                    </li>
                </ul>
                <button @click="logout"
                    class="w-full text-left px-3 py-2 rounded text-red-500 hover:bg-red-700 hover:text-white">
                    Déconnexion
                </button>
            </nav>

        </aside>
        <main class="flex-1 bg-gray-50 p-6 overflow-y-auto">
            <div>
                <h1 class="text-2xl font-bold mb-4">Customer Churn System</h1>
            </div>
            <slot />
        </main>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

function logout () {
  // Supprime la clé/token
  localStorage.removeItem('api-key')

  // Redirige vers login
  router.push('/login')
}
</script> -->
<template>
    <div class="flex h-full w-full">
        <!-- Overlay mobile -->
        <div v-if="sidebarOpen" class="fixed inset-0 bg-black bg-opacity-50 lg:hidden z-30" @click="sidebarOpen = false"></div>

        <!-- Sidebar -->
        <aside :class="[
            'fixed lg:static inset-y-0 left-0 z-40 w-64 md:w-72 lg:w-80 h-full bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col transition-transform duration-300 ease-in-out shadow-2xl',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        ]">
            <!-- Header Sidebar -->
            <div class="relative p-6 pb-8 border-b border-slate-700/50">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-lg flex items-center justify-center font-bold text-white shadow-lg">
                            CS
                        </div>
                        <div>
                            <h2 class="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">ChurnSys</h2>
                            <p class="text-xs text-slate-400">Dashboard</p>
                        </div>
                    </div>
                    <button @click="sidebarOpen = false" class="lg:hidden p-1 hover:bg-slate-700 rounded-lg transition-colors">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
            </div>

            <!-- Navigation -->
            <nav class="flex-1 px-3 py-6 space-y-2 overflow-y-auto">
                <p class="px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Menu</p>
                <router-link to="/dashboard"
                    :class="[
                        'flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group',
                        $route.path === '/dashboard'
                            ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/50'
                            : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                    ]">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-3m0 0l7-4 7 4M5 9v10a1 1 0 001 1h12a1 1 0 001-1V9m-9 11l4-4m0 0l4 4m-4-4v4"></path>
                    </svg>
                    <span class="font-medium">Dashboard</span>
                    <span v-if="$route.path === '/dashboard'" class="ml-auto w-2 h-2 bg-white rounded-full"></span>
                </router-link>

                <router-link to="/about"
                    :class="[
                        'flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group',
                        $route.path === '/about'
                            ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/50'
                            : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                    ]">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="font-medium">À Propos</span>
                    <span v-if="$route.path === '/about'" class="ml-auto w-2 h-2 bg-white rounded-full"></span>
                </router-link>
                <router-link to="/customers-dashboard"
                    :class="[
                        'flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group',
                        $route.path === '/customers-dashboard'
                            ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/50'
                            : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                    ]">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="font-medium">Customer Infos</span>
                    <span v-if="$route.path === '/customers-dashboard'" class="ml-auto w-2 h-2 bg-white rounded-full"></span>
                </router-link>
            </nav>

            <!-- Footer Sidebar -->
            <div class="p-3 border-t border-slate-700/50">
                <button @click="logout"
                    class="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-300 hover:bg-red-900/20 hover:text-red-400 transition-all duration-200 font-medium group">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
                    </svg>
                    <span>Déconnexion</span>
                </button>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="flex-1 flex flex-col w-full overflow-hidden">
            <!-- Header avec bouton mobile -->
            <div class="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <button @click="sidebarOpen = !sidebarOpen" class="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors">
                        <svg class="w-6 h-6 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
                        </svg>
                    </button>
                    <h1 class="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">Customer Churn System</h1>
                </div>
            </div>

            <!-- Contenu scrollable -->
            <div class="flex-1 overflow-y-auto bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-6 lg:p-8">
                <slot />
            </div>
        </main>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ref } from 'vue'

const router = useRouter()
// const route = useRoute()
const sidebarOpen = ref(false)

function logout () {
  localStorage.removeItem('api-key')
  router.push('/login')
}
</script>

<style scoped>
/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #64748b;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
