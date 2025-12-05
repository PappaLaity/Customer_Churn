import { createRouter, createWebHistory } from 'vue-router'

// Lazy load all components for optimal code splitting
const HomeView = () => import(/* webpackChunkName: "home" */ '../views/HomeView.vue')
const LoginView = () => import(/* webpackChunkName: "auth" */ '../views/LoginView.vue')
const RegisterView = () => import(/* webpackChunkName: "auth" */ '../views/RegisterView.vue')
const SurveyForm = () => import(/* webpackChunkName: "survey" */ '../views/SurveyForm.vue')
const AboutView = () => import(/* webpackChunkName: "about" */ '../views/AboutView.vue')
const DashboardView = () => import(/* webpackChunkName: "dashboard" */ '../views/DashboardView.vue')
const CustomerInfos = () => import(/* webpackChunkName: "dashboard" */ '../views/CustomerInfos.vue')
const UserView = () => import(/* webpackChunkName: "users" */ '../views/users/UserView.vue')
const MakePredictions = () => import(/* webpackChunkName: "predictions" */ '../views/MakePredictions.vue')

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/login',
    name: 'login',
    component: LoginView
  },
  {
    path: '/register',
    name: 'register',
    component: RegisterView
  },
  {
    path: '/survey',
    name: 'survey',
    component: SurveyForm
  },
  {
    path: '/about',
    name: 'about',
    component: AboutView
  },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: DashboardView,
    // Prefetch dashboard for faster navigation after login
    meta: { prefetch: true }
  },
  {
    path: '/customers-dashboard',
    name: 'customers-dashboard',
    component: CustomerInfos,
    meta: { prefetch: true }
  },
  {
    path: '/users',
    name: 'users',
    component: UserView
  },
  {
    path: '/predictions',
    name: 'predictions',
    component: MakePredictions,
    meta: { prefetch: true }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

router.beforeEach((to, from, next) => {
  // Admin routes (require API key authentication)
  const adminRoutes = ['dashboard', 'about', 'customers-dashboard', 'users', 'predictions']

  const token = localStorage.getItem('api-key')

  if (adminRoutes.includes(to.name) && !token) {
    // Admin route without auth → redirect to login
    next({ name: 'login' })
  } else {
    // Public route or authenticated → allow access
    next()
  }
})

export default router
