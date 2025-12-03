# Guest vs Admin Access Control

## Current Access Levels

### 🌐 **Public Routes** (No Authentication Required)

| Route | Purpose | Who Can Access |
|-------|---------|----------------|
| `/` | Home page | Everyone |
| `/login` | Login page | Everyone |
| `/survey` | Customer survey form | **Guests & Admins** |

**Guests can:**
- ✅ Fill out customer survey  
- ✅ Submit feedback
- ✅ Get churn prediction results
- ❌ **Cannot** see dashboard, metrics, or models

---

### 🔒 **Admin Routes** (Authentication Required)

| Route | Purpose | Requires API Key |
|-------|---------|------------------|
| `/dashboard` | Model metrics & stats | ✅ Yes |
| `/customers-dashboard` | Customer data view | ✅ Yes |
| `/users` | User management | ✅ Yes |
| `/about` | System information | ✅ Yes |

**Admins can:**
- ✅ View all metrics and models
- ✅ Access customer data
- ✅ Manage users
- ✅ See system health
- ✅ Also fill surveys

---

## How It Works

### Router Guard Logic

```javascript
router.beforeEach((to, from, next) => {
  const publicRoutes = ['home', 'login', 'survey']
  const adminRoutes = ['dashboard', 'about', 'customers-dashboard', 'users']
  
  const token = localStorage.getItem('api-key')

  if (adminRoutes.includes(to.name) && !token) {
    // Trying to access admin page without login
    next({ name: 'login' })  // Redirect to login
  } else {
    // Public route or authenticated user
    next()  // Allow access
  }
})
```

---

## User Journeys

### **Guest User Journey:**
1. Visits website → `/`
2. Clicks "Fill Survey" → `/survey`
3. Fills form and submits
4. Gets prediction result
5. **Cannot access** `/dashboard` (redirected to login)

### **Admin User Journey:**
1. Visits website → `/`
2. Clicks "Admin Login" → `/login`
3. Enters API key
4. Can access `/dashboard`, `/customers-dashboard`, etc.
5. Can also fill `/survey` if needed

---

## API Endpoint Access

### Public Endpoint (No Auth)
```javascript
// Anyone can use this
POST /survey/submit
// No API key required
// Rate limited: 10 requests/minute per IP
```

### Admin Endpoints (Auth Required)
```javascript
// Requires API key in header
GET /models           // ← X-API-Key required
GET /model/version    // ← X-API-Key required  
GET /metrics          // ← X-API-Key required
GET /customers/infos  // ← X-API-Key required
```

---

## Security Summary

| Aspect | Guests | Admins |
|--------|--------|--------|
| **Survey Access** | ✅ Full | ✅ Full |
| **Dashboard Access** | ❌ Blocked | ✅ Full |
| **Metrics Access** | ❌ Blocked | ✅ Full |
| **Model Info** | ❌ Hidden | ✅ Full |
| **Customer Data** | ❌ Hidden | ✅ Full |
| **Rate Limiting** | 10/min | Same |

---

## Current Implementation

✅ **Already configured correctly!**

Your setup already works this way:
- Survey (`/survey`) is **public** - no API key in the code
- Dashboard requires API key check in router
- Backend `/survey/submit` endpoint is public

**No changes needed** - it's already guest-friendly! 🎉

---

## Optional: Add "Guest Mode" Label

If you want to make it clearer to users, you can add a badge:

```vue
<!-- In SurveyForm.vue header -->
<div class="flex items-center justify-between mb-6">
  <h2 class="text-3xl font-bold text-white">Customer Survey</h2>
  <span class="px-3 py-1 bg-green-500/20 text-green-400 text-xs font-semibold rounded-full">
    🌐 Public Access
  </span>
</div>
```

---

**Your system is already configured for guest access!** Guests can only use `/survey`, admins get full dashboard access.
