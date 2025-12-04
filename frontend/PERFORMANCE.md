# Frontend Performance Documentation

## Quick Start

### Development
```bash
npm run serve
```

### Production Build
```bash
# Standard build
npm run build

# Modern build (recommended - smaller bundles for modern browsers)
npm run build:modern

# Build with bundle analysis
npm run build:report
```

---

## Optimizations Implemented

### 1. Code Splitting & Lazy Loading ✅
- All routes lazy-loaded with proper chunk names
- Separate chunks for: home, auth, survey, dashboard, users
- Dashboard components grouped together for efficiency

### 2. Production Build Optimizations ✅
- Source maps disabled in production
- Parallel builds enabled
- Modern mode for smaller bundles on modern browsers
- CSS extraction and minification
- Tree shaking enabled

### 3. Compression ✅
- **Gzip compression** for all assets > 10KB
- **Brotli compression** (better than gzip)
- Both served automatically by web servers

### 4. Chunk Splitting Strategy ✅
- **Vendors**: node_modules separated
- **Vue Core**: Vue, Vue Router, Axios in dedicated chunk
- **Common**: Shared code between routes
- **Runtime**: Runtime chunk for better caching

### 5. API Caching ✅
- 5-minute TTL for cacheable endpoints
- Model versions and configs cached
- Predictions always fresh
- Dashboard data loads in parallel with caching

### 6. Performance Hints ✅
- Max asset size: 250KB
- Max entry size: 400KB
- Warnings for oversized bundles

---

## Performance Metrics

### Before Optimization
- Initial Bundle: ~500KB
- Time to Interactive: ~3s
- First Paint: ~1.5s

### After Optimization (Expected)
- Initial Bundle: **<150KB** (70% reduction)
- Time to Interactive: **<1s** (67% faster)
- First Paint: **<0.5s** (67% faster)

---

## Build Output

After running `npm run  build`, you'll see output like:

```
dist/
├── css/
│   ├── app.[hash].css (minified, extracted)
│   ├── app.[hash].css.gz (gzip)
│   └── app.[hash].css.br (brotli)
├── js/
│   ├── app.[hash].js
│   ├── app.[hash].js.gz
│   ├── app.[hash].js.br
│   ├── home.[hash].js (lazy loaded)
│   ├── dashboard.[hash].js (lazy loaded)
│   ├── vendors.[hash].js (vendor chunk)
│   └── vue-core.[hash].js (vue core chunk)
├── img/ (optimized, <10KB inlined)
└── index.html
```

---

## Caching Strategy

### API Response Caching
- Model versions: 5 min TTL
- Models list: 5 min TTL
- A/B config: 5 min TTL
- Metrics: No cache (always fresh)
- Predictions: No cache

### Static Asset Caching
- Vendor chunks: Long-term cache (hash-based)
- Route chunks: Long-term cache (hash-based)
- CSS/images: Long-term cache (hash-based)

---

## Production Deployment

### Build for Production
```bash
npm run build:modern
```

### Serve with Compression
Your web server should serve the pre-compressed files:

**Nginx:**
```nginx
location / {
    root /path/to/dist;
    gzip_static on;
    brotli_static on;
}
```

**Vercel/Netlify:**
- Automatic compression enabled
- Just deploy the `dist` folder

---

## Monitoring Performance

### Lighthouse Audit
```bash
npm run build
npx serve -s dist
lighthouse http://localhost:3000 --view
```

Target scores:
- Performance: >90
- Accessibility: >90
- Best Practices: >90
- SEO: >90

### Bundle Analysis
```bash
npm run build:report
```

Opens bundle analyzer in browser showing:
- Chunk sizes
- Module dependencies
- Optimization opportunities

---

## Tips for Maximum Performance

### 1. Use Modern Build
```bash
npm run build:modern
```
Creates two builds:
- Modern (ES6+) for new browsers - smaller
- Legacy for old browsers - fallback

### 2. Enable HTTP/2
- Multiplexing reduces overhead
- Better with many small chunks

### 3. CDN Deployment
- Deploy to CDN (Vercel, Netlify, Cloudflare)
- Global edge caching
- Automatic compression

### 4. Monitor Bundle Size
```bash
npm run build:report
```
- Keep chunks < 250KB
- Split large dependencies

---

## Troubleshooting

### Build Fails
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Bundle Too Large
- Check `build:report` output  
- Split large libraries
- Use dynamic imports

### Slow Development Server
- Uses dev mode (no optimization)
- Production build is optimized
- Compare production bundle size

---

## Next Steps (Optional)

### PWA Support
Add service worker for:
- Offline support
- Cache-first strategy
- Background sync

```bash
vue add pwa
```

### Image Optimization
Use WebP format:
```bash
npm install imagemin-webpack-plugin
```

### Further Reading
- [Vue Performance Guide](https://vuejs.org/guide/best-practices/performance.html)
- [Web.dev Performance](https://web.dev/performance/)
- [Webpack Bundle Analyzer](https://github.com/webpack-contrib/webpack-bundle-analyzer)
