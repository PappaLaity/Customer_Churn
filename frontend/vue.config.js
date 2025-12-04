const { defineConfig } = require('@vue/cli-service')
const CompressionPlugin = require('compression-webpack-plugin')

module.exports = defineConfig({
  transpileDependencies: [],

  // Production optimizations
  productionSourceMap: false, // Disable source maps in production (smaller bundle)

  // Parallel build for faster compilation
  parallel: true,

  // CSS extraction and optimization
  css: {
    extract: {
      ignoreOrder: true // Suppress order warnings
    },
    sourceMap: false // No CSS source maps
  },

  devServer: {
    proxy: {
      '/api': {
        target: 'http://customer-churn.francecentral.cloudapp.azure.com:8000',
        changeOrigin: true,
        secure: false,
        pathRewrite: { '^/api': '' }, // supprime /api avant d’envoyer au backend
      },
    },
  },

  // Webpack configuration
  configureWebpack: config => {
    if (process.env.NODE_ENV === 'production') {
      // Compression plugins
      config.plugins.push(
        // Gzip compression
        new CompressionPlugin({
          filename: '[path][base].gz',
          algorithm: 'gzip',
          test: /\.(js|css|html|svg)$/,
          threshold: 10240, // Only compress files > 10KB
          minRatio: 0.8
        }),
        // Brotli compression (better than gzip)
        new CompressionPlugin({
          filename: '[path][base].br',
          algorithm: 'brotliCompress',
          test: /\.(js|css|html|svg)$/,
          threshold: 10240,
          minRatio: 0.8
        })
      )
    }

    // Performance hints
    config.performance = {
      hints: 'warning',
      maxAssetSize: 250000, // 250KB per asset
      maxEntrypointSize: 400000 // 400KB total entry
    }
  },

  // Chain webpack for advanced optimization
  chainWebpack: config => {
    // Split chunks for optimal caching
    if (process.env.NODE_ENV === 'production') {
      config.optimization.splitChunks({
        chunks: 'all',
        cacheGroups: {
          // Vendor chunks (node_modules)
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 10,
            reuseExistingChunk: true
          },
          // Common code shared between routes
          common: {
            minChunks: 2,
            priority: 5,
            reuseExistingChunk: true,
            enforce: true
          },
          // Vue and core libraries
          vue: {
            test: /[\\/]node_modules[\\/](vue|vue-router|axios)[\\/]/,
            name: 'vue-core',
            priority: 20
          }
        }
      })

      // Runtime chunk for better caching
      config.optimization.runtimeChunk('single')

      // Minimize and tree-shake
      config.optimization.minimize(true)
    }
  }
})
