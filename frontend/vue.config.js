const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
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
})
