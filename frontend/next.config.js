/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Niivue's daikon dependency tries to use Node 'fs' module which doesn't
  // exist in the browser. Tell webpack to stub it out.
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },

  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backend}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;