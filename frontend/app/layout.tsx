import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CranioVision — Clinical Brain Tumor Analysis',
  description:
    'AI-assisted brain tumor segmentation, anatomical analysis, and clinical risk assessment.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/*
          Fonts are served from Google's CDN at runtime, NOT bundled via
          next/font/google. The build-time fetch from `fonts.gstatic.com`
          was 404-ing on this network, leaving Next with broken font URLs.
          A runtime <link> bypasses that — the browser fetches the woff2
          files directly with its own caching + a `display=swap` fallback.
        */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
        />
      </head>
      <body className="bg-surface-page text-text-primary min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
