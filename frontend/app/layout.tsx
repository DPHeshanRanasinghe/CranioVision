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
      <body className="bg-white text-gray-900 min-h-screen">{children}</body>
    </html>
  );
}