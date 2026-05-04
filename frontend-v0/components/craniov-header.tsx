'use client'

import Link from 'next/link'

export function CranioVisionHeader() {
  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-lg bg-blue-600" />
              <h1 className="font-serif text-2xl font-semibold text-gray-900">CranioVision</h1>
            </div>
            <p className="text-sm text-gray-600">AI-assisted brain tumor segmentation and clinical analysis</p>
          </div>
          <nav className="flex gap-6">
            <Link href="#" className="text-sm text-gray-600 hover:text-gray-900">
              Documentation
            </Link>
            <Link href="#" className="text-sm text-gray-600 hover:text-gray-900">
              GitHub
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
