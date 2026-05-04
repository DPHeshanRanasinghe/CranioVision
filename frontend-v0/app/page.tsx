import { CranioVisionHeader } from '@/components/craniov-header'
import { UploadCard } from '@/components/craniov-upload-card'
import { ProgressCard } from '@/components/craniov-progress-card'
import { ModelPickerTabs } from '@/components/craniov-model-picker'
import { ViewerPanel } from '@/components/craniov-viewer-panel'
import { MetricsPanel } from '@/components/craniov-metrics-panel'
import { AnatomyCard } from '@/components/craniov-anatomy-card'
import { EloquentCard } from '@/components/craniov-eloquent-card'
import { AgreementBanner } from '@/components/craniov-agreement-banner'

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <CranioVisionHeader />

      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="space-y-8">
          {/* Upload and Progress Row */}
          <div className="grid gap-6 md:grid-cols-2">
            <UploadCard />
            <ProgressCard isIdle={true} />
          </div>

          {/* Model Picker */}
          <ModelPickerTabs />

          {/* Viewer and Metrics Row */}
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <ViewerPanel />
            </div>
            <MetricsPanel />
          </div>

          {/* Anatomy and Eloquent Row */}
          <div className="grid gap-6 md:grid-cols-2">
            <AnatomyCard />
            <EloquentCard />
          </div>

          {/* Agreement Banner */}
          <AgreementBanner />

          {/* Footer */}
          <footer className="border-t border-gray-200 pt-8 text-center text-xs text-gray-600">
            <p>
              CranioVision · Research use only — not for clinical diagnosis · University of Moratuwa · 2026 ·{' '}
              <span className="font-mono">v1.0.0</span>
            </p>
          </footer>
        </div>
      </main>
    </div>
  )
}
