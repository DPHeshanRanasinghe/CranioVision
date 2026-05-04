'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Toggle } from '@/components/ui/toggle'
import { Slider } from '@/components/ui/slider'
import { Zap } from 'lucide-react'

export function ViewerPanel() {
  return (
    <div className="space-y-4">
      <Card className="border border-gray-200 bg-white">
        <CardHeader>
          <CardTitle>3D Anatomical Viewer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Toggle aria-label="Toggle segmentation" defaultPressed>
              Segmentation
            </Toggle>
            <Toggle aria-label="Toggle heatmap">Grad-CAM Heatmap</Toggle>
            <Button variant="outline" size="sm" className="ml-auto">
              <Zap className="mr-2 h-4 w-4" />
              Generate
            </Button>
          </div>

          <div className="flex min-h-96 items-center justify-center rounded-lg bg-gray-800 text-center">
            <div className="space-y-2">
              <div className="text-gray-400">
                <div className="text-sm">3D Viewer Mount Point</div>
                <div className="text-xs text-gray-500">(Niivue will render here)</div>
              </div>
            </div>
          </div>

          <div className="space-y-4 rounded-lg bg-gray-50 p-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Segmentation Opacity</label>
              <Slider defaultValue={[100]} max={100} step={1} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Heatmap Opacity</label>
              <Slider defaultValue={[60]} max={100} step={1} />
            </div>
          </div>

          <p className="text-xs text-gray-600">
            Patient T1c MRI with model predictions overlaid. Use mouse to rotate, scroll to zoom.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
