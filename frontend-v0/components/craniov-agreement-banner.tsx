'use client'

import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Download, Brain } from 'lucide-react'

export function AgreementBanner() {
  return (
    <Card className="border border-gray-200 bg-white">
      <CardContent className="p-8">
        <div className="grid gap-8 md:grid-cols-2">
          <div className="flex flex-col items-center justify-center border-r border-gray-200 text-center">
            <p className="text-4xl font-bold text-gray-900">98.4%</p>
            <p className="mt-2 text-sm font-medium text-gray-600">Multi-model Unanimous</p>
          </div>
          <div className="space-y-4">
            <p className="text-sm text-gray-700">
              All 3 architectures agreed on 98.4% of voxels. Disagreement concentrated at tumor boundaries —
              radiologist review recommended at edge regions.
            </p>
            <div className="flex gap-3 pt-2">
              <Button className="bg-blue-600 hover:bg-blue-700">
                <Download className="mr-2 h-4 w-4" />
                Download Clinical Report (PDF)
              </Button>
              <Button variant="outline">
                <Brain className="mr-2 h-4 w-4" />
                Generate XAI Explanation
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
