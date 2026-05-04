'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Check } from 'lucide-react'

interface ProgressCardProps {
  isIdle?: boolean
}

export function ProgressCard({ isIdle = true }: ProgressCardProps) {
  const stages = ['Preprocessing', 'Inference', 'Ensemble', 'Atlas Registration', 'Anatomy', 'Done']
  const currentStageIndex = isIdle ? -1 : 2
  const progressPercent = isIdle ? 0 : 50

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Analysis Progress</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {isIdle ? (
          <div className="flex items-center justify-center rounded-lg bg-gray-50 py-12 text-center">
            <p className="text-sm text-gray-600">Awaiting upload</p>
          </div>
        ) : (
          <>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Progress value={progressPercent} className="flex-1" />
                <span className="ml-4 text-lg font-semibold text-gray-900">{progressPercent}%</span>
              </div>
              <p className="text-sm text-gray-600">
                {currentStageIndex >= 0 && currentStageIndex < stages.length
                  ? stages[currentStageIndex]
                  : 'Awaiting upload'}{' '}
                • Elapsed: 2m 34s
              </p>
            </div>

            <div className="space-y-2">
              {stages.map((stage, idx) => (
                <div key={stage} className="flex items-center gap-2">
                  <div
                    className={`flex h-5 w-5 items-center justify-center rounded-full ${
                      idx <= currentStageIndex
                        ? 'bg-blue-600 text-white'
                        : 'border border-gray-300 bg-white text-gray-300'
                    }`}
                  >
                    {idx <= currentStageIndex && <Check className="h-3 w-3" />}
                  </div>
                  <span className="text-sm text-gray-700">{stage}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
