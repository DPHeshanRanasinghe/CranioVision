'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'

export function MetricsPanel() {
  const volumes = [
    { label: 'Total Tumor', value: '45.2' },
    { label: 'Edema', value: '28.7' },
    { label: 'Enhancing Tumor', value: '12.3' },
    { label: 'Necrotic Core', value: '4.2' },
  ]

  const dice = [
    { label: 'Mean Dice', value: '0.905' },
    { label: 'Whole Tumor Dice', value: '0.912' },
    { label: 'Tumor Core Dice', value: '0.893' },
    { label: 'Enhancing Tumor Dice', value: '0.881' },
  ]

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Volume Breakdown</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          {volumes.map((item) => (
            <div key={item.label} className="space-y-1">
              <p className="text-sm font-medium text-gray-600">{item.label}</p>
              <p className="text-2xl font-bold font-mono text-gray-900">{item.value} cm³</p>
            </div>
          ))}
        </div>

        <Separator className="bg-gray-200" />

        <div>
          <p className="mb-4 text-sm font-semibold text-gray-700">Performance vs Ground Truth</p>
          <div className="space-y-3">
            {dice.map((item) => (
              <div key={item.label} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{item.label}</span>
                <span className="font-mono font-semibold text-gray-900">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
