'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'

interface RiskRegion {
  region: string
  distance: string
  risk: 'HIGH' | 'MODERATE' | 'LOW' | 'MINIMAL'
}

export function EloquentCard() {
  const regions: RiskRegion[] = [
    { region: 'Primary Motor Cortex', distance: 'INVOLVED', risk: 'HIGH' },
    { region: 'Supplementary Motor Area', distance: '1.4 mm', risk: 'HIGH' },
    { region: 'Primary Somatosensory Cortex', distance: 'INVOLVED', risk: 'HIGH' },
    { region: "Broca's Area", distance: '12.7 mm', risk: 'LOW' },
    { region: "Wernicke's Area", distance: 'INVOLVED', risk: 'HIGH' },
    { region: 'Primary Visual Cortex', distance: 'INVOLVED', risk: 'HIGH' },
  ]

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'HIGH':
        return 'bg-red-100 text-red-800 border-red-300'
      case 'MODERATE':
        return 'bg-orange-100 text-orange-800 border-orange-300'
      case 'LOW':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'MINIMAL':
        return 'bg-green-100 text-green-800 border-green-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Eloquent Cortex Risk</CardTitle>
        <CardDescription>Surgical risk based on proximity to functional regions</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          {regions.map((item) => (
            <div key={item.region} className="flex items-center justify-between rounded-lg border border-gray-200 bg-gray-50 p-3">
              <span className="text-sm font-medium text-gray-700">{item.region}</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-600">{item.distance}</span>
                <Badge className={`rounded-full border ${getRiskColor(item.risk)} text-xs`}>{item.risk}</Badge>
              </div>
            </div>
          ))}
        </div>

        <Alert className="border-amber-200 bg-amber-50">
          <AlertCircle className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-sm text-amber-800">
            <strong>Clinical recommendation:</strong> Pre-operative functional MRI and awake craniotomy planning
            recommended for HIGH-risk regions.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  )
}
